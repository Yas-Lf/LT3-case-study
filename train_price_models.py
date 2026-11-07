"""Fit horizon price models and the XGBoost and LSTM comparisons.

CRZY and TAME are fit separately. Validation chooses the ridge penalty and
when the tree and LSTM stop. The test runs are scored once after that.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "price_features.sqlite"
OUT_PATH = ROOT / "price_model_results.sqlite"

from build_price_features import FEATURES

LSTM_FEATURES = [
    "ret_1",
    "spread",
    "imb_touch",
    "micro_gap",
    "signed_vol_1",
    "imb_5c",
    "rvol_5",
    "dist_start",
]
HORIZONS = (1, 5, 31)
TICKERS = ("CRZY", "TAME")
TRUSTED = {1: "zero", 5: "hgb", 31: "ridge"}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
MEAN_MODELS = ("zero", "ridge", "hgb", "xgb", "lstm")
TREE_PARAMS = {
    "max_iter": 100,
    "max_depth": 3,
    "min_samples_leaf": 200,
    "learning_rate": 0.05,
    "random_state": 0,
}


def mae(y, pred) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def pinball(y, pred, quantile: float) -> float:
    error = y - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def run_scores(y, pred, run_ids) -> dict[str, float]:
    scores = {}
    for run_id in pd.unique(run_ids):
        mask = run_ids == run_id
        scores[str(run_id)] = mae(y[mask], pred[mask])
    return scores


def fraction_better(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    wins = sum(candidate[run] < baseline[run] for run in candidate)
    return wins / len(candidate)


class SmallLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(LSTM_FEATURES), hidden_size=16, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(16, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequences)
        return self.head(self.drop(output[:, -1, :])).squeeze(-1)


def load_panel() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{SRC_PATH}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            """
            SELECT p.*, r.split
            FROM panel p
            JOIN runs r USING (run_id)
            """,
            con,
        )
    finally:
        con.close()
    return frame


def impute_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train", FEATURES]
    medians = train.median(numeric_only=True).fillna(0.0)
    filled = frame.copy()
    filled[FEATURES] = filled[FEATURES].fillna(medians)
    return filled


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return tuple((block - center) / scale for block in (train, *others))


def window_matrix(values, all_runs, all_ticks, target_runs, target_ticks):
    lookup = {(run, int(tick)): index for index, (run, tick) in enumerate(zip(all_runs, all_ticks))}
    windows = np.empty((len(target_ticks), 31, values.shape[1]), dtype=np.float64)
    keep = np.ones(len(target_ticks), dtype=bool)
    for row, (run, tick) in enumerate(zip(target_runs, target_ticks)):
        positions = []
        for offset in range(int(tick) - 30, int(tick) + 1):
            found = lookup.get((run, offset))
            if found is None:
                keep[row] = False
                break
            positions.append(found)
        if keep[row]:
            windows[row] = values[positions]
    return windows, keep


def fit_ridge(x_train, y_train, x_val, y_val):
    best = None
    best_mae = np.inf
    best_alpha = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        score = mae(y_val, model.predict(x_val))
        if score < best_mae:
            best = model
            best_mae = score
            best_alpha = alpha
    return best, f"alpha={best_alpha}"


def fit_hgb(x_train, y_train, loss: str = "squared_error", quantile: float | None = None):
    kwargs = dict(TREE_PARAMS)
    kwargs["loss"] = loss
    if quantile is not None:
        kwargs["quantile"] = quantile
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(x_train, y_train)
    return model


def fit_xgb(x_train, y_train, x_val, y_val, objective: str = "reg:squarederror", quantile: float | None = None):
    kwargs = {
        "n_estimators": 100,
        "max_depth": 3,
        "min_child_weight": 200,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "objective": objective,
        "random_state": 0,
        "n_jobs": -1,
        "early_stopping_rounds": 10,
        "eval_metric": "mae",
    }
    if quantile is not None:
        kwargs["quantile_alpha"] = quantile
    model = XGBRegressor(**kwargs)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model, f"best_iteration={int(model.best_iteration)}"


def fit_lstm(x_train, y_train, x_val, y_val):
    torch.manual_seed(0)
    model = SmallLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )
    val_x = torch.as_tensor(x_val, dtype=torch.float32)
    best_state = None
    best_mae = np.inf
    best_epoch = 0
    waits = 0
    epochs_run = 0
    for epoch in range(1, 21):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_x).numpy()
        score = mae(y_val, pred)
        epochs_run = epoch
        if score < best_mae - 1e-8:
            best_mae = score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            waits = 0
        else:
            waits += 1
            if waits >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, f"epochs={epochs_run},best={best_epoch}"


def predict_lstm(model: SmallLSTM, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(sequences), 512):
            batch = torch.as_tensor(sequences[start : start + 512], dtype=torch.float32)
            pieces.append(model(batch).numpy())
    return np.concatenate(pieces)


def metric_row(model, ticker, horizon, split, y, pred, run_ids, detail, pinball_q=None, band_pred=None):
    zero = np.zeros(len(y))
    by_model = run_scores(y, pred, run_ids)
    by_zero = run_scores(y, zero, run_ids)
    row = {
        "model": model,
        "ticker": ticker,
        "horizon": horizon,
        "split": split,
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "zero_mae": mae(y, zero),
        "zero_rmse": rmse(y, zero),
        "frac_runs_beat_zero": fraction_better(by_model, by_zero),
        "pinball": None,
        "band_pinball": None,
        "detail": detail,
    }
    if pinball_q is not None:
        row["pinball"] = pinball(y, pred, pinball_q)
        row["band_pinball"] = pinball(y, band_pred, pinball_q)
    return row, by_model


def fit_ticker(frame: pd.DataFrame, ticker: str):
    part = frame.loc[frame["ticker"] == ticker].sort_values(["run_id", "case_tick"]).reset_index(drop=True)
    observed_ret_31 = part["ret_31"].notna()
    part = impute_ticker(part)
    lstm_train = part.loc[part["split"] == "train", LSTM_FEATURES].to_numpy(dtype=float)
    center = lstm_train.mean(axis=0)
    scale = lstm_train.std(axis=0)
    scale[scale == 0] = 1.0
    lstm_values = (part[LSTM_FEATURES].to_numpy(dtype=float) - center) / scale
    metrics = []
    predictions = []
    test_predictions = []
    comparisons = []

    for horizon in HORIZONS:
        label = f"y_{horizon}"
        eligible = part.loc[observed_ret_31 & part[label].notna()].copy()
        windows, keep = window_matrix(
            lstm_values,
            part["run_id"].to_numpy(),
            part["case_tick"].to_numpy(),
            eligible["run_id"].to_numpy(),
            eligible["case_tick"].to_numpy(),
        )
        if not keep.all():
            print(f"{ticker} y_{horizon}: dropped {int((~keep).sum())} rows with a broken window", file=sys.stderr)
            eligible = eligible.iloc[np.flatnonzero(keep)].copy()
            windows = windows[keep]
        y = eligible[label].to_numpy(dtype=float)
        runs = eligible["run_id"].to_numpy()
        raw = eligible[FEATURES].to_numpy(dtype=float)
        split = eligible["split"].to_numpy()
        train = split == "train"
        valid = split == "validation"
        test = split == "test"
        x_train, x_val, x_test = standardize(raw[train], raw[valid], raw[test])
        print(
            f"{ticker} y_{horizon}: train {train.sum()} validation {valid.sum()} test {test.sum()}",
            file=sys.stderr,
        )

        ridge, ridge_detail = fit_ridge(x_train, y[train], x_val, y[valid])
        hgb = fit_hgb(raw[train], y[train])
        xgb, xgb_detail = fit_xgb(raw[train], y[train], raw[valid], y[valid])
        lstm, lstm_detail = fit_lstm(windows[train], y[train], windows[valid], y[valid])
        details = {
            "zero": "",
            "ridge": ridge_detail,
            "hgb": "max_iter=100,max_depth=3,min_samples_leaf=200",
            "xgb": xgb_detail,
            "lstm": lstm_detail,
        }
        fitted = {
            "ridge": lambda block, scaled=None: ridge.predict(scaled),
            "hgb": lambda block, scaled=None: hgb.predict(block),
            "xgb": lambda block, scaled=None: xgb.predict(block),
            "lstm": lambda block, scaled=None, seq=None: predict_lstm(lstm, seq),
        }

        split_preds = {}
        blocks = {"train": (raw[train], x_train, windows[train]), "validation": (raw[valid], x_val, windows[valid]), "test": (raw[test], x_test, windows[test])}
        masks = {"train": train, "validation": valid, "test": test}
        for name in ("validation", "test"):
            block, scaled, seq = blocks[name]
            y_split = y[masks[name]]
            run_split = runs[masks[name]]
            preds = {
                "zero": np.zeros(len(y_split)),
                "ridge": fitted["ridge"](block, scaled),
                "hgb": fitted["hgb"](block),
                "xgb": fitted["xgb"](block),
                "lstm": fitted["lstm"](block, seq=seq),
            }
            split_preds[name] = preds
            for model_name in MEAN_MODELS:
                row, _ = metric_row(
                    model_name, ticker, horizon, name, y_split, preds[model_name], run_split, details[model_name]
                )
                metrics.append(row)
        val_preds = split_preds["validation"]
        test_preds = split_preds["test"]

        if horizon == 31:
            band_low = float(np.quantile(y[train], 0.1))
            band_high = float(np.quantile(y[train], 0.9))
            quantile_models = {
                "hgb_q10": fit_hgb(raw[train], y[train], loss="quantile", quantile=0.1),
                "hgb_q90": fit_hgb(raw[train], y[train], loss="quantile", quantile=0.9),
            }
            xgb_q10, xgb_q10_detail = fit_xgb(
                raw[train], y[train], raw[valid], y[valid], objective="reg:quantileerror", quantile=0.1
            )
            xgb_q90, xgb_q90_detail = fit_xgb(
                raw[train], y[train], raw[valid], y[valid], objective="reg:quantileerror", quantile=0.9
            )
            quantile_models["xgb_q10"] = xgb_q10
            quantile_models["xgb_q90"] = xgb_q90
            q_details = {
                "hgb_q10": "quantile=0.1",
                "hgb_q90": "quantile=0.9",
                "xgb_q10": xgb_q10_detail,
                "xgb_q90": xgb_q90_detail,
                "band_q10": f"constant={band_low}",
                "band_q90": f"constant={band_high}",
            }
            for name in ("validation", "test"):
                block = raw[masks[name]]
                y_split = y[masks[name]]
                run_split = runs[masks[name]]
                q_preds = {
                    "hgb_q10": quantile_models["hgb_q10"].predict(block),
                    "hgb_q90": quantile_models["hgb_q90"].predict(block),
                    "xgb_q10": quantile_models["xgb_q10"].predict(block),
                    "xgb_q90": quantile_models["xgb_q90"].predict(block),
                    "band_q10": np.full(len(y_split), band_low),
                    "band_q90": np.full(len(y_split), band_high),
                }
                split_preds[name].update(q_preds)
                if name == "validation":
                    val_preds = split_preds[name]
                else:
                    test_preds = split_preds[name]
                for model_name, pred in q_preds.items():
                    quantile = 0.1 if model_name.endswith("q10") else 0.9
                    band = q_preds["band_q10"] if quantile == 0.1 else q_preds["band_q90"]
                    row, _ = metric_row(
                        model_name,
                        ticker,
                        horizon,
                        name,
                        y_split,
                        pred,
                        run_split,
                        q_details[model_name],
                        pinball_q=quantile,
                        band_pred=band,
                    )
                    metrics.append(row)

        trusted_name = TRUSTED[horizon]
        trusted_by_run = run_scores(y[valid], val_preds[trusted_name], runs[valid])
        comparisons.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "trusted": trusted_name,
                "trusted_mae": mae(y[valid], val_preds[trusted_name]),
                "zero_mae": mae(y[valid], np.zeros(valid.sum())),
                "xgb_mae": mae(y[valid], val_preds["xgb"]),
                "lstm_mae": mae(y[valid], val_preds["lstm"]),
                "xgb_beats": fraction_better(run_scores(y[valid], val_preds["xgb"], runs[valid]), trusted_by_run) > 0.5,
                "lstm_beats": fraction_better(run_scores(y[valid], val_preds["lstm"], runs[valid]), trusted_by_run) > 0.5,
            }
        )
        for split_name, mask, preds in (
            ("validation", valid, val_preds),
            ("test", test, test_preds),
        ):
            frame = eligible.iloc[np.flatnonzero(mask)][["run_id", "ticker", "case_tick"]].copy()
            frame["horizon"] = horizon
            frame["label"] = y[mask]
            for name in ("zero", "ridge", "hgb", "xgb", "lstm"):
                frame[name] = preds[name]
            for name in ("hgb_q10", "hgb_q90", "xgb_q10", "xgb_q90", "band_q10", "band_q90"):
                frame[name] = preds.get(name, np.nan)
            if split_name == "validation":
                predictions.append(frame)
            else:
                test_predictions.append(frame)

    return metrics, predictions, test_predictions, comparisons


def _prediction_rows(predictions: pd.DataFrame) -> list[tuple]:
    rows = []
    for record in predictions.itertuples(index=False):
        rows.append(
            (
                record.run_id,
                record.ticker,
                int(record.case_tick),
                int(record.horizon),
                float(record.label),
                float(record.zero),
                float(record.ridge),
                float(record.hgb),
                float(record.xgb),
                float(record.lstm),
                None if pd.isna(record.hgb_q10) else float(record.hgb_q10),
                None if pd.isna(record.hgb_q90) else float(record.hgb_q90),
                None if pd.isna(record.xgb_q10) else float(record.xgb_q10),
                None if pd.isna(record.xgb_q90) else float(record.xgb_q90),
                None if pd.isna(record.band_q10) else float(record.band_q10),
                None if pd.isna(record.band_q90) else float(record.band_q90),
            )
        )
    return rows


CREATE_PREDICTIONS = """
            CREATE TABLE {table} (
              run_id TEXT NOT NULL,
              ticker TEXT NOT NULL,
              case_tick INTEGER NOT NULL,
              horizon INTEGER NOT NULL,
              label REAL NOT NULL,
              zero_pred REAL NOT NULL,
              ridge REAL NOT NULL,
              hgb REAL NOT NULL,
              xgb REAL NOT NULL,
              lstm REAL NOT NULL,
              hgb_q10 REAL,
              hgb_q90 REAL,
              xgb_q10 REAL,
              xgb_q90 REAL,
              band_q10 REAL,
              band_q90 REAL,
              PRIMARY KEY (run_id, ticker, case_tick, horizon)
            )
            """


def write_results(metrics: list[dict], predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> None:
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    con = sqlite3.connect(OUT_PATH)
    try:
        con.execute("BEGIN")
        con.execute(
            """
            CREATE TABLE metrics (
              model TEXT NOT NULL,
              ticker TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              split TEXT NOT NULL,
              mae REAL NOT NULL,
              rmse REAL NOT NULL,
              zero_mae REAL NOT NULL,
              zero_rmse REAL NOT NULL,
              frac_runs_beat_zero REAL NOT NULL,
              pinball REAL,
              band_pinball REAL,
              detail TEXT,
              PRIMARY KEY (model, ticker, horizon, split)
            )
            """
        )
        con.execute(CREATE_PREDICTIONS.format(table="validation_predictions"))
        con.execute(CREATE_PREDICTIONS.format(table="test_predictions"))
        con.executemany(
            """
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["model"],
                    row["ticker"],
                    row["horizon"],
                    row["split"],
                    row["mae"],
                    row["rmse"],
                    row["zero_mae"],
                    row["zero_rmse"],
                    row["frac_runs_beat_zero"],
                    row["pinball"],
                    row["band_pinball"],
                    row["detail"],
                )
                for row in metrics
            ],
        )
        insert_sql = "INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        con.executemany(insert_sql.format(table="validation_predictions"), _prediction_rows(predictions))
        con.executemany(insert_sql.format(table="test_predictions"), _prediction_rows(test_predictions))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def print_comparison(comparisons: list[dict]) -> None:
    print("ticker horizon trusted trusted_mae zero_mae xgb_mae lstm_mae xgb_beats lstm_beats")
    for row in comparisons:
        print(
            f"{row['ticker']} {row['horizon']} {row['trusted']} "
            f"{row['trusted_mae']:.5f} {row['zero_mae']:.5f} {row['xgb_mae']:.5f} {row['lstm_mae']:.5f} "
            f"{'yes' if row['xgb_beats'] else 'no'} {'yes' if row['lstm_beats'] else 'no'}"
        )


def main() -> None:
    if len(FEATURES) != 80:
        raise RuntimeError(f"expected 80 features, found {len(FEATURES)}")
    frame = load_panel()
    metrics = []
    predictions = []
    test_predictions = []
    comparisons = []
    for ticker in TICKERS:
        ticker_metrics, ticker_predictions, ticker_test, ticker_comparisons = fit_ticker(frame, ticker)
        metrics.extend(ticker_metrics)
        predictions.extend(ticker_predictions)
        test_predictions.extend(ticker_test)
        comparisons.extend(ticker_comparisons)
    write_results(
        metrics,
        pd.concat(predictions, ignore_index=True),
        pd.concat(test_predictions, ignore_index=True),
    )
    print_comparison(comparisons)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
