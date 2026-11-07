"""Test which feature groups the 5-tick and 31-tick models use.

Ridge, boosting, and XGBoost are refit after dropping one group.
The LSTM is refit after dropping one of its eight inputs.
Columns inside a passing group are then shuffled on the full model.
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

from build_price_features import FEATURES
from train_price_models import (
    LSTM_FEATURES,
    TREE_PARAMS,
    load_panel,
    impute_ticker,
    mae,
    pinball,
    predict_lstm,
    window_matrix,
)

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "feature_tests.sqlite"
NOTE_PATH = ROOT / "research" / "lt3_findings.md"

TICKERS = ("CRZY", "TAME")
HORIZONS = (5, 31)
THRESHOLD = 0.0005  # 0.05 cents, in price points
RIDGE_ALPHA = {
    ("CRZY", 5): 100.0,
    ("CRZY", 31): 1000.0,
    ("TAME", 5): 10.0,
    ("TAME", 31): 1000.0,
}
GROUPS = {
    "returns": ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_15", "ret_31"],
    "trend": [
        "sma_dist_5",
        "sma_dist_10",
        "sma_dist_20",
        "sma_dist_31",
        "ema_dist_5",
        "ema_dist_10",
        "ema_dist_20",
        "macd_5_15",
        "macd_signal",
        "range_pos_10",
        "range_pos_31",
        "boll_20",
        "rsi_5",
        "rsi_14",
        "dist_start",
    ],
    "volatility": ["rvol_5", "rvol_10", "rvol_31", "var_ratio_31"],
    "touch": [
        "spread",
        "spread_mid",
        "bid_sz",
        "ask_sz",
        "log_bid_sz",
        "log_ask_sz",
        "imb_touch",
        "micro_gap",
        "d_spread",
        "d_imb",
    ],
    "depth": [
        "depth_bid_1c",
        "depth_bid_2c",
        "depth_bid_5c",
        "depth_bid_10c",
        "depth_bid_all",
        "depth_ask_1c",
        "depth_ask_2c",
        "depth_ask_5c",
        "depth_ask_10c",
        "depth_ask_all",
        "log_depth_1c",
        "log_depth_2c",
        "log_depth_5c",
        "log_depth_10c",
        "log_depth_all",
        "imb_5c",
        "imb_10c",
        "imb_all",
        "dist_bid_l5",
        "dist_bid_l10",
        "dist_ask_l5",
        "dist_ask_l10",
        "conc_bid",
        "conc_ask",
        "d_depth_5c",
    ],
    "walk": ["walk_bid", "walk_ask"],
    "tape": [
        "signed_vol_1",
        "signed_vol_5",
        "signed_vol_10",
        "print_count_5",
        "vol_ratio_5_31",
        "last_print_mid",
        "print_vwap_dist_5",
        "print_vwap_dist_31",
    ],
    "clock": ["tick_frac", "ticks_left", "in_window", "ticks_to_window", "ticks_since_window"],
    "other": ["other_ret_1", "other_ret_5", "other_spread", "other_imb_touch"],
}
COLUMN_GROUP = {column: group for group, columns in GROUPS.items() for column in columns}


class SmallLSTM(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=16, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(16, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequences)
        return self.head(self.drop(output[:, -1, :])).squeeze(-1)


def _assert_groups() -> None:
    grouped = [column for columns in GROUPS.values() for column in columns]
    if sorted(grouped) != sorted(FEATURES):
        missing = set(FEATURES) - set(grouped)
        extra = set(grouped) - set(FEATURES)
        raise RuntimeError(f"group columns mismatch missing={missing} extra={extra}")


def run_mae(y, pred, run_ids) -> dict[str, float]:
    scores = {}
    for run_id in pd.unique(run_ids):
        mask = run_ids == run_id
        scores[str(run_id)] = mae(y[mask], pred[mask])
    return scores


def run_pinball(y, pred, run_ids, quantile: float) -> dict[str, float]:
    scores = {}
    for run_id in pd.unique(run_ids):
        mask = run_ids == run_id
        scores[str(run_id)] = pinball(y[mask], pred[mask], quantile)
    return scores


def fraction_rose(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    return sum(candidate[run] > baseline[run] for run in candidate) / len(candidate)


def standardize(train: np.ndarray, *others: np.ndarray):
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return tuple((block - center) / scale for block in (train, *others))


def fit_ridge(x_train, y_train, alpha: float):
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(x_train, y_train)
    return model


def fit_hgb(x_train, y_train, loss: str = "squared_error", quantile: float | None = None):
    kwargs = dict(TREE_PARAMS)
    kwargs["loss"] = loss
    if quantile is not None:
        kwargs["quantile"] = quantile
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(x_train, y_train)
    return model


def fit_xgb(x_train, y_train, x_val, y_val):
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        min_child_weight=200,
        learning_rate=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
        early_stopping_rounds=10,
        eval_metric="mae",
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def fit_lstm(x_train, y_train, x_val, y_val):
    torch.manual_seed(0)
    model = SmallLSTM(x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
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
    waits = 0
    for _epoch in range(20):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = mae(y_val, model(val_x).numpy())
        if score < best_mae - 1e-8:
            best_mae = score
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            waits = 0
        else:
            waits += 1
            if waits >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def ablation_row(ticker, horizon, model, group, score, full_score, frac, error_kind):
    return {
        "ticker": ticker,
        "horizon": horizon,
        "model": model,
        "feature_group": group,
        "mae": score,
        "full_mae": full_score,
        "delta_mae": score - full_score,
        "frac_runs_error_rose": frac,
        "error_kind": error_kind,
        "passed": int(score - full_score >= THRESHOLD and frac > 0.5),
    }


def column_index(names: list[str], wanted: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    return np.array([lookup[name] for name in wanted], dtype=int)


def prepare_ticker(frame: pd.DataFrame, ticker: str, horizon: int):
    part = frame.loc[frame["ticker"] == ticker].sort_values(["run_id", "case_tick"]).reset_index(drop=True)
    observed = part["ret_31"].notna()
    part = impute_ticker(part)
    label = f"y_{horizon}"
    eligible = part.loc[observed & part[label].notna()].copy()
    windows, keep = window_matrix(
        part[LSTM_FEATURES].to_numpy(dtype=float),
        part["run_id"].to_numpy(),
        part["case_tick"].to_numpy(),
        eligible["run_id"].to_numpy(),
        eligible["case_tick"].to_numpy(),
    )
    if not keep.all():
        eligible = eligible.iloc[np.flatnonzero(keep)].copy()
        windows = windows[keep]
    y = eligible[label].to_numpy(dtype=float)
    runs = eligible["run_id"].to_numpy()
    raw = eligible[FEATURES].to_numpy(dtype=float)
    split = eligible["split"].to_numpy()
    train = split == "train"
    valid = split == "validation"
    test = split == "test"
    return {
        "y": y,
        "runs": runs,
        "raw": raw,
        "windows": windows,
        "train": train,
        "valid": valid,
        "test": test,
    }


def shuffle_column(block: np.ndarray, index: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = block.copy()
    shuffled[:, index] = rng.permutation(shuffled[:, index])
    return shuffled


def shuffle_sequence_feature(windows: np.ndarray, index: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = windows.copy()
    order = rng.permutation(len(windows))
    shuffled[:, :, index] = windows[order, :, index]
    return shuffled


def test_tabular(data, ticker: str, horizon: int):
    y = data["y"]
    runs = data["runs"]
    raw = data["raw"]
    train, valid = data["train"], data["valid"]
    alpha = RIDGE_ALPHA[(ticker, horizon)]
    ablation = []
    importance = []
    kept = []
    rng = np.random.default_rng(0)

    models = {}
    full_pred = {}
    full_score = {}
    full_by_run = {}

    x_train, x_val = standardize(raw[train], raw[valid])
    models["ridge"] = fit_ridge(x_train, y[train], alpha)
    models["hgb"] = fit_hgb(raw[train], y[train])
    models["xgb"] = fit_xgb(raw[train], y[train], raw[valid], y[valid])
    full_pred["ridge"] = models["ridge"].predict(x_val)
    full_pred["hgb"] = models["hgb"].predict(raw[valid])
    full_pred["xgb"] = models["xgb"].predict(raw[valid])
    for name in ("ridge", "hgb", "xgb"):
        full_score[name] = mae(y[valid], full_pred[name])
        full_by_run[name] = run_mae(y[valid], full_pred[name], runs[valid])
        ablation.append(
            ablation_row(ticker, horizon, name, "all", full_score[name], full_score[name], 0.0, "mae")
        )
        print(f"{ticker} y_{horizon} {name} full mae={full_score[name]:.5f}", file=sys.stderr)

    passed_groups = {name: [] for name in ("ridge", "hgb", "xgb")}
    for group, columns in GROUPS.items():
        keep = [column for column in FEATURES if column not in columns]
        idx = column_index(FEATURES, keep)
        sub = raw[:, idx]
        sub_train, sub_val = standardize(sub[train], sub[valid])
        dropped = {
            "ridge": fit_ridge(sub_train, y[train], alpha).predict(sub_val),
            "hgb": fit_hgb(sub[train], y[train]).predict(sub[valid]),
            "xgb": fit_xgb(sub[train], y[train], sub[valid], y[valid]).predict(sub[valid]),
        }
        for name, pred in dropped.items():
            score = mae(y[valid], pred)
            by_run = run_mae(y[valid], pred, runs[valid])
            frac = fraction_rose(by_run, full_by_run[name])
            row = ablation_row(ticker, horizon, name, group, score, full_score[name], frac, "mae")
            ablation.append(row)
            if row["passed"]:
                passed_groups[name].append(group)
            print(
                f"{ticker} y_{horizon} {name} drop {group} delta={row['delta_mae']:.5f} "
                f"frac={frac:.2f} pass={row['passed']}",
                file=sys.stderr,
            )

    if horizon == 31:
        q_models = {
            "hgb_q10": fit_hgb(raw[train], y[train], loss="quantile", quantile=0.1),
            "hgb_q90": fit_hgb(raw[train], y[train], loss="quantile", quantile=0.9),
        }
        q_full = {name: model.predict(raw[valid]) for name, model in q_models.items()}
        q_score = {}
        q_by_run = {}
        for name, quantile in (("hgb_q10", 0.1), ("hgb_q90", 0.9)):
            q_score[name] = pinball(y[valid], q_full[name], quantile)
            q_by_run[name] = run_pinball(y[valid], q_full[name], runs[valid], quantile)
            ablation.append(
                ablation_row(ticker, horizon, name, "all", q_score[name], q_score[name], 0.0, "pinball")
            )
            passed_groups[name] = []
        for group, columns in GROUPS.items():
            keep = [column for column in FEATURES if column not in columns]
            idx = column_index(FEATURES, keep)
            sub = raw[:, idx]
            dropped = {
                "hgb_q10": fit_hgb(sub[train], y[train], loss="quantile", quantile=0.1).predict(sub[valid]),
                "hgb_q90": fit_hgb(sub[train], y[train], loss="quantile", quantile=0.9).predict(sub[valid]),
            }
            for name, quantile in (("hgb_q10", 0.1), ("hgb_q90", 0.9)):
                score = pinball(y[valid], dropped[name], quantile)
                by_run = run_pinball(y[valid], dropped[name], runs[valid], quantile)
                frac = fraction_rose(by_run, q_by_run[name])
                row = ablation_row(ticker, horizon, name, group, score, q_score[name], frac, "pinball")
                ablation.append(row)
                if row["passed"]:
                    passed_groups[name].append(group)

        for name, quantile in (("hgb_q10", 0.1), ("hgb_q90", 0.9)):
            columns = sorted({column for group in passed_groups[name] for column in GROUPS[group]})
            for column in columns:
                index = FEATURES.index(column)
                shuffled = shuffle_column(raw[valid], index, rng)
                pred = q_models[name].predict(shuffled)
                delta = pinball(y[valid], pred, quantile) - q_score[name]
                importance.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "model": name,
                        "column": column,
                        "feature_group": COLUMN_GROUP[column],
                        "delta_mae": delta,
                        "passed": int(delta >= THRESHOLD),
                    }
                )
                if delta >= THRESHOLD:
                    kept.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "model": name,
                            "column": column,
                            "feature_group": COLUMN_GROUP[column],
                        }
                    )

    for name in ("ridge", "hgb", "xgb"):
        columns = sorted({column for group in passed_groups[name] for column in GROUPS[group]})
        for column in columns:
            index = FEATURES.index(column)
            shuffled = shuffle_column(raw[valid], index, rng)
            if name == "ridge":
                _, scaled = standardize(raw[train], shuffled)
                pred = models[name].predict(scaled)
            else:
                pred = models[name].predict(shuffled)
            delta = mae(y[valid], pred) - full_score[name]
            importance.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "model": name,
                    "column": column,
                    "feature_group": COLUMN_GROUP[column],
                    "delta_mae": delta,
                    "passed": int(delta >= THRESHOLD),
                }
            )
            if delta >= THRESHOLD:
                kept.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "model": name,
                        "column": column,
                        "feature_group": COLUMN_GROUP[column],
                    }
                )

    return ablation, importance, kept


def test_lstm(data, ticker: str, horizon: int):
    y = data["y"]
    runs = data["runs"]
    windows = data["windows"]
    train, valid = data["train"], data["valid"]
    train_w = windows[train]
    center = train_w.reshape(-1, train_w.shape[-1]).mean(axis=0)
    scale = train_w.reshape(-1, train_w.shape[-1]).std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (windows - center) / scale
    full = fit_lstm(scaled[train], y[train], scaled[valid], y[valid])
    full_pred = predict_lstm(full, scaled[valid])
    full_score = mae(y[valid], full_pred)
    full_by_run = run_mae(y[valid], full_pred, runs[valid])
    ablation = [ablation_row(ticker, horizon, "lstm", "all", full_score, full_score, 0.0, "mae")]
    print(f"{ticker} y_{horizon} lstm full mae={full_score:.5f}", file=sys.stderr)
    passed = []
    for index, name in enumerate(LSTM_FEATURES):
        keep = [i for i in range(len(LSTM_FEATURES)) if i != index]
        dropped = fit_lstm(scaled[train][:, :, keep], y[train], scaled[valid][:, :, keep], y[valid])
        pred = predict_lstm(dropped, scaled[valid][:, :, keep])
        score = mae(y[valid], pred)
        frac = fraction_rose(run_mae(y[valid], pred, runs[valid]), full_by_run)
        row = ablation_row(ticker, horizon, "lstm", name, score, full_score, frac, "mae")
        ablation.append(row)
        if row["passed"]:
            passed.append(name)
        print(
            f"{ticker} y_{horizon} lstm drop {name} delta={row['delta_mae']:.5f} "
            f"frac={frac:.2f} pass={row['passed']}",
            file=sys.stderr,
        )

    rng = np.random.default_rng(1)
    importance = []
    kept = []
    for name in passed:
        index = LSTM_FEATURES.index(name)
        shuffled = shuffle_sequence_feature(scaled[valid], index, rng)
        pred = predict_lstm(full, shuffled)
        delta = mae(y[valid], pred) - full_score
        group = COLUMN_GROUP.get(name, "lstm")
        importance.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "model": "lstm",
                "column": name,
                "feature_group": group,
                "delta_mae": delta,
                "passed": int(delta >= THRESHOLD),
            }
        )
        if delta >= THRESHOLD:
            kept.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "model": "lstm",
                    "column": name,
                    "feature_group": group,
                }
            )
    return ablation, importance, kept


def write_db(ablation: list[dict], importance: list[dict], kept: list[dict]) -> None:
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    con = sqlite3.connect(OUT_PATH)
    try:
        con.execute("BEGIN")
        con.execute(
            """
            CREATE TABLE group_ablation (
              ticker TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              model TEXT NOT NULL,
              feature_group TEXT NOT NULL,
              mae REAL NOT NULL,
              full_mae REAL NOT NULL,
              delta_mae REAL NOT NULL,
              frac_runs_error_rose REAL NOT NULL,
              error_kind TEXT NOT NULL,
              passed INTEGER NOT NULL,
              PRIMARY KEY (ticker, horizon, model, feature_group)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE column_importance (
              ticker TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              model TEXT NOT NULL,
              column TEXT NOT NULL,
              feature_group TEXT NOT NULL,
              delta_mae REAL NOT NULL,
              passed INTEGER NOT NULL,
              PRIMARY KEY (ticker, horizon, model, column)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE kept_features (
              ticker TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              model TEXT NOT NULL,
              column TEXT NOT NULL,
              feature_group TEXT NOT NULL,
              PRIMARY KEY (ticker, horizon, model, column)
            )
            """
        )
        con.executemany(
            "INSERT INTO group_ablation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["ticker"],
                    row["horizon"],
                    row["model"],
                    row["feature_group"],
                    row["mae"],
                    row["full_mae"],
                    row["delta_mae"],
                    row["frac_runs_error_rose"],
                    row["error_kind"],
                    row["passed"],
                )
                for row in ablation
            ],
        )
        con.executemany(
            "INSERT INTO column_importance VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["ticker"],
                    row["horizon"],
                    row["model"],
                    row["column"],
                    row["feature_group"],
                    row["delta_mae"],
                    row["passed"],
                )
                for row in importance
            ],
        )
        con.executemany(
            "INSERT INTO kept_features VALUES (?, ?, ?, ?, ?)",
            [
                (row["ticker"], row["horizon"], row["model"], row["column"], row["feature_group"])
                for row in kept
            ],
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _cents(value: float) -> str:
    return f"{value * 100:.2f}"


def fill_research_note(ablation: list[dict], kept: list[dict]) -> None:
    lines = []
    lines.append("A group of features works if, after it is removed and the model is refit, validation error rises by at least 0.05 cents and the error rises on a majority of the 20 validation runs. One-tick models are excluded. Individual columns inside a working group are then shuffled on the validation rows of the full model. A column is kept when that shuffle raises error by at least 0.05 cents.")
    lines.append("")
    lines.append("Scores are in `feature_tests.sqlite`. The keep list was chosen on validation. The test runs were not used to decide which features work.")
    lines.append("")
    lines.append("### Groups that raised validation error")
    lines.append("")
    lines.append("| Ticker | Horizon | Model | Group | Error rise (cents) | Runs where error rose |")
    lines.append("|---|---|---|---|---|---|")
    passing = [row for row in ablation if row["passed"] and row["feature_group"] != "all"]
    passing.sort(key=lambda row: (row["ticker"], row["horizon"], row["model"], -row["delta_mae"]))
    if not passing:
        lines.append("| — | — | — | none | — | — |")
    else:
        for row in passing:
            lines.append(
                f"| {row['ticker']} | {row['horizon']} | {row['model']} | {row['feature_group']} | "
                f"{_cents(row['delta_mae'])} | {row['frac_runs_error_rose']:.0%} |"
            )
    lines.append("")
    lines.append("### Columns kept after shuffling")
    lines.append("")
    lines.append("| Ticker | Horizon | Model | Column | Group |")
    lines.append("|---|---|---|---|---|")
    ordered = sorted(kept, key=lambda row: (row["ticker"], row["horizon"], row["model"], row["feature_group"], row["column"]))
    if not ordered:
        lines.append("| — | — | — | none met the 0.05-cent shuffle rule | — |")
    else:
        for row in ordered:
            lines.append(
                f"| {row['ticker']} | {row['horizon']} | {row['model']} | `{row['column']}` | {row['feature_group']} |"
            )
    lines.append("")
    unused = []
    for ticker in TICKERS:
        for horizon in HORIZONS:
            for model in ("ridge", "hgb", "xgb", "lstm"):
                groups_used = {
                    row["feature_group"]
                    for row in passing
                    if row["ticker"] == ticker and row["horizon"] == horizon and row["model"] == model
                }
                if model == "lstm":
                    continue
                missing = [name for name in GROUPS if name not in groups_used]
                if missing:
                    unused.append(f"- {ticker} {horizon}-tick {model}: {', '.join(missing)}")
    if unused:
        lines.append("### Groups that did not pass")
        lines.append("")
        lines.extend(unused)
        lines.append("")
    body = "\n".join(lines)
    text = NOTE_PATH.read_text(encoding="utf-8")
    heading = "## Feature test\n"
    next_heading = "## What remains\n"
    heading_at = text.index(heading)
    next_at = text.index(next_heading)
    NOTE_PATH.write_text(text[:heading_at] + heading + "\n" + body + "\n" + text[next_at:], encoding="utf-8")


def main() -> None:
    _assert_groups()
    frame = load_panel()
    ablation = []
    importance = []
    kept = []
    for ticker in TICKERS:
        for horizon in HORIZONS:
            data = prepare_ticker(frame, ticker, horizon)
            tab_a, tab_i, tab_k = test_tabular(data, ticker, horizon)
            lstm_a, lstm_i, lstm_k = test_lstm(data, ticker, horizon)
            ablation.extend(tab_a)
            ablation.extend(lstm_a)
            importance.extend(tab_i)
            importance.extend(lstm_i)
            kept.extend(tab_k)
            kept.extend(lstm_k)
    write_db(ablation, importance, kept)
    fill_research_note(ablation, kept)
    print(f"wrote {OUT_PATH}")
    print(f"updated {NOTE_PATH}")


if __name__ == "__main__":
    main()
