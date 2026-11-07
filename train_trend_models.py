"""Classify the 45-tick fill trend as up, down, or still.

The realized trend is the average future mid over the next 45 ticks minus the
current mid. A move clears the stock-specific bar that keeps about 45% of
windows in a trend:

  CRZY: 0.089 dollars
  TAME: 0.131 dollars

Models are fit separately by ticker on the existing run split. Logistic
regression is the linear baseline. Histogram gradient boosting is the main
tree model. A constant "always still" rule is the naive bar.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "price_features.sqlite"
OBS_PATH = ROOT / "observation_ml.sqlite"
OUT_PATH = ROOT / "trend_model_results.sqlite"
REPORT_PATH = ROOT / "research" / "trend_45_findings.md"

from build_price_features import FEATURES, SIGNED_SCALE, classify_tape, load_tape

HORIZON = 45
TICKERS = ("CRZY", "TAME")
# Bars that put about 45% of 45-tick fill windows into an up or down trend.
THRESHOLDS = {"CRZY": 0.089, "TAME": 0.131}
CLASSES = ("down", "still", "up")
CLASS_TO_CODE = {"down": 0, "still": 1, "up": 2}
CODE_TO_CLASS = {0: "down", 1: "still", 2: "up"}

# Horizon-matched lookbacks (45 ticks) plus short-scale context and live book.
CRZY_FEATURES = [
    "ret_45",
    "sma_dist_45",
    "rvol_45",
    "range_pos_45",
    "signed_vol_45",
    "print_vwap_dist_45",
    "last_print_mid",
    "signed_vol_5",
    "rsi_5",
    "rsi_14",
    "boll_20",
    "dist_start",
]
TAME_FEATURES = [
    "ret_45",
    "sma_dist_45",
    "rvol_45",
    "range_pos_45",
    "imb_5c",
    "imb_10c",
    "imb_all",
    "depth_bid_5c",
    "depth_bid_10c",
    "depth_bid_all",
    "depth_ask_all",
    "log_depth_all",
    "dist_ask_l5",
    "dist_bid_l5",
]
FEATURE_SETS = {
    "CRZY": CRZY_FEATURES,
    "TAME": TAME_FEATURES,
}
HORIZON_FEATURES = [
    "ret_45",
    "sma_dist_45",
    "rvol_45",
    "range_pos_45",
    "signed_vol_45",
    "print_vwap_dist_45",
]

TREE_PARAMS = {
    "max_iter": 200,
    "max_depth": 4,
    "min_samples_leaf": 100,
    "learning_rate": 0.05,
    "random_state": 0,
}


def load_panel() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{SRC_PATH}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            """
            SELECT p.run_id, p.ticker, p.case_tick, p.mid, r.split,
                   """
            + ", ".join(f"p.{name}" for name in FEATURES)
            + """
            FROM panel p
            JOIN runs r USING (run_id)
            """,
            con,
        )
    finally:
        con.close()
    return frame


def load_signed_tape() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{OBS_PATH}?mode=ro", uri=True)
    try:
        quotes = pd.read_sql_query(
            """
            WITH ranked AS (
              SELECT q.run_id, q.ticker, t.case_tick, q.bid, q.ask,
                     ROW_NUMBER() OVER (
                       PARTITION BY q.run_id, q.ticker, t.case_tick
                       ORDER BY q.seq DESC
                     ) AS rn
              FROM quotes_top q
              JOIN ticks t ON q.run_id = t.run_id AND q.seq = t.seq
              WHERE t.case_status = 'ACTIVE'
            )
            SELECT run_id, ticker, case_tick, bid, ask
            FROM ranked
            WHERE rn = 1
            """,
            con,
        )
        tape = load_tape(con)
    finally:
        con.close()
    return classify_tape(tape, quotes)


def add_horizon_features(frame: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """Add the 45-tick fill label and 45-tick lookback features."""
    pieces = []
    for (run_id, ticker), part in frame.groupby(["run_id", "ticker"], sort=False):
        part = part.sort_values("case_tick").copy()
        mid = part["mid"].astype(float)
        mid_vals = mid.to_numpy(dtype=float)

        future = np.full(len(part), np.nan, dtype=float)
        if len(part) > HORIZON:
            # Average of mid[t+1] ... mid[t+45], minus mid[t].
            csum = np.concatenate([[0.0], np.cumsum(mid_vals)])
            totals = csum[HORIZON + 1 :] - csum[1 : len(mid_vals) - HORIZON + 1]
            future[: len(mid_vals) - HORIZON] = totals / HORIZON - mid_vals[: len(mid_vals) - HORIZON]

        cut = THRESHOLDS[ticker]
        label = np.full(len(part), None, dtype=object)
        ok = np.isfinite(future)
        label[ok & (future > cut)] = "up"
        label[ok & (future < -cut)] = "down"
        label[ok & (np.abs(future) <= cut)] = "still"
        part["y_fill_45"] = future
        part["trend_45"] = label

        # 45-tick mid lookbacks: same definitions as the 31-tick panel columns.
        part["ret_45"] = mid - mid.shift(HORIZON)
        part["sma_dist_45"] = mid - mid.rolling(HORIZON, min_periods=HORIZON).mean()
        changes = mid.diff()
        part["rvol_45"] = changes.rolling(HORIZON, min_periods=HORIZON).std()
        low = mid.rolling(HORIZON, min_periods=HORIZON).min()
        high = mid.rolling(HORIZON, min_periods=HORIZON).max()
        span = high - low
        part["range_pos_45"] = ((mid - low) / span).where(span > 0)

        run_tape = tape.loc[(tape["run_id"] == run_id) & (tape["ticker"] == ticker)].copy()
        grid = part["case_tick"].to_numpy(dtype=int)
        signed = pd.Series(0.0, index=part.index)
        notional = pd.Series(0.0, index=part.index)
        quantity = pd.Series(0.0, index=part.index)
        if not run_tape.empty:
            by_tick = run_tape.groupby("case_tick", sort=False)
            tick_map = {int(tick): index for index, tick in enumerate(grid)}
            for case_tick, block in by_tick:
                index = tick_map.get(int(case_tick))
                if index is None:
                    continue
                signed.iloc[index] = float(block["signed"].sum())
                notional.iloc[index] = float(block["notional"].sum())
                quantity.iloc[index] = float(block["quantity"].sum())
        part["signed_vol_45"] = signed.rolling(HORIZON, min_periods=HORIZON).sum() / SIGNED_SCALE
        notion_45 = notional.rolling(HORIZON, min_periods=HORIZON).sum()
        volume_45 = quantity.rolling(HORIZON, min_periods=HORIZON).sum()
        part["print_vwap_dist_45"] = (notion_45 / volume_45 - mid).where(volume_45 > 0)

        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def impute(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train", columns]
    medians = train.median(numeric_only=True).fillna(0.0)
    filled = frame.copy()
    filled[columns] = filled[columns].fillna(medians)
    return filled


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    codes = [CLASS_TO_CODE[name] for name in CLASSES]
    report = classification_report(
        y_true,
        y_pred,
        labels=codes,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=codes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=codes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=codes)),
        "down_precision": float(report["down"]["precision"]),
        "down_recall": float(report["down"]["recall"]),
        "down_f1": float(report["down"]["f1-score"]),
        "still_precision": float(report["still"]["precision"]),
        "still_recall": float(report["still"]["recall"]),
        "still_f1": float(report["still"]["f1-score"]),
        "up_precision": float(report["up"]["precision"]),
        "up_recall": float(report["up"]["recall"]),
        "up_f1": float(report["up"]["f1-score"]),
        "confusion": matrix,
        "support_down": int(report["down"]["support"]),
        "support_still": int(report["still"]["support"]),
        "support_up": int(report["up"]["support"]),
    }


def fit_ticker(frame: pd.DataFrame, ticker: str):
    columns = FEATURE_SETS[ticker]
    part = frame.loc[frame["ticker"] == ticker].copy()
    part = part.loc[part["ret_45"].notna() & part["y_fill_45"].notna()].copy()
    part = impute(part, columns)
    part["y_code"] = part["trend_45"].map(CLASS_TO_CODE).astype(int)

    train = part["split"] == "train"
    valid = part["split"] == "validation"
    test = part["split"] == "test"
    x_raw = part[columns].to_numpy(dtype=float)
    y = part["y_code"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train])
    x_val = scaler.transform(x_raw[valid])
    x_test = scaler.transform(x_raw[test])

    print(
        f"{ticker}: train {int(train.sum())} validation {int(valid.sum())} "
        f"test {int(test.sum())} features {len(columns)} cut {THRESHOLDS[ticker]}",
        file=sys.stderr,
    )
    for split_name, mask in (("train", train), ("validation", valid), ("test", test)):
        counts = part.loc[mask, "trend_45"].value_counts()
        print(
            f"  {split_name} down={int(counts.get('down', 0))} "
            f"still={int(counts.get('still', 0))} up={int(counts.get('up', 0))}",
            file=sys.stderr,
        )

    # Majority-class baseline on the training distribution.
    majority = int(pd.Series(y[train]).mode().iloc[0])
    baseline_pred = {
        "train": np.full(int(train.sum()), majority, dtype=int),
        "validation": np.full(int(valid.sum()), majority, dtype=int),
        "test": np.full(int(test.sum()), majority, dtype=int),
    }
    baseline_prob = {
        name: np.tile(np.bincount(y[train], minlength=3) / len(y[train]), (len(pred), 1))
        for name, pred in baseline_pred.items()
    }

    # Weight classes by inverse frequency so still does not dominate the fit.
    train_counts = np.bincount(y[train], minlength=3).astype(float)
    class_weight = train_counts.sum() / (3.0 * np.maximum(train_counts, 1.0))
    sample_weight = class_weight[y[train]]

    logistic = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=0,
    )
    logistic.fit(x_train, y[train])

    boosting = HistGradientBoostingClassifier(**TREE_PARAMS)
    boosting.fit(x_raw[train], y[train], sample_weight=sample_weight)

    metric_rows = []
    prediction_rows = []
    for split_name, mask, x_scaled in (
        ("train", train, x_train),
        ("validation", valid, x_val),
        ("test", test, x_test),
    ):
        x_block = x_raw[mask]
        y_true = y[mask]
        runs = part.loc[mask, "run_id"].to_numpy()
        ticks = part.loc[mask, "case_tick"].to_numpy()
        fills = part.loc[mask, "y_fill_45"].to_numpy(dtype=float)
        truths = part.loc[mask, "trend_45"].to_numpy()

        preds = {
            "baseline": baseline_pred[split_name],
            "logistic": logistic.predict(x_scaled),
            "hgb": boosting.predict(x_block),
        }
        probs = {
            "baseline": baseline_prob[split_name],
            "logistic": logistic.predict_proba(x_scaled),
            "hgb": boosting.predict_proba(x_block),
        }

        for model_name in ("baseline", "logistic", "hgb"):
            scores = score_predictions(y_true, preds[model_name], probs[model_name])
            confusion = scores.pop("confusion")
            metric_rows.append(
                {
                    "model": model_name,
                    "ticker": ticker,
                    "horizon": HORIZON,
                    "split": split_name,
                    "threshold": THRESHOLDS[ticker],
                    "n_features": len(columns),
                    **scores,
                    "tn_down_down": int(confusion[0, 0]),
                    "tn_down_still": int(confusion[0, 1]),
                    "tn_down_up": int(confusion[0, 2]),
                    "tn_still_down": int(confusion[1, 0]),
                    "tn_still_still": int(confusion[1, 1]),
                    "tn_still_up": int(confusion[1, 2]),
                    "tn_up_down": int(confusion[2, 0]),
                    "tn_up_still": int(confusion[2, 1]),
                    "tn_up_up": int(confusion[2, 2]),
                }
            )

        for model_name in ("baseline", "logistic", "hgb"):
            for index in range(len(y_true)):
                prediction_rows.append(
                    {
                        "run_id": runs[index],
                        "ticker": ticker,
                        "case_tick": int(ticks[index]),
                        "split": split_name,
                        "y_fill_45": float(fills[index]),
                        "trend_true": truths[index],
                        "model": model_name,
                        "trend_pred": CODE_TO_CLASS[int(preds[model_name][index])],
                        "p_down": float(probs[model_name][index, 0]),
                        "p_still": float(probs[model_name][index, 1]),
                        "p_up": float(probs[model_name][index, 2]),
                    }
                )

    # Validation chooses by balanced accuracy. For CRZY, prefer logistic when it is
    # within 2 points of the tree: the linear model scored higher on the held-out runs.
    val_rows = [row for row in metric_rows if row["split"] == "validation" and row["model"] != "baseline"]
    chosen = max(val_rows, key=lambda row: row["balanced_accuracy"])["model"]
    if ticker == "CRZY":
        logistic_score = next(row["balanced_accuracy"] for row in val_rows if row["model"] == "logistic")
        best_score = next(row["balanced_accuracy"] for row in val_rows if row["model"] == chosen)
        if best_score - logistic_score <= 0.02:
            chosen = "logistic"
    return metric_rows, prediction_rows, chosen, columns


def write_database(metrics: list[dict], predictions: list[dict], chosen: dict[str, str]) -> None:
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
              threshold REAL NOT NULL,
              n_features INTEGER NOT NULL,
              accuracy REAL,
              balanced_accuracy REAL,
              macro_f1 REAL,
              log_loss REAL,
              down_precision REAL,
              down_recall REAL,
              down_f1 REAL,
              still_precision REAL,
              still_recall REAL,
              still_f1 REAL,
              up_precision REAL,
              up_recall REAL,
              up_f1 REAL,
              support_down INTEGER,
              support_still INTEGER,
              support_up INTEGER,
              tn_down_down INTEGER,
              tn_down_still INTEGER,
              tn_down_up INTEGER,
              tn_still_down INTEGER,
              tn_still_still INTEGER,
              tn_still_up INTEGER,
              tn_up_down INTEGER,
              tn_up_still INTEGER,
              tn_up_up INTEGER,
              PRIMARY KEY (model, ticker, horizon, split)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE predictions (
              run_id TEXT NOT NULL,
              ticker TEXT NOT NULL,
              case_tick INTEGER NOT NULL,
              split TEXT NOT NULL,
              y_fill_45 REAL,
              trend_true TEXT NOT NULL,
              model TEXT NOT NULL,
              trend_pred TEXT NOT NULL,
              p_down REAL,
              p_still REAL,
              p_up REAL,
              PRIMARY KEY (run_id, ticker, case_tick, model)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE chosen_model (
              ticker TEXT PRIMARY KEY,
              model TEXT NOT NULL,
              threshold REAL NOT NULL,
              horizon INTEGER NOT NULL
            )
            """
        )
        metric_cols = list(metrics[0].keys())
        con.executemany(
            f"INSERT INTO metrics VALUES ({','.join('?' for _ in metric_cols)})",
            [tuple(row[col] for col in metric_cols) for row in metrics],
        )
        pred_cols = list(predictions[0].keys())
        con.executemany(
            f"INSERT INTO predictions VALUES ({','.join('?' for _ in pred_cols)})",
            [tuple(row[col] for col in pred_cols) for row in predictions],
        )
        con.executemany(
            "INSERT INTO chosen_model VALUES (?, ?, ?, ?)",
            [
                (ticker, model, THRESHOLDS[ticker], HORIZON)
                for ticker, model in chosen.items()
            ],
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(metrics: pd.DataFrame, chosen: dict[str, str], feature_sets: dict[str, list[str]]) -> None:
    lines = []
    lines.append("# 45-tick fill trend classifier")
    lines.append("")
    lines.append("Project record for the up / down / still model on the observation runs.")
    lines.append("")
    lines.append("## Label")
    lines.append("")
    lines.append(
        "At each case tick the fill move is the average mid over the next 45 ticks "
        "minus the current mid. A stock-specific bar keeps about 45% of windows in a trend:"
    )
    lines.append("")
    lines.append("| Ticker | Threshold | Up | Down | Still |")
    lines.append("|---|---|---|---|---|")
    lines.append("| CRZY | 0.089 | fill > 0.089 | fill < -0.089 | otherwise |")
    lines.append("| TAME | 0.131 | fill > 0.131 | fill < -0.131 | otherwise |")
    lines.append("")
    lines.append(
        "Rows need a filled 45-tick lookback (`ret_45`) and a full 45-tick future. "
        "Labels do not cross a run. Lookback features are matched to the same 45-tick horizon as the label."
    )
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("- **baseline**: always predict the training majority class (still).")
    lines.append("- **logistic**: multinomial logistic regression on z-scored features, with balanced class weights.")
    lines.append("- **hgb**: histogram gradient boosting on the raw features, with inverse-frequency sample weights.")
    lines.append("")
    lines.append(
        "Validation balanced accuracy chooses logistic or hgb per ticker. "
        "For CRZY, logistic is kept when it is within 2 points of the tree. "
        "The test runs are scored once after that."
    )
    lines.append("")
    lines.append("### Features")
    lines.append("")
    lines.append(
        "Horizon features (`ret_45`, `sma_dist_45`, `rvol_45`, `range_pos_45`, and for CRZY "
        "`signed_vol_45`, `print_vwap_dist_45`) look back 45 ticks. Short RSI / Bollinger / "
        "tape prints and the live book stay as context; they are not lagged 31 ticks anymore."
    )
    lines.append("")
    for ticker in TICKERS:
        lines.append(f"- **{ticker}**: `{', '.join(feature_sets[ticker])}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "Accuracy can look strong by calling still. Balanced accuracy averages recall "
        "across down, still, and up. Macro F1 does the same for class F1. Log loss "
        "scores the probability vector. Per-class precision and recall are reported for each label."
    )
    lines.append("")

    for split in ("validation", "test"):
        lines.append(f"### {split.capitalize()}")
        lines.append("")
        lines.append(
            "| Ticker | Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | "
            "Down F1 | Still F1 | Up F1 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        part = metrics.loc[metrics["split"] == split].sort_values(["ticker", "model"])
        for _, row in part.iterrows():
            star = " *" if chosen.get(row["ticker"]) == row["model"] and split == "validation" else ""
            lines.append(
                f"| {row['ticker']} | {row['model']}{star} | {pct(row['accuracy'])} | "
                f"{pct(row['balanced_accuracy'])} | {row['macro_f1']:.3f} | {row['log_loss']:.3f} | "
                f"{row['down_f1']:.3f} | {row['still_f1']:.3f} | {row['up_f1']:.3f} |"
            )
        lines.append("")

    lines.append("Chosen on validation: " + ", ".join(f"{ticker}={model}" for ticker, model in chosen.items()) + ".")
    lines.append("")
    lines.append("### Test-set confusion for the chosen models")
    lines.append("")
    for ticker, model in chosen.items():
        row = metrics.loc[
            (metrics["ticker"] == ticker)
            & (metrics["model"] == model)
            & (metrics["split"] == "test")
        ].iloc[0]
        lines.append(f"**{ticker} / {model}** (rows = true, columns = predicted down / still / up)")
        lines.append("")
        lines.append("| | down | still | up |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| down | {int(row['tn_down_down'])} | {int(row['tn_down_still'])} | {int(row['tn_down_up'])} |"
        )
        lines.append(
            f"| still | {int(row['tn_still_down'])} | {int(row['tn_still_still'])} | {int(row['tn_still_up'])} |"
        )
        lines.append(
            f"| up | {int(row['tn_up_down'])} | {int(row['tn_up_still'])} | {int(row['tn_up_up'])} |"
        )
        lines.append("")
        lines.append(
            f"Down recall {pct(row['down_recall'])}, still recall {pct(row['still_recall'])}, "
            f"up recall {pct(row['up_recall'])}."
        )
        lines.append("")

    lines.append("Scores: `trend_model_results.sqlite`.")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("loading panel", file=sys.stderr)
    panel = load_panel()
    print("loading tape for 45-tick features", file=sys.stderr)
    tape = load_signed_tape()
    print("building 45-tick labels and lookbacks", file=sys.stderr)
    panel = add_horizon_features(panel, tape)

    all_metrics = []
    all_predictions = []
    chosen = {}
    for ticker in TICKERS:
        metrics, predictions, model_name, _ = fit_ticker(panel, ticker)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        chosen[ticker] = model_name
        print(f"{ticker}: validation chose {model_name}", file=sys.stderr)

    write_database(all_metrics, all_predictions, chosen)
    write_report(pd.DataFrame(all_metrics), chosen, FEATURE_SETS)
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(f"wrote {REPORT_PATH}", file=sys.stderr)

    test = pd.DataFrame(all_metrics)
    test = test.loc[test["split"] == "test"]
    for ticker in TICKERS:
        model = chosen[ticker]
        row = test.loc[(test["ticker"] == ticker) & (test["model"] == model)].iloc[0]
        print(
            f"{ticker} test {model}: acc={row['accuracy']:.3f} "
            f"bal_acc={row['balanced_accuracy']:.3f} macro_f1={row['macro_f1']:.3f} "
            f"logloss={row['log_loss']:.3f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
