"""Write test-set prediction plots for the research note."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent
DB = ROOT / "price_model_results.sqlite"
FEATURES = ROOT / "price_features.sqlite"
FIG_DIR = ROOT / "research" / "figures"


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def load_test() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT * FROM test_predictions", con)
    finally:
        con.close()


def first_test_run(ticker: str) -> str:
    con = sqlite3.connect(f"file:{FEATURES}?mode=ro", uri=True)
    try:
        runs = pd.read_sql_query(
            "SELECT run_id FROM runs WHERE split = 'test' ORDER BY started_utc, run_id",
            con,
        )
    finally:
        con.close()
    return str(runs["run_id"].iloc[0])


def scatter_figure(frame: pd.DataFrame, ticker: str, horizon: int, path: Path) -> None:
    part = frame.loc[(frame["ticker"] == ticker) & (frame["horizon"] == horizon)]
    y = part["label"].to_numpy(dtype=float)
    hgb = part["hgb"].to_numpy(dtype=float)
    zero = np.zeros_like(y)
    lo = float(np.nanmin(np.concatenate([y, hgb])))
    hi = float(np.nanmax(np.concatenate([y, hgb])))
    pad = 0.05 * (hi - lo if hi > lo else 0.1)
    lims = (lo - pad, hi + pad)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y, zero, s=8, alpha=0.25, label="predict 0", color="#888888")
    ax.scatter(y, hgb, s=10, alpha=0.35, label="boosting", color="#1f4e79")
    ax.plot(lims, lims, color="black", linewidth=1, label="perfect forecast")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true mid change")
    ax.set_ylabel("predicted mid change")
    ax.set_title(f"{ticker}  {horizon}-tick  test")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def path_figure(frame: pd.DataFrame, ticker: str, run_id: str, path: Path) -> None:
    part = frame.loc[
        (frame["ticker"] == ticker) & (frame["horizon"] == 31) & (frame["run_id"] == run_id)
    ].sort_values("case_tick")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(
        part["case_tick"],
        part["hgb_q10"],
        part["hgb_q90"],
        color="#1f4e79",
        alpha=0.15,
        label="boosted 10–90% band",
    )
    ax.plot(part["case_tick"], part["label"], color="black", linewidth=1.2, label="true y_31")
    ax.plot(part["case_tick"], part["hgb"], color="#1f4e79", linewidth=1.2, label="boosting")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_xlabel("case tick")
    ax.set_ylabel("31-tick mid change")
    ax.set_title(f"{ticker}  test run {run_id}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_r2(frame: pd.DataFrame) -> None:
    print("ticker horizon model r2")
    for ticker in ("CRZY", "TAME"):
        for horizon in (5, 31):
            part = frame.loc[(frame["ticker"] == ticker) & (frame["horizon"] == horizon)]
            y = part["label"].to_numpy(dtype=float)
            for name in ("zero_pred", "ridge", "hgb", "xgb", "lstm"):
                print(ticker, horizon, name, f"{r2_score(y, part[name].to_numpy(dtype=float)):.4f}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_test()
    print_r2(frame)
    for ticker in ("CRZY", "TAME"):
        for horizon in (5, 31):
            scatter_figure(
                frame,
                ticker,
                horizon,
                FIG_DIR / f"test_scatter_{ticker.lower()}_h{horizon}.png",
            )
        path_figure(
            frame,
            ticker,
            first_test_run(ticker),
            FIG_DIR / f"test_path_{ticker.lower()}_h31.png",
        )
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
