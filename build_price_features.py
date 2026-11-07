"""Build price_features.sqlite from the LT3 observation recording.

One panel row is the last ACTIVE quote of a case tick. ret_k looks backward.
y_k looks forward. Rolling windows never cross a run.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "observation_ml.sqlite"
OUT_PATH = ROOT / "price_features.sqlite"

WINDOWS = ((14, 29), (74, 89), (134, 149), (194, 209), (254, 269))
WINDOW_STARTS = tuple(start for start, _ in WINDOWS)
WINDOW_ENDS = tuple(end for _, end in WINDOWS)
MAX_ORDER = {"CRZY": 25_000.0, "TAME": 10_000.0}
START_PRICE = {"CRZY": 10.0, "TAME": 25.0}
SIGNED_SCALE = 70_000.0

PRICE_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "ret_31",
    "sma_dist_5",
    "sma_dist_10",
    "sma_dist_20",
    "sma_dist_31",
    "ema_dist_5",
    "ema_dist_10",
    "ema_dist_20",
    "macd_5_15",
    "macd_signal",
    "rvol_5",
    "rvol_10",
    "rvol_31",
    "range_pos_10",
    "range_pos_31",
    "boll_20",
    "rsi_5",
    "rsi_14",
    "var_ratio_31",
    "dist_start",
]
BOOK_FEATURES = [
    "spread",
    "spread_mid",
    "bid_sz",
    "ask_sz",
    "log_bid_sz",
    "log_ask_sz",
    "imb_touch",
    "micro_gap",
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
    "walk_bid",
    "walk_ask",
    "d_spread",
    "d_imb",
    "d_depth_5c",
]
TAPE_FEATURES = [
    "signed_vol_1",
    "signed_vol_5",
    "signed_vol_10",
    "print_count_5",
    "vol_ratio_5_31",
    "last_print_mid",
    "print_vwap_dist_5",
    "print_vwap_dist_31",
]
CLOCK_FEATURES = [
    "tick_frac",
    "ticks_left",
    "in_window",
    "ticks_to_window",
    "ticks_since_window",
]
OTHER_FEATURES = [
    "other_ret_1",
    "other_ret_5",
    "other_spread",
    "other_imb_touch",
]
FEATURES = PRICE_FEATURES + BOOK_FEATURES + TAPE_FEATURES + CLOCK_FEATURES + OTHER_FEATURES
LABELS = ["y_1", "y_5", "y_31"]
KEYS = ["batch", "run_id", "ticker", "case_tick", "seq"]
AUDIT = ["mid", "bid", "ask"]
PANEL_COLUMNS = KEYS + AUDIT + FEATURES + LABELS

assert len(FEATURES) == 80, len(FEATURES)


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _log_size(size: float) -> float:
    if not _finite(size) or size <= 0:
        return np.nan
    return float(np.log(size))


def _imbalance(bid_depth: float, ask_depth: float) -> float:
    if not _finite(bid_depth) or not _finite(ask_depth):
        return np.nan
    total = bid_depth + ask_depth
    if total <= 0:
        return np.nan
    return (bid_depth - ask_depth) / total


def wilder_rsi(mid: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. Both smoothed sides at zero map to 50."""
    delta = mid.diff()
    gain = delta.clip(lower=0.0).to_numpy(dtype=float)
    loss = (-delta).clip(lower=0.0).to_numpy(dtype=float)
    out = np.full(len(mid), np.nan)
    avg_gain = avg_loss = None
    seeded = False
    for i in range(period, len(mid)):
        if not seeded:
            window_g = gain[i - period + 1 : i + 1]
            window_l = loss[i - period + 1 : i + 1]
            if not (np.isfinite(window_g).all() and np.isfinite(window_l).all()):
                continue
            avg_gain = float(window_g.mean())
            avg_loss = float(window_l.mean())
            seeded = True
        else:
            if not (np.isfinite(gain[i]) and np.isfinite(loss[i])):
                seeded = False
                avg_gain = avg_loss = None
                continue
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_gain == 0 and avg_loss == 0:
            out[i] = 50.0
        elif avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(out, index=mid.index)


def variance_ratio(mid: pd.Series) -> pd.Series:
    """var(six 5-tick sums) / (5 * var(30 one-tick changes))."""
    changes = mid.diff().to_numpy(dtype=float)
    out = np.full(len(mid), np.nan)
    for i in range(30, len(mid)):
        window = changes[i - 29 : i + 1]
        if window.size != 30 or not np.isfinite(window).all():
            continue
        block_var = window.reshape(6, 5).sum(axis=1).var(ddof=1)
        one_var = window.var(ddof=1)
        if one_var == 0:
            continue
        out[i] = block_var / (5.0 * one_var)
    return pd.Series(out, index=mid.index)


def ticks_to_window(tick: int) -> float:
    for start, end in WINDOWS:
        if start <= tick <= end:
            return 0.0
    for start in WINDOW_STARTS:
        if tick < start:
            return float(start - tick)
    return np.nan


def ticks_since_window(tick: int) -> float:
    if tick < WINDOW_STARTS[0]:
        return np.nan
    for start, end in WINDOWS:
        if start <= tick <= end:
            return 0.0
    previous = max(end for end in WINDOW_ENDS if end < tick)
    return float(tick - previous)


def in_window(tick: int) -> int:
    return int(any(start <= tick <= end for start, end in WINDOWS))


def load_quotes(con: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        WITH ranked AS (
          SELECT q.batch, q.run_id, q.ticker, t.case_tick, q.seq,
                 q.bid, q.ask, q.bid_size, q.ask_size,
                 ROW_NUMBER() OVER (
                   PARTITION BY q.run_id, q.ticker, t.case_tick
                   ORDER BY q.seq DESC
                 ) AS rn
          FROM quotes_top q
          JOIN ticks t ON q.run_id = t.run_id AND q.seq = t.seq
          WHERE t.case_status = 'ACTIVE'
        )
        SELECT batch, run_id, ticker, case_tick, seq, bid, ask, bid_size, ask_size
        FROM ranked
        WHERE rn = 1
        """,
        con,
    )
    frame["case_tick"] = frame["case_tick"].astype(int)
    frame["seq"] = frame["seq"].astype(int)
    return frame


def load_tape(con: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT run_id, ticker, id, case_tick, price, quantity
        FROM tape
        """,
        con,
    )
    frame["case_tick"] = frame["case_tick"].astype(int)
    return frame


def load_run_meta(con: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT run_id, value AS started_utc
        FROM meta
        WHERE key = 'started_utc'
        """,
        con,
    )
    return frame


def load_book(con: sqlite3.Connection, run_id: str, seqs: list[int]) -> pd.DataFrame:
    columns = ["seq", "ticker", "side", "level_idx", "price", "quantity"]
    if not seqs:
        return pd.DataFrame(columns=columns)
    frames = []
    for offset in range(0, len(seqs), 800):
        chunk = seqs[offset : offset + 800]
        marks = ",".join("?" * len(chunk))
        query = f"""
            SELECT seq, ticker, side, level_idx, price, quantity
            FROM book_levels
            WHERE run_id = ? AND seq IN ({marks})
        """
        frames.append(pd.read_sql_query(query, con, params=(run_id, *chunk)))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def price_block(mid: pd.Series, ticker: str) -> pd.DataFrame:
    out = pd.DataFrame(index=mid.index)
    for lag in (1, 2, 3, 5, 10, 15, 31):
        out[f"ret_{lag}"] = mid - mid.shift(lag)
    for window in (5, 10, 20, 31):
        out[f"sma_dist_{window}"] = mid - mid.rolling(window, min_periods=window).mean()
    ema = {}
    for span in (5, 10, 15, 20):
        ema[span] = mid.ewm(span=span, adjust=False).mean()
    for span in (5, 10, 20):
        out[f"ema_dist_{span}"] = mid - ema[span]
    macd = ema[5] - ema[15]
    out["macd_5_15"] = macd
    out["macd_signal"] = macd - macd.rolling(5, min_periods=5).mean()
    changes = mid.diff()
    for window in (5, 10, 31):
        out[f"rvol_{window}"] = changes.rolling(window, min_periods=window).std()
    for window in (10, 31):
        low = mid.rolling(window, min_periods=window).min()
        high = mid.rolling(window, min_periods=window).max()
        span = high - low
        out[f"range_pos_{window}"] = ((mid - low) / span).where(span > 0)
    sma_20 = mid.rolling(20, min_periods=20).mean()
    std_20 = mid.rolling(20, min_periods=20).std()
    out["boll_20"] = ((mid - sma_20) / std_20).where(std_20 > 0)
    out["rsi_5"] = wilder_rsi(mid, 5)
    out["rsi_14"] = wilder_rsi(mid, 14)
    out["var_ratio_31"] = variance_ratio(mid)
    start = START_PRICE[ticker]
    out["dist_start"] = (mid - start) / start
    out["y_1"] = mid.shift(-1) - mid
    out["y_5"] = mid.shift(-5) - mid
    out["y_31"] = mid.shift(-31) - mid
    return out


def tape_block(grid: pd.Index, tape: pd.DataFrame, mid: pd.Series) -> pd.DataFrame:
    empty = pd.DataFrame(
        {
            "signed": 0.0,
            "quantity": 0.0,
            "prints": 0.0,
            "notional": 0.0,
            "last_price": np.nan,
        },
        index=grid,
    )
    if not tape.empty:
        last = (
            tape.sort_values("id")
            .groupby("case_tick", sort=False)
            .tail(1)
            .set_index("case_tick")["price"]
        )
        grouped = tape.groupby("case_tick", sort=False)
        filled = pd.DataFrame(
            {
                "signed": grouped["signed"].sum(),
                "quantity": grouped["quantity"].sum(),
                "prints": grouped["quantity"].size(),
                "notional": grouped["notional"].sum(),
            }
        )
        filled["last_price"] = last
        empty.update(filled.reindex(grid))
        empty["signed"] = empty["signed"].fillna(0.0)
        empty["quantity"] = empty["quantity"].fillna(0.0)
        empty["prints"] = empty["prints"].fillna(0.0)
        empty["notional"] = empty["notional"].fillna(0.0)
        empty["last_price"] = filled["last_price"].reindex(grid).ffill()

    out = pd.DataFrame(index=grid)
    for window in (1, 5, 10):
        out[f"signed_vol_{window}"] = (
            empty["signed"].rolling(window, min_periods=window).sum() / SIGNED_SCALE
        )
    out["print_count_5"] = empty["prints"].rolling(5, min_periods=5).sum()
    vol_5 = empty["quantity"].rolling(5, min_periods=5).sum()
    vol_31 = empty["quantity"].rolling(31, min_periods=31).sum()
    out["vol_ratio_5_31"] = (vol_5 / vol_31).where(vol_31 > 0)
    out["last_print_mid"] = empty["last_price"] - mid
    for window in (5, 31):
        notion = empty["notional"].rolling(window, min_periods=window).sum()
        volume = empty["quantity"].rolling(window, min_periods=window).sum()
        out[f"print_vwap_dist_{window}"] = (notion / volume - mid).where(volume > 0)
    return out


def _depth(price: np.ndarray, quantity: np.ndarray, touch: float, side: str, cents: float | None) -> float:
    if price.size == 0 or not _finite(touch):
        return np.nan
    if cents is None:
        return float(quantity.sum())
    width = cents / 100.0
    if side == "bid":
        keep = touch - price <= width + 1e-9
    else:
        keep = price - touch <= width + 1e-9
    if not keep.any():
        return 0.0
    return float(quantity[keep].sum())


def _level_distance(price: np.ndarray, level_idx: np.ndarray, touch: float, side: str, level: int) -> float:
    if price.size == 0 or not _finite(touch):
        return np.nan
    match = np.flatnonzero(level_idx == level)
    if match.size == 0:
        return np.nan
    level_price = float(price[match[0]])
    if side == "bid":
        return touch - level_price
    return level_price - touch


def _walk(price: np.ndarray, quantity: np.ndarray, target: float) -> float:
    if price.size == 0 or target <= 0:
        return np.nan
    remaining = target
    notional = 0.0
    filled = 0.0
    touch = float(price[0])
    for level_price, level_qty in zip(price, quantity):
        take = min(float(level_qty), remaining)
        if take <= 0:
            continue
        notional += float(level_price) * take
        filled += take
        remaining -= take
        if remaining <= 1e-6:
            break
    if filled + 1e-6 < target:
        return np.nan
    return notional / target - touch


def _pack_side(book: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    packed = {}
    if book.empty:
        return packed
    for (seq, side), part in book.groupby(["seq", "side"], sort=False):
        part = part.sort_values("level_idx")
        packed[(int(seq), str(side))] = (
            part["price"].to_numpy(dtype=float),
            part["quantity"].to_numpy(dtype=float),
            part["level_idx"].to_numpy(dtype=int),
        )
    return packed


def book_row(quote: pd.Series, packed: dict, ticker: str) -> dict:
    bid = float(quote["bid"]) if _finite(quote["bid"]) else np.nan
    ask = float(quote["ask"]) if _finite(quote["ask"]) else np.nan
    bid_sz = float(quote["bid_size"]) if _finite(quote["bid_size"]) else np.nan
    ask_sz = float(quote["ask_size"]) if _finite(quote["ask_size"]) else np.nan
    mid = (bid + ask) / 2.0 if _finite(bid) and _finite(ask) and bid > 0 and ask > 0 else np.nan
    spread = ask - bid if _finite(mid) else np.nan
    row = {
        "mid": mid,
        "bid": bid if _finite(bid) else np.nan,
        "ask": ask if _finite(ask) else np.nan,
        "spread": spread,
        "spread_mid": spread / mid if _finite(spread) and _finite(mid) and mid != 0 else np.nan,
        "bid_sz": bid_sz,
        "ask_sz": ask_sz,
        "log_bid_sz": _log_size(bid_sz),
        "log_ask_sz": _log_size(ask_sz),
        "imb_touch": _imbalance(bid_sz, ask_sz),
        "micro_gap": np.nan,
    }
    if _finite(bid) and _finite(ask) and _finite(bid_sz) and _finite(ask_sz) and bid_sz + ask_sz > 0 and _finite(mid):
        micro = (ask * bid_sz + bid * ask_sz) / (bid_sz + ask_sz)
        row["micro_gap"] = micro - mid

    empty = (np.empty(0), np.empty(0), np.empty(0, dtype=int))
    seq = int(quote["seq"])
    bid_px, bid_qty, bid_idx = packed.get((seq, "bid"), empty)
    ask_px, ask_qty, ask_idx = packed.get((seq, "ask"), empty)
    book_bid = float(bid_px[0]) if bid_px.size else bid
    book_ask = float(ask_px[0]) if ask_px.size else ask
    for cents, name in zip((1, 2, 5, 10, None), ("1c", "2c", "5c", "10c", "all")):
        row[f"depth_bid_{name}"] = _depth(bid_px, bid_qty, book_bid, "bid", cents)
        row[f"depth_ask_{name}"] = _depth(ask_px, ask_qty, book_ask, "ask", cents)
        total = row[f"depth_bid_{name}"] + row[f"depth_ask_{name}"]
        row[f"log_depth_{name}"] = np.log(total) if _finite(total) and total > 0 else np.nan
    for name in ("5c", "10c", "all"):
        row[f"imb_{name}"] = _imbalance(row[f"depth_bid_{name}"], row[f"depth_ask_{name}"])
    for level in (5, 10):
        row[f"dist_bid_l{level}"] = _level_distance(bid_px, bid_idx, book_bid, "bid", level)
        row[f"dist_ask_l{level}"] = _level_distance(ask_px, ask_idx, book_ask, "ask", level)
    row["conc_bid"] = (
        bid_sz / row["depth_bid_10c"]
        if _finite(bid_sz) and _finite(row["depth_bid_10c"]) and row["depth_bid_10c"] > 0
        else np.nan
    )
    row["conc_ask"] = (
        ask_sz / row["depth_ask_10c"]
        if _finite(ask_sz) and _finite(row["depth_ask_10c"]) and row["depth_ask_10c"] > 0
        else np.nan
    )
    target = MAX_ORDER[ticker]
    row["walk_bid"] = _walk(bid_px, bid_qty, target)
    row["walk_ask"] = _walk(ask_px, ask_qty, target)
    return row


def clock_block(ticks: pd.Index) -> pd.DataFrame:
    values = ticks.to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "tick_frac": values / 299.0,
            "ticks_left": 299 - values,
            "in_window": [in_window(int(tick)) for tick in values],
            "ticks_to_window": [ticks_to_window(int(tick)) for tick in values],
            "ticks_since_window": [ticks_since_window(int(tick)) for tick in values],
        },
        index=ticks,
    )


def build_series(
    quotes: pd.DataFrame,
    tape: pd.DataFrame,
    book: pd.DataFrame,
) -> pd.DataFrame:
    ticker = str(quotes["ticker"].iloc[0])
    observed = quotes.set_index("case_tick").sort_index()
    start = int(observed.index.min())
    stop = int(observed.index.max())
    grid = pd.RangeIndex(start, stop + 1, name="case_tick")

    bid = observed["bid"].reindex(grid).astype(float)
    ask = observed["ask"].reindex(grid).astype(float)
    valid = bid.gt(0) & ask.gt(0)
    mid = ((bid + ask) / 2.0).where(valid)

    frame = price_block(mid, ticker)
    frame = frame.join(tape_block(grid, tape, mid))
    book_rows = []
    book_index = []
    packed = _pack_side(book)
    for tick, quote in observed.iterrows():
        book_rows.append(book_row(quote, packed, ticker))
        book_index.append(int(tick))
    book_frame = pd.DataFrame(book_rows, index=pd.Index(book_index, name="case_tick"))
    book_frame = book_frame.reindex(grid)
    depth_5 = book_frame["depth_bid_5c"] + book_frame["depth_ask_5c"]
    book_frame["d_spread"] = book_frame["spread"].diff()
    book_frame["d_imb"] = book_frame["imb_touch"].diff()
    book_frame["d_depth_5c"] = depth_5.diff()
    frame = frame.join(book_frame)
    frame = frame.join(clock_block(grid))
    frame["batch"] = observed["batch"].iloc[0]
    frame["run_id"] = observed["run_id"].iloc[0]
    frame["ticker"] = ticker
    frame["seq"] = observed["seq"].reindex(grid)
    keep = frame.loc[observed.index].copy()
    keep.index.name = "case_tick"
    keep = keep.reset_index()
    keep["seq"] = keep["seq"].astype(int)
    keep["case_tick"] = keep["case_tick"].astype(int)
    keep["in_window"] = keep["in_window"].astype(int)
    return keep


def add_other_name(panel: pd.DataFrame) -> pd.DataFrame:
    own = panel[["run_id", "case_tick", "ticker", "ret_1", "ret_5", "spread", "imb_touch"]]
    crzy = own.loc[own["ticker"] == "CRZY"].drop(columns="ticker")
    tame = own.loc[own["ticker"] == "TAME"].drop(columns="ticker")
    renamed = {
        "ret_1": "other_ret_1",
        "ret_5": "other_ret_5",
        "spread": "other_spread",
        "imb_touch": "other_imb_touch",
    }
    crzy_as_other = crzy.rename(columns=renamed)
    tame_as_other = tame.rename(columns=renamed)
    left = panel.loc[panel["ticker"] == "CRZY"].merge(
        tame_as_other, on=["run_id", "case_tick"], how="left"
    )
    right = panel.loc[panel["ticker"] == "TAME"].merge(
        crzy_as_other, on=["run_id", "case_tick"], how="left"
    )
    return pd.concat([left, right], ignore_index=True)


def classify_tape(tape: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    quote_cols = quotes[["run_id", "ticker", "case_tick", "bid", "ask"]]
    merged = tape.merge(quote_cols, on=["run_id", "ticker", "case_tick"], how="left")
    bid = merged["bid"].to_numpy(dtype=float)
    ask = merged["ask"].to_numpy(dtype=float)
    price = merged["price"].to_numpy(dtype=float)
    quantity = merged["quantity"].to_numpy(dtype=float)
    signed = np.zeros(len(merged), dtype=float)
    tradable = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    signed = np.where(tradable & (price >= ask), quantity, signed)
    signed = np.where(tradable & (price <= bid) & ~(price >= ask), -quantity, signed)
    merged["signed"] = signed
    merged["notional"] = price * quantity
    return merged


def build_panel(src: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes = load_quotes(src)
    tape = classify_tape(load_tape(src), quotes)
    meta = load_run_meta(src)
    pieces = []
    run_ids = sorted(quotes["run_id"].unique())
    for index, run_id in enumerate(run_ids, start=1):
        run_quotes = quotes.loc[quotes["run_id"] == run_id]
        seqs = sorted(int(seq) for seq in run_quotes["seq"].unique())
        book = load_book(src, run_id, seqs)
        run_tape = tape.loc[tape["run_id"] == run_id]
        for ticker, quote_part in run_quotes.groupby("ticker", sort=False):
            tape_part = run_tape.loc[run_tape["ticker"] == ticker]
            book_part = book.loc[book["ticker"] == ticker] if not book.empty else book
            pieces.append(build_series(quote_part, tape_part, book_part))
        print(f"built {index}/{len(run_ids)} {run_id}", file=sys.stderr)
    panel = add_other_name(pd.concat(pieces, ignore_index=True))
    panel = panel[PANEL_COLUMNS]
    runs = (
        panel.groupby("run_id", sort=False)
        .agg(batch=("batch", "first"), max_case_tick=("case_tick", "max"))
        .reset_index()
    )
    runs = runs.merge(meta, on="run_id", how="left")
    runs = runs.sort_values(["started_utc", "run_id"], kind="mergesort").reset_index(drop=True)
    if len(runs) != 100:
        raise RuntimeError(f"expected 100 runs, found {len(runs)}")
    splits = np.array(["train"] * 60 + ["validation"] * 20 + ["test"] * 20)
    runs["split"] = splits
    runs = runs[["run_id", "batch", "started_utc", "max_case_tick", "split"]]
    return panel, runs


def catalog_rows() -> list[tuple[str, str, str]]:
    rows = [
        ("ret_1", "price", "Mid change over the previous 1 case tick."),
        ("ret_2", "price", "Mid change over the previous 2 case ticks."),
        ("ret_3", "price", "Mid change over the previous 3 case ticks."),
        ("ret_5", "price", "Mid change over the previous 5 case ticks."),
        ("ret_10", "price", "Mid change over the previous 10 case ticks."),
        ("ret_15", "price", "Mid change over the previous 15 case ticks."),
        ("ret_31", "price", "Mid change over the previous 31 case ticks."),
        ("sma_dist_5", "price", "Mid minus the 5-tick simple moving average."),
        ("sma_dist_10", "price", "Mid minus the 10-tick simple moving average."),
        ("sma_dist_20", "price", "Mid minus the 20-tick simple moving average."),
        ("sma_dist_31", "price", "Mid minus the 31-tick simple moving average."),
        ("ema_dist_5", "price", "Mid minus the 5-tick exponential moving average."),
        ("ema_dist_10", "price", "Mid minus the 10-tick exponential moving average."),
        ("ema_dist_20", "price", "Mid minus the 20-tick exponential moving average."),
        ("macd_5_15", "price", "EMA(5) minus EMA(15) of the mid."),
        ("macd_signal", "price", "macd_5_15 minus its 5-tick average."),
        ("rvol_5", "price", "Standard deviation of the last 5 one-tick mid changes."),
        ("rvol_10", "price", "Standard deviation of the last 10 one-tick mid changes."),
        ("rvol_31", "price", "Standard deviation of the last 31 one-tick mid changes."),
        ("range_pos_10", "price", "Position of the mid inside the last 10 ticks' high-low range."),
        ("range_pos_31", "price", "Position of the mid inside the last 31 ticks' high-low range."),
        ("boll_20", "price", "(Mid minus SMA(20)) divided by the 20-tick standard deviation of the mid."),
        ("rsi_5", "price", "Wilder RSI of the mid with period 5. Zero gain and loss maps to 50."),
        ("rsi_14", "price", "Wilder RSI of the mid with period 14. Zero gain and loss maps to 50."),
        (
            "var_ratio_31",
            "price",
            "Variance of six non-overlapping 5-tick sums divided by 5 times the variance of the last 30 one-tick changes.",
        ),
        ("dist_start", "price", "(Mid minus start price) / start price. Start is 10 for CRZY and 25 for TAME."),
        ("spread", "book", "Ask minus bid, from the end-of-tick quote."),
        ("spread_mid", "book", "Spread divided by mid."),
        ("bid_sz", "book", "Size at the bid."),
        ("ask_sz", "book", "Size at the ask."),
        ("log_bid_sz", "book", "Natural log of bid size. Null when size is not positive."),
        ("log_ask_sz", "book", "Natural log of ask size. Null when size is not positive."),
        ("imb_touch", "book", "(Bid size minus ask size) / (bid size plus ask size)."),
        ("micro_gap", "book", "Size-weighted microprice minus mid."),
        ("depth_bid_1c", "book", "Bid size within 1 cent of the touch, including the touch."),
        ("depth_bid_2c", "book", "Bid size within 2 cents of the touch, including the touch."),
        ("depth_bid_5c", "book", "Bid size within 5 cents of the touch, including the touch."),
        ("depth_bid_10c", "book", "Bid size within 10 cents of the touch, including the touch."),
        ("depth_bid_all", "book", "All visible bid size."),
        ("depth_ask_1c", "book", "Ask size within 1 cent of the touch, including the touch."),
        ("depth_ask_2c", "book", "Ask size within 2 cents of the touch, including the touch."),
        ("depth_ask_5c", "book", "Ask size within 5 cents of the touch, including the touch."),
        ("depth_ask_10c", "book", "Ask size within 10 cents of the touch, including the touch."),
        ("depth_ask_all", "book", "All visible ask size."),
        ("log_depth_1c", "book", "Natural log of bid plus ask size within 1 cent."),
        ("log_depth_2c", "book", "Natural log of bid plus ask size within 2 cents."),
        ("log_depth_5c", "book", "Natural log of bid plus ask size within 5 cents."),
        ("log_depth_10c", "book", "Natural log of bid plus ask size within 10 cents."),
        ("log_depth_all", "book", "Natural log of all visible bid plus ask size."),
        ("imb_5c", "book", "Depth imbalance within 5 cents."),
        ("imb_10c", "book", "Depth imbalance within 10 cents."),
        ("imb_all", "book", "Imbalance of all visible depth."),
        ("dist_bid_l5", "book", "Bid touch minus the price at bid level 5."),
        ("dist_bid_l10", "book", "Bid touch minus the price at bid level 10."),
        ("dist_ask_l5", "book", "Ask level 5 price minus the ask touch."),
        ("dist_ask_l10", "book", "Ask level 10 price minus the ask touch."),
        ("conc_bid", "book", "Bid touch size divided by bid depth within 10 cents."),
        ("conc_ask", "book", "Ask touch size divided by ask depth within 10 cents."),
        (
            "walk_bid",
            "book",
            "Bid-side VWAP minus the bid touch for one max order (25,000 CRZY or 10,000 TAME). Null if the book is too thin.",
        ),
        (
            "walk_ask",
            "book",
            "Ask-side VWAP minus the ask touch for one max order (25,000 CRZY or 10,000 TAME). Null if the book is too thin.",
        ),
        ("d_spread", "book", "One-tick change in spread."),
        ("d_imb", "book", "One-tick change in touch imbalance."),
        ("d_depth_5c", "book", "One-tick change in total depth within 5 cents."),
        ("signed_vol_1", "tape", "Signed print volume this tick, divided by 70,000. Ask prints are positive."),
        ("signed_vol_5", "tape", "Signed print volume over 5 ticks, divided by 70,000."),
        ("signed_vol_10", "tape", "Signed print volume over 10 ticks, divided by 70,000."),
        ("print_count_5", "tape", "Number of tape prints over the last 5 ticks."),
        ("vol_ratio_5_31", "tape", "Print volume over 5 ticks divided by print volume over 31 ticks."),
        ("last_print_mid", "tape", "Most recent print at or before this tick, minus the mid."),
        ("print_vwap_dist_5", "tape", "Print VWAP over 5 ticks minus the mid."),
        ("print_vwap_dist_31", "tape", "Print VWAP over 31 ticks minus the mid."),
        ("tick_frac", "clock", "case_tick / 299."),
        ("ticks_left", "clock", "299 minus case_tick."),
        ("in_window", "clock", "1 when the tick is inside a tender arrival window, else 0."),
        ("ticks_to_window", "clock", "0 inside a window, else ticks until the next window opens. Null after tick 269."),
        ("ticks_since_window", "clock", "Null before tick 14, 0 inside a window, else ticks since the previous window closed."),
        ("other_ret_1", "cross", "The other ticker's ret_1 on the same run and tick."),
        ("other_ret_5", "cross", "The other ticker's ret_5 on the same run and tick."),
        ("other_spread", "cross", "The other ticker's spread on the same run and tick."),
        ("other_imb_touch", "cross", "The other ticker's touch imbalance on the same run and tick."),
        ("y_1", "label", "Mid one tick ahead minus the current mid."),
        ("y_5", "label", "Mid five ticks ahead minus the current mid."),
        ("y_31", "label", "Mid 31 ticks ahead minus the current mid."),
    ]
    names = [name for name, _, _ in rows]
    expected = FEATURES + LABELS
    if names != expected:
        raise RuntimeError("feature catalog does not match the column list")
    return rows


def _sql_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_database(panel: pd.DataFrame, runs: pd.DataFrame) -> None:
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    con = sqlite3.connect(OUT_PATH)
    try:
        feature_sql = ",\n          ".join(f"{name} REAL" for name in FEATURES)
        label_sql = ",\n          ".join(f"{name} REAL" for name in LABELS)
        con.execute("BEGIN")
        con.execute(
            f"""
            CREATE TABLE panel (
              batch TEXT NOT NULL,
              run_id TEXT NOT NULL,
              ticker TEXT NOT NULL,
              case_tick INTEGER NOT NULL,
              seq INTEGER NOT NULL,
              mid REAL,
              bid REAL,
              ask REAL,
              {feature_sql},
              {label_sql},
              PRIMARY KEY (run_id, ticker, case_tick)
            )
            """
        )
        con.execute("CREATE INDEX idx_panel_ticker_tick ON panel (ticker, case_tick)")
        con.execute(
            """
            CREATE TABLE runs (
              run_id TEXT PRIMARY KEY,
              batch TEXT NOT NULL,
              started_utc TEXT,
              max_case_tick INTEGER NOT NULL,
              split TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE feature_catalog (
              name TEXT PRIMARY KEY,
              family TEXT NOT NULL,
              definition TEXT NOT NULL
            )
            """
        )
        rows = [tuple(_sql_value(value) for value in record) for record in panel.itertuples(index=False, name=None)]
        marks = ",".join("?" * len(PANEL_COLUMNS))
        con.executemany(f"INSERT INTO panel VALUES ({marks})", rows)
        con.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            list(runs.itertuples(index=False, name=None)),
        )
        con.executemany("INSERT INTO feature_catalog VALUES (?, ?, ?)", catalog_rows())
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def verify(src: sqlite3.Connection) -> None:
    con = sqlite3.connect(f"file:{OUT_PATH}?mode=ro", uri=True)
    failures = []

    def check(condition: bool, message: str) -> None:
        print(("OK  " if condition else "FAIL") + " " + message)
        if not condition:
            failures.append(message)

    panel = pd.read_sql_query("SELECT * FROM panel", con)
    runs = pd.read_sql_query("SELECT * FROM runs", con)
    catalog = pd.read_sql_query("SELECT name, family FROM feature_catalog", con)
    source_rows = load_quotes(src)

    series = panel.groupby(["run_id", "ticker"], sort=False).size()
    check(len(series) == 200, f"series count is {len(series)}")
    check(len(panel) == len(source_rows), f"panel rows {len(panel)} vs source rows {len(source_rows)}")
    check(set(catalog["name"]) == set(FEATURES + LABELS), "catalog covers features and labels")
    check(len(FEATURES) == 80, "feature column count is 80")
    check(list(runs["split"]).count("train") == 60, "60 train runs")
    check(list(runs["split"]).count("validation") == 20, "20 validation runs")
    check(list(runs["split"]).count("test") == 20, "20 test runs")
    ordered = runs.sort_values(["started_utc", "run_id"], kind="mergesort")["split"].tolist()
    check(ordered == ["train"] * 60 + ["validation"] * 20 + ["test"] * 20, "split follows started_utc order")

    early = panel.loc[panel["case_tick"] <= 30, "ret_31"]
    check(early.isna().all(), f"ret_31 null on ticks 0-30 ({int(early.isna().sum())}/{len(early)})")

    label_mismatch = 0
    y31_tail_bad = 0
    for (_, _), group in panel.groupby(["run_id", "ticker"], sort=False):
        group = group.sort_values("case_tick")
        mids = {int(tick): mid for tick, mid in zip(group["case_tick"], group["mid"])}
        ticks = [int(tick) for tick in group["case_tick"]]
        tail = set(sorted(ticks)[-31:])
        stored = group.set_index("case_tick")
        for tick in ticks:
            mid = mids[tick]
            ret = stored.at[tick, "ret_31"]
            previous = mids.get(tick - 31, np.nan)
            if tick <= 30:
                if _finite(ret):
                    label_mismatch += 1
            elif _finite(mid) and _finite(previous):
                if not _finite(ret) or abs(float(ret) - (float(mid) - float(previous))) > 1e-8:
                    label_mismatch += 1
            elif _finite(ret):
                label_mismatch += 1
            future = mids.get(tick + 31, np.nan)
            y31 = stored.at[tick, "y_31"]
            if tick in tail:
                if _finite(y31):
                    y31_tail_bad += 1
            elif _finite(mid) and _finite(future):
                if not _finite(y31) or abs(float(y31) - (float(future) - float(mid))) > 1e-8:
                    label_mismatch += 1
            y5 = stored.at[tick, "y_5"]
            ahead = mids.get(tick + 5, np.nan)
            if _finite(mid) and _finite(ahead):
                if not _finite(y5) or abs(float(y5) - (float(ahead) - float(mid))) > 1e-8:
                    label_mismatch += 1
            elif _finite(y5):
                label_mismatch += 1

    check(label_mismatch == 0, f"labels and ret_31 match mids inside each series ({label_mismatch} mismatches)")
    check(y31_tail_bad == 0, f"y_31 null on the last 31 ticks of each series ({y31_tail_bad} filled)")

    spread = panel["spread"].dropna()
    spread_median = float(spread.median())
    check(0.02 <= spread_median <= 0.04, f"median spread {spread_median:.4f}")
    ret_abs = panel["ret_1"].abs().dropna()
    ret_p75 = float(ret_abs.quantile(0.75))
    check(0.005 <= ret_p75 <= 0.05, f"75th percentile of |ret_1| is {ret_p75:.4f}")
    con.close()
    if failures:
        raise SystemExit("verification failed:\n- " + "\n- ".join(failures))
    print("verification passed")


def main() -> None:
    src = sqlite3.connect(f"file:{SRC_PATH}?mode=ro", uri=True)
    try:
        panel, runs = build_panel(src)
        write_database(panel, runs)
        print(f"wrote {OUT_PATH} rows={len(panel)} runs={len(runs)}")
        verify(src)
    finally:
        src.close()


if __name__ == "__main__":
    main()
