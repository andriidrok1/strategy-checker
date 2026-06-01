"""Chart pattern detection. Foundation: pivot points → geometric pattern matching.

Each detector returns list of Pattern dicts with:
- type: pattern name
- entry_date: confirmation candle index
- entry_price: confirmation close
- direction: 1 (long) or -1 (short)
- stop_loss: absolute price for SL
- take_profit: absolute price for TP (measured move)
- meta: extra info for debugging (pivot indices, neckline, etc)

Detectors then converted to vectorbt-compatible entry/exit signals.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class Pattern:
    type: str
    entry_date: pd.Timestamp
    entry_price: float
    direction: int  # 1 long, -1 short
    stop_loss: float
    take_profit: float
    meta: dict


# ─── Pivot Detection ────────────────────────────────────────────────

def find_pivots(df: pd.DataFrame, prominence_pct: float = 0.02, distance: int = 5) -> pd.DataFrame:
    """Detect swing highs and lows.

    prominence_pct — minimum height of pivot relative to mean price (0.02 = 2%)
    distance — minimum bars between pivots of same type

    Returns DataFrame with: idx (int position), date, price, type ('high'|'low').
    """
    high_arr = df["high"].values
    low_arr = df["low"].values
    mean_price = float(df["close"].mean())
    prom = mean_price * prominence_pct

    high_idx, _ = find_peaks(high_arr, prominence=prom, distance=distance)
    low_idx, _ = find_peaks(-low_arr, prominence=prom, distance=distance)

    rows = []
    for i in high_idx:
        rows.append({"idx": int(i), "date": df.index[i], "price": float(high_arr[i]), "type": "high"})
    for i in low_idx:
        rows.append({"idx": int(i), "date": df.index[i], "price": float(low_arr[i]), "type": "low"})

    pivots = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["idx", "date", "price", "type"]
    )
    return pivots


# ─── Double Top ─────────────────────────────────────────────────────

def detect_double_top(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    height_tol: float = 0.03,
    min_separation: int = 10,
    max_separation: int = 100,
    confirm_window: int = 50,
) -> list[Pattern]:
    """Double top: 2 consecutive high pivots at similar price, confirmed by close below neckline.

    height_tol — peaks must be within this % of each other
    min/max_separation — bars between the two peaks
    confirm_window — how many bars after peak2 to wait for neckline break
    """
    patterns = []
    highs = pivots[pivots["type"] == "high"].reset_index(drop=True)

    for i in range(len(highs) - 1):
        p1, p2 = highs.iloc[i], highs.iloc[i + 1]
        sep = p2["idx"] - p1["idx"]

        if sep < min_separation or sep > max_separation:
            continue

        # Heights similar?
        height_diff = abs(p1["price"] - p2["price"]) / p1["price"]
        if height_diff > height_tol:
            continue

        # Neckline = lowest low between peaks
        between = df.iloc[p1["idx"]:p2["idx"] + 1]
        neckline = float(between["low"].min())

        # Confirmation: first close below neckline within confirm_window after p2
        end_idx = min(p2["idx"] + 1 + confirm_window, len(df))
        after = df.iloc[p2["idx"] + 1:end_idx]
        below = after[after["close"] < neckline]
        if below.empty:
            continue

        entry_date = below.index[0]
        entry_price = float(below["close"].iloc[0])
        peak_high = max(p1["price"], p2["price"])

        # Measured move: pattern height projected down from neckline
        target = neckline - (peak_high - neckline)

        patterns.append(Pattern(
            type="double_top",
            entry_date=entry_date,
            entry_price=entry_price,
            direction=-1,
            stop_loss=peak_high * 1.005,  # 0.5% above highest peak
            take_profit=target,
            meta={
                "p1_date": p1["date"], "p1_price": p1["price"],
                "p2_date": p2["date"], "p2_price": p2["price"],
                "neckline": neckline,
            },
        ))

    return patterns


# ─── Double Bottom ──────────────────────────────────────────────────

def detect_double_bottom(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    height_tol: float = 0.03,
    min_separation: int = 10,
    max_separation: int = 100,
    confirm_window: int = 50,
) -> list[Pattern]:
    """Double bottom: 2 consecutive low pivots at similar price, confirmed by close above neckline."""
    patterns = []
    lows = pivots[pivots["type"] == "low"].reset_index(drop=True)

    for i in range(len(lows) - 1):
        p1, p2 = lows.iloc[i], lows.iloc[i + 1]
        sep = p2["idx"] - p1["idx"]

        if sep < min_separation or sep > max_separation:
            continue

        height_diff = abs(p1["price"] - p2["price"]) / p1["price"]
        if height_diff > height_tol:
            continue

        # Neckline = highest high between troughs
        between = df.iloc[p1["idx"]:p2["idx"] + 1]
        neckline = float(between["high"].max())

        # Confirmation: first close above neckline
        end_idx = min(p2["idx"] + 1 + confirm_window, len(df))
        after = df.iloc[p2["idx"] + 1:end_idx]
        above = after[after["close"] > neckline]
        if above.empty:
            continue

        entry_date = above.index[0]
        entry_price = float(above["close"].iloc[0])
        trough_low = min(p1["price"], p2["price"])

        # Measured move: pattern height projected up from neckline
        target = neckline + (neckline - trough_low)

        patterns.append(Pattern(
            type="double_bottom",
            entry_date=entry_date,
            entry_price=entry_price,
            direction=1,
            stop_loss=trough_low * 0.995,
            take_profit=target,
            meta={
                "p1_date": p1["date"], "p1_price": p1["price"],
                "p2_date": p2["date"], "p2_price": p2["price"],
                "neckline": neckline,
            },
        ))

    return patterns


# ─── Head and Shoulders ─────────────────────────────────────────────

def detect_head_shoulders(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    shoulder_tol: float = 0.04,
    head_premium: float = 0.02,
    min_separation: int = 5,
    max_total_span: int = 150,
    confirm_window: int = 50,
) -> list[Pattern]:
    """Head & Shoulders (bearish reversal): 3 high pivots, middle higher.

    shoulder_tol — left/right shoulders within this % of each other
    head_premium — head must exceed shoulders by at least this %
    """
    patterns = []
    highs = pivots[pivots["type"] == "high"].reset_index(drop=True)
    lows = pivots[pivots["type"] == "low"].reset_index(drop=True)

    for i in range(len(highs) - 2):
        ls, h, rs = highs.iloc[i], highs.iloc[i + 1], highs.iloc[i + 2]

        if h["idx"] - ls["idx"] < min_separation or rs["idx"] - h["idx"] < min_separation:
            continue
        if rs["idx"] - ls["idx"] > max_total_span:
            continue

        # Head must be highest
        if h["price"] <= ls["price"] * (1 + head_premium):
            continue
        if h["price"] <= rs["price"] * (1 + head_premium):
            continue

        # Shoulders similar
        if abs(ls["price"] - rs["price"]) / ls["price"] > shoulder_tol:
            continue

        # Neckline from two lows: one between LS-H, one between H-RS
        lows_between_1 = lows[(lows["idx"] > ls["idx"]) & (lows["idx"] < h["idx"])]
        lows_between_2 = lows[(lows["idx"] > h["idx"]) & (lows["idx"] < rs["idx"])]
        if lows_between_1.empty or lows_between_2.empty:
            continue

        n1 = lows_between_1.iloc[lows_between_1["price"].argmin()]
        n2 = lows_between_2.iloc[lows_between_2["price"].argmin()]
        neckline = (n1["price"] + n2["price"]) / 2

        # Confirmation: close below neckline within window after RS
        end_idx = min(rs["idx"] + 1 + confirm_window, len(df))
        after = df.iloc[rs["idx"] + 1:end_idx]
        below = after[after["close"] < neckline]
        if below.empty:
            continue

        entry_date = below.index[0]
        entry_price = float(below["close"].iloc[0])

        # Measured move: head - neckline distance projected down from entry
        head_height = h["price"] - neckline
        target = neckline - head_height

        patterns.append(Pattern(
            type="head_and_shoulders",
            entry_date=entry_date,
            entry_price=entry_price,
            direction=-1,
            stop_loss=h["price"] * 1.005,
            take_profit=target,
            meta={
                "ls_date": ls["date"], "ls_price": ls["price"],
                "head_date": h["date"], "head_price": h["price"],
                "rs_date": rs["date"], "rs_price": rs["price"],
                "neckline": neckline,
            },
        ))

    return patterns


# ─── Inverted Head and Shoulders ────────────────────────────────────

def detect_inverted_head_shoulders(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    shoulder_tol: float = 0.04,
    head_premium: float = 0.02,
    min_separation: int = 5,
    max_total_span: int = 150,
    confirm_window: int = 50,
) -> list[Pattern]:
    """Inverted H&S (bullish reversal): 3 low pivots, middle lower."""
    patterns = []
    lows = pivots[pivots["type"] == "low"].reset_index(drop=True)
    highs = pivots[pivots["type"] == "high"].reset_index(drop=True)

    for i in range(len(lows) - 2):
        ls, h, rs = lows.iloc[i], lows.iloc[i + 1], lows.iloc[i + 2]

        if h["idx"] - ls["idx"] < min_separation or rs["idx"] - h["idx"] < min_separation:
            continue
        if rs["idx"] - ls["idx"] > max_total_span:
            continue

        # Head must be lowest
        if h["price"] >= ls["price"] * (1 - head_premium):
            continue
        if h["price"] >= rs["price"] * (1 - head_premium):
            continue

        # Shoulders similar
        if abs(ls["price"] - rs["price"]) / ls["price"] > shoulder_tol:
            continue

        # Neckline from two highs between
        highs_between_1 = highs[(highs["idx"] > ls["idx"]) & (highs["idx"] < h["idx"])]
        highs_between_2 = highs[(highs["idx"] > h["idx"]) & (highs["idx"] < rs["idx"])]
        if highs_between_1.empty or highs_between_2.empty:
            continue

        n1 = highs_between_1.iloc[highs_between_1["price"].argmax()]
        n2 = highs_between_2.iloc[highs_between_2["price"].argmax()]
        neckline = (n1["price"] + n2["price"]) / 2

        # Confirmation: close above neckline
        end_idx = min(rs["idx"] + 1 + confirm_window, len(df))
        after = df.iloc[rs["idx"] + 1:end_idx]
        above = after[after["close"] > neckline]
        if above.empty:
            continue

        entry_date = above.index[0]
        entry_price = float(above["close"].iloc[0])

        head_depth = neckline - h["price"]
        target = neckline + head_depth

        patterns.append(Pattern(
            type="inverse_head_and_shoulders",
            entry_date=entry_date,
            entry_price=entry_price,
            direction=1,
            stop_loss=h["price"] * 0.995,
            take_profit=target,
            meta={
                "ls_date": ls["date"], "ls_price": ls["price"],
                "head_date": h["date"], "head_price": h["price"],
                "rs_date": rs["date"], "rs_price": rs["price"],
                "neckline": neckline,
            },
        ))

    return patterns


# ─── Triangles (ascending, descending, symmetrical) ─────────────────

def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares line fit. Returns (slope, intercept)."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(slope), float(intercept)


def _dedupe_patterns(patterns: list[Pattern], min_gap_bars: int = 30) -> list[Pattern]:
    """Keep only first pattern within min_gap_bars window — remove near-duplicates."""
    if not patterns:
        return []
    patterns = sorted(patterns, key=lambda p: p.entry_date)
    kept = [patterns[0]]
    for p in patterns[1:]:
        gap = (p.entry_date - kept[-1].entry_date).days
        if gap >= min_gap_bars:
            kept.append(p)
    return kept


def _detect_triangles(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    target_kind: str,
    window_bars: int = 80,
    min_pivots_each: int = 2,
    flat_threshold_pct: float = 0.0005,
    confirm_window: int = 30,
) -> list[Pattern]:
    """Internal: scans for triangle patterns of given kind.

    target_kind: 'ascending' | 'descending' | 'symmetrical'

    Classification (per-bar % slope):
    - ascending: top flat, bottom rising
    - descending: top falling, bottom flat
    - symmetrical: top falling, bottom rising
    """
    patterns = []

    if len(pivots) < min_pivots_each * 2:
        return patterns

    for end_i in range(min_pivots_each * 2, len(pivots)):
        end_pos = int(pivots.iloc[end_i]["idx"])
        start_pos = max(0, end_pos - window_bars)

        window = pivots[(pivots["idx"] >= start_pos) & (pivots["idx"] <= end_pos)]
        highs_w = window[window["type"] == "high"]
        lows_w = window[window["type"] == "low"]

        if len(highs_w) < min_pivots_each or len(lows_w) < min_pivots_each:
            continue

        top_slope, top_intercept = _fit_line(highs_w["idx"].values, highs_w["price"].values)
        bot_slope, bot_intercept = _fit_line(lows_w["idx"].values, lows_w["price"].values)

        mean_price = (highs_w["price"].mean() + lows_w["price"].mean()) / 2
        top_pct = top_slope / mean_price
        bot_pct = bot_slope / mean_price

        # Classify
        if target_kind == "ascending":
            if not (abs(top_pct) < flat_threshold_pct and bot_pct > flat_threshold_pct):
                continue
            expected_dir = 1
        elif target_kind == "descending":
            if not (top_pct < -flat_threshold_pct and abs(bot_pct) < flat_threshold_pct):
                continue
            expected_dir = -1
        elif target_kind == "symmetrical":
            if not (top_pct < -flat_threshold_pct and bot_pct > flat_threshold_pct):
                continue
            expected_dir = 0  # any breakout direction
        else:
            continue

        # Look for breakout
        end_j = min(end_pos + 1 + confirm_window, len(df))
        for j in range(end_pos + 1, end_j):
            top_line = top_intercept + top_slope * j
            bot_line = bot_intercept + bot_slope * j
            close_j = float(df["close"].iloc[j])
            high_j = float(df["high"].iloc[j])
            low_j = float(df["low"].iloc[j])

            direction = 0
            broken_line = 0.0
            if close_j > top_line:
                direction = 1
                broken_line = top_line
            elif close_j < bot_line:
                direction = -1
                broken_line = bot_line

            if direction == 0:
                continue

            # For ascending/descending: only accept expected breakout direction
            if expected_dir != 0 and direction != expected_dir:
                # Wrong-direction breakout = pattern failed, skip
                break

            # Triangle height at pattern start (widest part)
            top_at_start = top_intercept + top_slope * start_pos
            bot_at_start = bot_intercept + bot_slope * start_pos
            height = abs(top_at_start - bot_at_start)

            entry_date = df.index[j]
            entry_price = close_j

            if direction == 1:
                target = entry_price + height
                stop = bot_line * 0.995
            else:
                target = entry_price - height
                stop = top_line * 1.005

            patterns.append(Pattern(
                type=f"{target_kind}_triangle",
                entry_date=entry_date,
                entry_price=entry_price,
                direction=direction,
                stop_loss=stop,
                take_profit=target,
                meta={
                    "top_slope_pct_per_bar": top_pct,
                    "bot_slope_pct_per_bar": bot_pct,
                    "window_start": df.index[start_pos],
                    "window_end": df.index[end_pos],
                    "breakout_line": broken_line,
                },
            ))
            break  # one pattern per window

    return _dedupe_patterns(patterns, min_gap_bars=window_bars // 2)


def detect_ascending_triangle(df, pivots, **kwargs):
    return _detect_triangles(df, pivots, "ascending", **kwargs)


def detect_descending_triangle(df, pivots, **kwargs):
    return _detect_triangles(df, pivots, "descending", **kwargs)


def detect_symmetrical_triangle(df, pivots, **kwargs):
    return _detect_triangles(df, pivots, "symmetrical", **kwargs)


# ─── Flags ──────────────────────────────────────────────────────────

def _detect_flags(
    df: pd.DataFrame,
    pivots: pd.DataFrame,  # unused for flags (movement-based not pivot-based)
    direction: int,  # 1 = bullish, -1 = bearish
    pole_lookback: int = 15,
    pole_min_pct: float = 0.10,
    flag_window: int = 20,
    flag_max_width_pct: float = 0.05,
    confirm_window: int = 15,
    min_gap_bars: int = 30,
) -> list[Pattern]:
    """Detect flag/pennant continuation patterns.

    Bullish flag (direction=1): strong up-move (pole), then tight consolidation,
    then breakout above flag highs.

    Bearish flag (direction=-1): mirror — strong down-move, consolidation, break below.

    pole_min_pct: minimum % move over pole_lookback bars (10% default)
    flag_max_width_pct: consolidation range must be tighter than this % of pole start price
    """
    patterns = []
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    last_entry_pos = -min_gap_bars  # allow first hit

    for i in range(pole_lookback + flag_window, n - confirm_window):
        # Pole = move from i - pole_lookback - flag_window to i - flag_window
        pole_start = i - pole_lookback - flag_window
        pole_end = i - flag_window
        pole_move = (close[pole_end] - close[pole_start]) / close[pole_start]

        if direction == 1 and pole_move < pole_min_pct:
            continue
        if direction == -1 and pole_move > -pole_min_pct:
            continue

        # Flag = consolidation from pole_end to i
        flag_highs = high[pole_end:i]
        flag_lows = low[pole_end:i]
        flag_top = float(flag_highs.max())
        flag_bot = float(flag_lows.min())
        flag_width = (flag_top - flag_bot) / close[pole_start]

        if flag_width > flag_max_width_pct:
            continue

        # Confirmation: bullish — close above flag_top, bearish — close below flag_bot
        confirmed_pos = None
        for j in range(i, min(i + confirm_window, n)):
            if direction == 1 and close[j] > flag_top:
                confirmed_pos = j
                break
            if direction == -1 and close[j] < flag_bot:
                confirmed_pos = j
                break

        if confirmed_pos is None:
            continue

        # Dedup
        if confirmed_pos - last_entry_pos < min_gap_bars:
            continue
        last_entry_pos = confirmed_pos

        entry_price = float(close[confirmed_pos])
        # Measured move: pole length projected from breakout
        pole_dollar_move = abs(close[pole_end] - close[pole_start])
        if direction == 1:
            target = entry_price + pole_dollar_move
            stop = flag_bot * 0.995
        else:
            target = entry_price - pole_dollar_move
            stop = flag_top * 1.005

        patterns.append(Pattern(
            type="bullish_flag" if direction == 1 else "bearish_flag",
            entry_date=df.index[confirmed_pos],
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop,
            take_profit=target,
            meta={
                "pole_start": df.index[pole_start],
                "pole_end": df.index[pole_end],
                "pole_move_pct": pole_move,
                "flag_top": flag_top,
                "flag_bot": flag_bot,
            },
        ))

    return patterns


def detect_bullish_flag(df, pivots, **kwargs):
    return _detect_flags(df, pivots, direction=1, **kwargs)


def detect_bearish_flag(df, pivots, **kwargs):
    return _detect_flags(df, pivots, direction=-1, **kwargs)


# ─── Pattern → Signals ──────────────────────────────────────────────

def patterns_to_signals(
    df: pd.DataFrame,
    patterns: list[Pattern],
    max_hold_bars: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Convert patterns into entry+exit signals, each with its own SL/TP.

    For each pattern, scans forward bar-by-bar until SL or TP hit, or max_hold_bars.
    Returns (long_entries, long_exits, short_entries, short_exits) bool Series.

    Note: vectorbt with from_signals + accumulate=False naturally skips overlapping
    entries while position open. Patterns whose entry falls inside another pattern's
    hold window will be ignored by vectorbt (acceptable for MVP).
    """
    long_entries = pd.Series(False, index=df.index)
    long_exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    for p in patterns:
        if p.entry_date not in df.index:
            continue
        entry_pos = df.index.get_loc(p.entry_date)

        exit_pos = min(entry_pos + max_hold_bars, len(df) - 1)
        for j in range(entry_pos + 1, min(entry_pos + max_hold_bars + 1, len(df))):
            bar = df.iloc[j]
            if p.direction == 1:  # long
                if bar["high"] >= p.take_profit:
                    exit_pos = j; break
                if bar["low"] <= p.stop_loss:
                    exit_pos = j; break
            else:  # short
                if bar["low"] <= p.take_profit:
                    exit_pos = j; break
                if bar["high"] >= p.stop_loss:
                    exit_pos = j; break

        if p.direction == 1:
            long_entries.iloc[entry_pos] = True
            long_exits.iloc[exit_pos] = True
        else:
            short_entries.iloc[entry_pos] = True
            short_exits.iloc[exit_pos] = True

    return long_entries, long_exits, short_entries, short_exits


# ─── Registry ───────────────────────────────────────────────────────

DETECTORS = {
    "double_top": detect_double_top,
    "double_bottom": detect_double_bottom,
    "head_and_shoulders": detect_head_shoulders,
    "inverse_head_and_shoulders": detect_inverted_head_shoulders,
    "ascending_triangle": detect_ascending_triangle,
    "descending_triangle": detect_descending_triangle,
    "symmetrical_triangle": detect_symmetrical_triangle,
    "bullish_flag": detect_bullish_flag,
    "bearish_flag": detect_bearish_flag,
}


def detect(df: pd.DataFrame, pattern_type: str, params: dict | None = None) -> list[Pattern]:
    """Run named detector on df with optional params."""
    if pattern_type not in DETECTORS:
        raise ValueError(f"Unknown pattern: {pattern_type}. Available: {list(DETECTORS)}")
    pivots = find_pivots(df, **(params or {}).get("pivot", {}))
    detector_params = {k: v for k, v in (params or {}).items() if k != "pivot"}
    return DETECTORS[pattern_type](df, pivots, **detector_params)


if __name__ == "__main__":
    import data
    for sym in ["SPY", "BTC"]:
        df = data.load(sym)
        pivots = find_pivots(df)
        n_highs = (pivots["type"] == "high").sum()
        n_lows = (pivots["type"] == "low").sum()
        print(f"\n=== {sym}: {len(df)} bars ===")
        print(f"Pivots: {n_highs} highs, {n_lows} lows")

        dt = detect_double_top(df, pivots)
        db = detect_double_bottom(df, pivots)
        print(f"Double tops found: {len(dt)}")
        print(f"Double bottoms found: {len(db)}")

        if dt:
            p = dt[0]
            print(f"  Sample top: entry {p.entry_date.date()} @ ${p.entry_price:.2f}, "
                  f"SL ${p.stop_loss:.2f}, TP ${p.take_profit:.2f}")
        if db:
            p = db[0]
            print(f"  Sample bottom: entry {p.entry_date.date()} @ ${p.entry_price:.2f}, "
                  f"SL ${p.stop_loss:.2f}, TP ${p.take_profit:.2f}")
