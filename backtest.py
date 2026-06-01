"""Backtest engine — computes indicators, evaluates rule expressions, runs vectorbt."""
import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt


# Indicators supported. Each maps to a function (df, spec) -> Series.
def _rsi(df, spec):
    return ta.rsi(df["close"], length=spec.get("period", 14))


def _sma(df, spec):
    return ta.sma(df["close"], length=spec.get("period", 20))


def _ema(df, spec):
    return ta.ema(df["close"], length=spec.get("period", 20))


def _atr(df, spec):
    return ta.atr(df["high"], df["low"], df["close"], length=spec.get("period", 14))


def _macd(df, spec):
    # returns DataFrame with MACD line, signal, hist — caller picks one
    fast = spec.get("fast", 12)
    slow = spec.get("slow", 26)
    signal = spec.get("signal", 9)
    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    component = spec.get("component", "line")  # line | signal | hist
    cols = list(macd_df.columns)
    if component == "signal":
        return macd_df[[c for c in cols if c.startswith("MACDs")][0]]
    if component == "hist":
        return macd_df[[c for c in cols if c.startswith("MACDh")][0]]
    return macd_df[[c for c in cols if c.startswith("MACD_")][0]]


def _stoch(df, spec):
    component = spec.get("component", "k")
    stoch_df = ta.stoch(df["high"], df["low"], df["close"],
                        k=spec.get("k", 14), d=spec.get("d", 3))
    cols = list(stoch_df.columns)
    target = "STOCHd" if component == "d" else "STOCHk"
    return stoch_df[[c for c in cols if c.startswith(target)][0]]


def _bbands(df, spec):
    component = spec.get("component", "middle")  # upper | middle | lower
    bb = ta.bbands(df["close"], length=spec.get("period", 20), std=spec.get("std", 2.0))
    cols = list(bb.columns)
    prefix = {"upper": "BBU", "middle": "BBM", "lower": "BBL"}[component]
    return bb[[c for c in cols if c.startswith(prefix)][0]]


def _adx(df, spec):
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=spec.get("period", 14))
    cols = list(adx_df.columns)
    return adx_df[[c for c in cols if c.startswith("ADX")][0]]


INDICATORS = {
    "rsi": _rsi,
    "sma": _sma,
    "ema": _ema,
    "atr": _atr,
    "macd": _macd,
    "stoch": _stoch,
    "bbands": _bbands,
    "adx": _adx,
}


def add_indicators(df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Compute and attach each indicator as a column named spec['name']."""
    df = df.copy()
    for spec in specs:
        kind = spec["type"].lower()
        if kind not in INDICATORS:
            raise ValueError(f"Unknown indicator type: {kind}. Supported: {list(INDICATORS)}")
        df[spec["name"]] = INDICATORS[kind](df, spec)
    return df


def _eval(df: pd.DataFrame, expr: str) -> pd.Series:
    """Evaluate expression like 'rsi_14 < 30 and close > sma_50' on df."""
    if not expr or not expr.strip():
        return pd.Series(False, index=df.index)
    # df.eval doesn't accept 'and'/'or' — convert to & / | with parens preserved
    expr_clean = expr.replace(" and ", " & ").replace(" or ", " | ")
    result = df.eval(expr_clean)
    if isinstance(result, pd.Series):
        return result.fillna(False).astype(bool)
    raise ValueError(f"Expression must yield a Series, got {type(result)}")


def _run_indicators_mode(df: pd.DataFrame, rules: dict, init_cash: float,
                          fees: float = 0.001, slippage: float = 0.0005):
    df_ind = add_indicators(df, rules.get("indicators", []))

    long_entries = _eval(df_ind, rules.get("entry_long", ""))
    long_exits = _eval(df_ind, rules.get("exit_long", ""))
    short_entries = _eval(df_ind, rules.get("entry_short", "") or "")
    short_exits = _eval(df_ind, rules.get("exit_short", "") or "")

    pf_kwargs = dict(
        close=df_ind["close"],
        entries=long_entries,
        exits=long_exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
    )
    if short_entries.any():
        pf_kwargs["short_entries"] = short_entries
        pf_kwargs["short_exits"] = short_exits
    if rules.get("stop_loss_pct") is not None:
        pf_kwargs["sl_stop"] = rules["stop_loss_pct"]
    if rules.get("take_profit_pct") is not None:
        pf_kwargs["tp_stop"] = rules["take_profit_pct"]

    return vbt.Portfolio.from_signals(**pf_kwargs)


def _run_pattern_mode(df: pd.DataFrame, rules: dict, init_cash: float):
    import patterns as pat

    spec = rules["pattern"]
    detected = pat.detect(df, spec["type"], spec.get("params"))
    long_e, long_x, short_e, short_x = pat.patterns_to_signals(
        df, detected, max_hold_bars=spec.get("max_hold_bars", 100)
    )

    pf_kwargs = dict(
        close=df["close"],
        entries=long_e,
        exits=long_x,
        init_cash=init_cash,
        fees=0.001,
        slippage=0.0005,
    )
    if short_e.any():
        pf_kwargs["short_entries"] = short_e
        pf_kwargs["short_exits"] = short_x

    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    pf._detected_patterns = detected  # attach for debugging/UI
    return pf


def run(df: pd.DataFrame, rules: dict, init_cash: float = 10_000,
        fees: float = 0.001, slippage: float = 0.0005) -> tuple[pd.DataFrame, pd.Series]:
    """Execute strategy rules against price data. Returns (trades, equity).

    fees: per-side as fraction (0.001 = 0.1%, default = retail spot crypto)
    slippage: per-side as fraction (0.0005 = 0.05%, conservative for liquid pairs)

    Dispatches based on rules:
    - rules["pattern"] present → chart pattern detection mode
    - else → indicator-based rules mode
    """
    if rules.get("pattern"):
        pf = _run_pattern_mode(df, rules, init_cash)
    else:
        pf = _run_indicators_mode(df, rules, init_cash, fees=fees, slippage=slippage)

    trades = pf.trades.records_readable
    equity = pf.value()
    return trades, equity


if __name__ == "__main__":
    import data
    print("Loading SPY...")
    df = data.load("SPY")

    rules = {
        "indicators": [
            {"name": "rsi_14", "type": "rsi", "period": 14},
            {"name": "sma_50", "type": "sma", "period": 50},
        ],
        "entry_long": "rsi_14 < 30 and close > sma_50",
        "exit_long": "rsi_14 > 70",
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
    }
    print("Running backtest: 'RSI<30 entry, RSI>70 exit, above SMA50 filter'")
    trades, equity = run(df, rules)
    print(f"  Trades: {len(trades)}")
    print(f"  Final equity: ${equity.iloc[-1]:.2f}")

    import metrics
    stats = metrics.compute(trades, equity)
    print("\n--- Metrics ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    verdict = metrics.verdict(stats)
    print(f"\nVerdict: {verdict['label']} ({verdict['reason']})")
