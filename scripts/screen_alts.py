"""Screening tailored for alt-crypto subset: ADA, ARB, ATOM, AVAX, DOGE, DOT, FIL, LINK, NEAR, UNI, XRP.

1-year daily data → adjusted thresholds:
- min trades: 15 (instead of 30 — shorter window)
- return: > 10%
- sharpe: > 0.5
- max DD: > -45% (alts are volatile, accept wider)

Adds crypto-tuned strategies (faster EMAs, RSI extremes, breakouts).
"""
import data
import backtest as bt
import metrics as m

INIT_CASH = 10_000
ALTS = ["ADA", "ARB", "ATOM", "AVAX", "DOGE", "DOT", "FIL", "LINK", "NEAR", "UNI", "XRP"]

# Crypto-focused strategy set
CANDIDATES = {
    # --- Short EMA crosses (crypto trends move fast) ---
    "Fast EMA cross (3/8)": {
        "indicators": [
            {"name": "ema_3", "type": "ema", "period": 3},
            {"name": "ema_8", "type": "ema", "period": 8},
        ],
        "entry_long": "ema_3 > ema_8", "exit_long": "ema_3 < ema_8",
    },
    "EMA cross (5/13)": {
        "indicators": [
            {"name": "ema_5", "type": "ema", "period": 5},
            {"name": "ema_13", "type": "ema", "period": 13},
        ],
        "entry_long": "ema_5 > ema_13", "exit_long": "ema_5 < ema_13",
    },
    "EMA cross (8/21)": {
        "indicators": [
            {"name": "ema_8", "type": "ema", "period": 8},
            {"name": "ema_21", "type": "ema", "period": 21},
        ],
        "entry_long": "ema_8 > ema_21", "exit_long": "ema_8 < ema_21",
    },
    "EMA cross (10/30)": {
        "indicators": [
            {"name": "ema_10", "type": "ema", "period": 10},
            {"name": "ema_30", "type": "ema", "period": 30},
        ],
        "entry_long": "ema_10 > ema_30", "exit_long": "ema_10 < ema_30",
    },
    "EMA cross (12/26)": {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "ema_12 > ema_26", "exit_long": "ema_12 < ema_26",
    },
    "EMA cross (20/50)": {
        "indicators": [
            {"name": "ema_20", "type": "ema", "period": 20},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "ema_20 > ema_50", "exit_long": "ema_20 < ema_50",
    },

    # --- Above-EMA momentum (trend regime) ---
    "Above 21 EMA": {
        "indicators": [{"name": "ema_21", "type": "ema", "period": 21}],
        "entry_long": "close > ema_21", "exit_long": "close < ema_21",
    },
    "Above 50 EMA": {
        "indicators": [{"name": "ema_50", "type": "ema", "period": 50}],
        "entry_long": "close > ema_50", "exit_long": "close < ema_50",
    },
    "Above 100 EMA": {
        "indicators": [{"name": "ema_100", "type": "ema", "period": 100}],
        "entry_long": "close > ema_100", "exit_long": "close < ema_100",
    },

    # --- RSI extremes (high volatility = more frequent extremes) ---
    "RSI extreme (25/75)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 25", "exit_long": "rsi_14 > 75",
        "stop_loss_pct": 0.10, "take_profit_pct": 0.20,
    },
    "RSI mean-rev (30/70)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 30", "exit_long": "rsi_14 > 70",
        "stop_loss_pct": 0.08, "take_profit_pct": 0.15,
    },
    "RSI dip + trend (above EMA50)": {
        "indicators": [
            {"name": "rsi_14", "type": "rsi", "period": 14},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "close > ema_50 and rsi_14 < 40",
        "exit_long": "rsi_14 > 65",
        "stop_loss_pct": 0.06, "take_profit_pct": 0.12,
    },

    # --- MACD ---
    "MACD signal cross": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_signal", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
        ],
        "entry_long": "macd_line > macd_signal", "exit_long": "macd_line < macd_signal",
    },
    "MACD + above EMA50": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_signal", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "macd_line > macd_signal and close > ema_50",
        "exit_long": "macd_line < macd_signal",
    },

    # --- Bollinger ---
    "Bollinger mean-rev (2 std)": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.0, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.0, "component": "middle"},
        ],
        "entry_long": "close < bb_lower", "exit_long": "close > bb_middle",
        "stop_loss_pct": 0.08,
    },
    "Bollinger touch (2.5 std)": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.5, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.5, "component": "middle"},
        ],
        "entry_long": "close < bb_lower", "exit_long": "close > bb_middle",
    },

    # --- ADX trend filter ---
    "Strong trend (ADX>25) + EMA cross": {
        "indicators": [
            {"name": "ema_9", "type": "ema", "period": 9},
            {"name": "ema_21", "type": "ema", "period": 21},
            {"name": "adx_14", "type": "adx", "period": 14},
        ],
        "entry_long": "ema_9 > ema_21 and adx_14 > 25",
        "exit_long": "ema_9 < ema_21",
    },

    # --- Stochastic ---
    "Stochastic oversold (<20)": {
        "indicators": [{"name": "stoch_k", "type": "stoch", "k": 14, "d": 3, "component": "k"}],
        "entry_long": "stoch_k < 20", "exit_long": "stoch_k > 80",
        "stop_loss_pct": 0.08,
    },

    # --- Chart patterns ---
    "Pattern: Double Bottom": {"pattern": {"type": "double_bottom"}},
    "Pattern: Inverse H&S": {"pattern": {"type": "inverse_head_and_shoulders"}},
    "Pattern: Ascending Triangle": {"pattern": {"type": "ascending_triangle"}},
    "Pattern: Symmetrical Triangle": {"pattern": {"type": "symmetrical_triangle"}},
    "Pattern: Bullish Flag": {"pattern": {"type": "bullish_flag"}},

    # === BEAR-MARKET / SHORT-SIDE STRATEGIES ===

    "SHORT: EMA cross (5/13)": {
        "indicators": [
            {"name": "ema_5", "type": "ema", "period": 5},
            {"name": "ema_13", "type": "ema", "period": 13},
        ],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "ema_5 < ema_13",
        "exit_short": "ema_5 > ema_13",
    },
    "SHORT: EMA cross (8/21)": {
        "indicators": [
            {"name": "ema_8", "type": "ema", "period": 8},
            {"name": "ema_21", "type": "ema", "period": 21},
        ],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "ema_8 < ema_21",
        "exit_short": "ema_8 > ema_21",
    },
    "SHORT: EMA cross (12/26)": {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "ema_12 < ema_26",
        "exit_short": "ema_12 > ema_26",
    },
    "SHORT: below 50 EMA": {
        "indicators": [{"name": "ema_50", "type": "ema", "period": 50}],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "close < ema_50",
        "exit_short": "close > ema_50",
    },
    "SHORT: below 100 EMA": {
        "indicators": [{"name": "ema_100", "type": "ema", "period": 100}],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "close < ema_100",
        "exit_short": "close > ema_100",
    },
    "SHORT: RSI overbought rallies (>60)": {
        "indicators": [
            {"name": "rsi_14", "type": "rsi", "period": 14},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "rsi_14 > 60 and close < ema_50",
        "exit_short": "rsi_14 < 40",
        "stop_loss_pct": 0.08, "take_profit_pct": 0.15,
    },
    "SHORT: MACD bearish + below EMA50": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_signal", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "",
        "exit_long": "",
        "entry_short": "macd_line < macd_signal and close < ema_50",
        "exit_short": "macd_line > macd_signal",
    },
    "SHORT: Descending triangle pattern": {"pattern": {"type": "descending_triangle"}},
    "SHORT: Head & Shoulders pattern": {"pattern": {"type": "head_and_shoulders"}},
    "SHORT: Double Top pattern": {"pattern": {"type": "double_top"}},
    "SHORT: Bearish Flag pattern": {"pattern": {"type": "bearish_flag"}},

    # --- LONG/SHORT both sides ---
    "BOTH SIDES: EMA cross (8/21)": {
        "indicators": [
            {"name": "ema_8", "type": "ema", "period": 8},
            {"name": "ema_21", "type": "ema", "period": 21},
        ],
        "entry_long": "ema_8 > ema_21",
        "exit_long": "ema_8 < ema_21",
        "entry_short": "ema_8 < ema_21",
        "exit_short": "ema_8 > ema_21",
    },
    "BOTH SIDES: EMA cross (12/26)": {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "ema_12 > ema_26",
        "exit_long": "ema_12 < ema_26",
        "entry_short": "ema_12 < ema_26",
        "exit_short": "ema_12 > ema_26",
    },
}


def main():
    print(f"Alt-crypto screening: {len(CANDIDATES)} strategies × {len(ALTS)} symbols "
          f"= {len(CANDIDATES) * len(ALTS)} backtests")
    print(f"Symbols: {', '.join(ALTS)}\n")

    results = []

    for strat_name, rules in CANDIDATES.items():
        for sym in ALTS:
            try:
                df = data.load(sym)
                trades, equity = bt.run(df, rules, init_cash=INIT_CASH)
                if trades.empty:
                    continue
                stats = m.compute(trades, equity)
                if "error" in stats:
                    continue

                results.append({
                    "strategy": strat_name,
                    "symbol": sym,
                    "trades": stats["total_trades"],
                    "win_rate": stats["win_rate"],
                    "total_return": stats["total_return"],
                    "sharpe": stats["sharpe_ratio"],
                    "max_dd": stats["max_drawdown"],
                    "profit_factor": stats["profit_factor"],
                    "final_equity": stats["final_equity"],
                    "profit": stats["final_equity"] - INIT_CASH,
                })
            except Exception as e:
                print(f"  ⚠ {strat_name} on {sym}: {e}")

    # Adjusted thresholds for 1-year data
    qualifying = [
        r for r in results
        if r["trades"] >= 15
        and r["total_return"] > 0.10
        and r["sharpe"] > 0.5
        and r["max_dd"] > -0.45
    ]

    def score(r):
        # Prioritize absolute profit on $10K
        return r["profit"] * r["sharpe"] / (1 + abs(r["max_dd"]))

    qualifying.sort(key=score, reverse=True)

    print(f"\n{'=' * 110}")
    print(f"  RESULTS: {len(qualifying)} of {len(results)} backtests PASSED")
    print(f"  Criteria: ≥15 trades, return > 10%, Sharpe > 0.5, max DD > -45%")
    print(f"  Capital: ${INIT_CASH:,} | Period: 1 year (2025-2026)")
    print('=' * 110)

    if not qualifying:
        print("\n  ⚠ No strategies passed.")
        return

    print(f"\n{'TOP 5 PROFITABLE STRATEGIES (alt-crypto)':^110}\n")
    print(f"{'#':<3}{'Strategy':<38}{'Coin':<7}{'Trades':>7}{'WR':>5}{'Return':>9}"
          f"{'Sharpe':>8}{'Max DD':>8}{'Final $':>10}{'Profit':>10}")
    print('-' * 110)
    for i, r in enumerate(qualifying[:5], 1):
        print(f"{i:<3}{r['strategy']:<38}{r['symbol']:<7}{r['trades']:>7}"
              f"{r['win_rate']:>4.0%}{r['total_return']:>+8.1%}"
              f"{r['sharpe']:>+7.2f}{r['max_dd']:>7.1%}"
              f"  ${r['final_equity']:>7,.0f}  ${r['profit']:>+7,.0f}")

    print(f"\n--- All {len(qualifying)} passing combos ---")
    for r in qualifying:
        print(f"  {r['strategy']:<38}{r['symbol']:<5} "
              f"trades={r['trades']:>3} ret={r['total_return']:>+7.1%} "
              f"sharpe={r['sharpe']:>+5.2f} dd={r['max_dd']:>6.1%} "
              f"profit=${r['profit']:>+7,.0f}")


if __name__ == "__main__":
    main()
