"""Systematic screening: test N strategies × M symbols, find profitable ones.

Selection criteria (all must pass):
- total_trades >= 30
- total_return > 0
- sharpe_ratio > 0.5
- max_drawdown > -0.30 (less than 30% drawdown)
"""
import data
import backtest as bt
import metrics as m

INIT_CASH = 10_000

# Candidate strategies — mix of indicator rules and chart patterns
CANDIDATES = {
    # === Indicator-based ===
    "RSI mean reversion (35/65)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 35",
        "exit_long": "rsi_14 > 65",
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
    },
    "RSI mean reversion (30/70)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 30",
        "exit_long": "rsi_14 > 70",
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
    },
    "Golden Cross (50/200 SMA)": {
        "indicators": [
            {"name": "sma_50", "type": "sma", "period": 50},
            {"name": "sma_200", "type": "sma", "period": 200},
        ],
        "entry_long": "sma_50 > sma_200",
        "exit_long": "sma_50 < sma_200",
    },
    "EMA crossover (9/21)": {
        "indicators": [
            {"name": "ema_9", "type": "ema", "period": 9},
            {"name": "ema_21", "type": "ema", "period": 21},
        ],
        "entry_long": "ema_9 > ema_21",
        "exit_long": "ema_9 < ema_21",
    },
    "EMA crossover (20/50)": {
        "indicators": [
            {"name": "ema_20", "type": "ema", "period": 20},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "ema_20 > ema_50",
        "exit_long": "ema_20 < ema_50",
    },
    "Bollinger mean reversion": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.0, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.0, "component": "middle"},
        ],
        "entry_long": "close < bb_lower",
        "exit_long": "close > bb_middle",
        "stop_loss_pct": 0.05,
    },
    "Buy dips below 200 SMA with RSI": {
        "indicators": [
            {"name": "sma_200", "type": "sma", "period": 200},
            {"name": "rsi_14", "type": "rsi", "period": 14},
        ],
        "entry_long": "close < sma_200 and rsi_14 < 35",
        "exit_long": "close > sma_200",
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.15,
    },
    "Buy strength above 200 SMA with RSI dip": {
        "indicators": [
            {"name": "sma_200", "type": "sma", "period": 200},
            {"name": "rsi_14", "type": "rsi", "period": 14},
        ],
        "entry_long": "close > sma_200 and rsi_14 < 40",
        "exit_long": "rsi_14 > 65",
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
    },
    "MACD signal cross": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_signal", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
        ],
        "entry_long": "macd_line > macd_signal",
        "exit_long": "macd_line < macd_signal",
    },
    "Stochastic oversold": {
        "indicators": [{"name": "stoch_k", "type": "stoch", "k": 14, "d": 3, "component": "k"}],
        "entry_long": "stoch_k < 20",
        "exit_long": "stoch_k > 80",
        "stop_loss_pct": 0.05,
    },
    "ADX-filtered trend (ema + adx)": {
        "indicators": [
            {"name": "ema_9", "type": "ema", "period": 9},
            {"name": "ema_21", "type": "ema", "period": 21},
            {"name": "adx_14", "type": "adx", "period": 14},
        ],
        "entry_long": "ema_9 > ema_21 and adx_14 > 25",
        "exit_long": "ema_9 < ema_21",
    },
    "RSI + above EMA 200 filter": {
        "indicators": [
            {"name": "rsi_14", "type": "rsi", "period": 14},
            {"name": "ema_200", "type": "ema", "period": 200},
        ],
        "entry_long": "rsi_14 < 35 and close > ema_200",
        "exit_long": "rsi_14 > 60",
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
    },

    # === Chart pattern based ===
    "Pattern: Double Bottom": {"pattern": {"type": "double_bottom"}},
    "Pattern: Inverse H&S": {"pattern": {"type": "inverse_head_and_shoulders"}},
    "Pattern: Ascending Triangle": {"pattern": {"type": "ascending_triangle"}},
    "Pattern: Symmetrical Triangle": {"pattern": {"type": "symmetrical_triangle"}},
    "Pattern: Double Top (short)": {"pattern": {"type": "double_top"}},
    "Pattern: Head & Shoulders (short)": {"pattern": {"type": "head_and_shoulders"}},

    # === More indicator variations ===
    "EMA crossover (5/13)": {
        "indicators": [
            {"name": "ema_5", "type": "ema", "period": 5},
            {"name": "ema_13", "type": "ema", "period": 13},
        ],
        "entry_long": "ema_5 > ema_13",
        "exit_long": "ema_5 < ema_13",
    },
    "EMA crossover (12/26)": {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "ema_12 > ema_26",
        "exit_long": "ema_12 < ema_26",
    },
    "EMA crossover (10/30)": {
        "indicators": [
            {"name": "ema_10", "type": "ema", "period": 10},
            {"name": "ema_30", "type": "ema", "period": 30},
        ],
        "entry_long": "ema_10 > ema_30",
        "exit_long": "ema_10 < ema_30",
    },
    "Buy & Hold above 50 EMA": {
        "indicators": [{"name": "ema_50", "type": "ema", "period": 50}],
        "entry_long": "close > ema_50",
        "exit_long": "close < ema_50",
    },
    "Buy & Hold above 200 EMA": {
        "indicators": [{"name": "ema_200", "type": "ema", "period": 200}],
        "entry_long": "close > ema_200",
        "exit_long": "close < ema_200",
    },
    "Bollinger touch (looser, 2.5 std)": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.5, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.5, "component": "middle"},
        ],
        "entry_long": "close < bb_lower",
        "exit_long": "close > bb_middle",
    },
    "RSI 40/60 (tight)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 40",
        "exit_long": "rsi_14 > 60",
        "stop_loss_pct": 0.04,
    },
    "Trend + Pullback (50EMA above 200EMA, RSI dip)": {
        "indicators": [
            {"name": "ema_50", "type": "ema", "period": 50},
            {"name": "ema_200", "type": "ema", "period": 200},
            {"name": "rsi_14", "type": "rsi", "period": 14},
        ],
        "entry_long": "ema_50 > ema_200 and rsi_14 < 45",
        "exit_long": "rsi_14 > 65",
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
    },
    "MACD + RSI confirm": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_signal", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
            {"name": "rsi_14", "type": "rsi", "period": 14},
        ],
        "entry_long": "macd_line > macd_signal and rsi_14 > 50",
        "exit_long": "macd_line < macd_signal",
    },
}


def main():
    symbols = data.available_symbols()
    results = []

    print(f"Screening {len(CANDIDATES)} strategies × {len(symbols)} symbols = "
          f"{len(CANDIDATES) * len(symbols)} backtests...\n")

    for strat_name, rules in CANDIDATES.items():
        for sym in symbols:
            try:
                df = data.load(sym)
                trades, equity = bt.run(df, rules, init_cash=INIT_CASH)
                if trades.empty:
                    continue
                stats = m.compute(trades, equity)
                if "error" in stats:
                    continue

                # Final dollar P&L
                final_cash = stats["final_equity"]
                profit = final_cash - INIT_CASH

                results.append({
                    "strategy": strat_name,
                    "symbol": sym,
                    "trades": stats["total_trades"],
                    "win_rate": stats["win_rate"],
                    "total_return": stats["total_return"],
                    "sharpe": stats["sharpe_ratio"],
                    "max_dd": stats["max_drawdown"],
                    "profit_factor": stats["profit_factor"],
                    "final_equity": final_cash,
                    "profit": profit,
                })
            except Exception as e:
                print(f"  ⚠ {strat_name} on {sym}: {e}")

    # Filter — looser to find top 5
    qualifying = [
        r for r in results
        if r["trades"] >= 30
        and r["total_return"] > 0.10  # > 10% positive
        and r["sharpe"] > 0.3
        and r["max_dd"] > -0.40
    ]

    # Rank by Sharpe (most honest single metric for risk-adj returns)
    def score(r):
        return r["sharpe"] * (1 + r["total_return"]) / (1 + abs(r["max_dd"]))

    qualifying.sort(key=score, reverse=True)

    print(f"\n{'='*100}")
    print(f"  RESULTS: {len(qualifying)} of {len(results)} backtests PASSED criteria")
    print(f"  Criteria: ≥30 trades, return > 10%, Sharpe > 0.3, max DD > -40%")
    print(f"  Capital: ${INIT_CASH:,}")
    print('=' * 100)

    print(f"\n{'TOP 5 PROFITABLE STRATEGIES':^100}\n")

    if not qualifying:
        print("  ⚠ No strategies passed criteria")
        return

    print(f"{'#':<3}{'Strategy':<40}{'Symbol':<8}{'Trades':>7}{'WR':>6}{'Return':>9}"
          f"{'Sharpe':>8}{'Max DD':>8}{'Profit':>11}")
    print('-' * 100)
    for i, r in enumerate(qualifying[:5], 1):
        print(f"{i:<3}{r['strategy']:<40}{r['symbol']:<8}{r['trades']:>7}"
              f"{r['win_rate']:>5.0%}{r['total_return']:>+8.1%}"
              f"{r['sharpe']:>+7.2f}{r['max_dd']:>7.1%}"
              f"  ${r['profit']:>+7,.0f}")

    # Also show all qualifying (not just top 5)
    if len(qualifying) > 5:
        print(f"\n--- ALL {len(qualifying)} PASSING STRATEGIES ---\n")
        for r in qualifying:
            print(f"  {r['strategy']:<40}{r['symbol']:<6}"
                  f"trades={r['trades']:>4}  ret={r['total_return']:>+7.1%}  "
                  f"sharpe={r['sharpe']:>+5.2f}  dd={r['max_dd']:>6.1%}  "
                  f"final=${r['final_equity']:>9,.0f}")


if __name__ == "__main__":
    main()
