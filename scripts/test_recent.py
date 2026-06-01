"""Test validated strategies on TRULY unseen recent data (2026-04-16 → 2026-05-28).

This is the strictest test possible — the strategy was designed without seeing this data at all.
"""
from pathlib import Path
import pandas as pd
import backtest as bt
import metrics as m

DATA_DIR = Path(__file__).parent / "data"
INIT_CASH = 10_000


def load_recent(symbol):
    return pd.read_parquet(DATA_DIR / f"{symbol}_recent.parquet")


# The 5 validated winners from walk-forward
STRATEGIES_TO_TEST = [
    ("SHORT: EMA cross (5/13)", "DOGE", "🟢 3/3 folds (Tier 1)", {
        "indicators": [
            {"name": "ema_5", "type": "ema", "period": 5},
            {"name": "ema_13", "type": "ema", "period": 13},
        ],
        "entry_long": "", "exit_long": "",
        "entry_short": "ema_5 < ema_13",
        "exit_short": "ema_5 > ema_13",
    }),
    ("BOTH SIDES: EMA cross (12/26)", "XRP", "🟢 3/3 folds (Tier 1)", {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "ema_12 > ema_26",
        "exit_long": "ema_12 < ema_26",
        "entry_short": "ema_12 < ema_26",
        "exit_short": "ema_12 > ema_26",
    }),
    ("Bollinger touch (2.5 std)", "FIL", "🟡 2/3 folds (Tier 2)", {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.5, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.5, "component": "middle"},
        ],
        "entry_long": "close < bb_lower",
        "exit_long": "close > bb_middle",
    }),
    ("SHORT: EMA cross (5/13)", "ADA", "🟡 2/3 folds (Tier 2)", {
        "indicators": [
            {"name": "ema_5", "type": "ema", "period": 5},
            {"name": "ema_13", "type": "ema", "period": 13},
        ],
        "entry_long": "", "exit_long": "",
        "entry_short": "ema_5 < ema_13",
        "exit_short": "ema_5 > ema_13",
    }),
    ("BOTH SIDES: EMA cross (12/26)", "ADA", "🟡 2/3 folds (Tier 2)", {
        "indicators": [
            {"name": "ema_12", "type": "ema", "period": 12},
            {"name": "ema_26", "type": "ema", "period": 26},
        ],
        "entry_long": "ema_12 > ema_26",
        "exit_long": "ema_12 < ema_26",
        "entry_short": "ema_12 < ema_26",
        "exit_short": "ema_12 > ema_26",
    }),
]


def main():
    # Buy-and-hold context
    print("=" * 100)
    print("  CONTEXT — Buy-and-Hold returns на свежих 43 днях (2026-04-16 → 2026-05-28):")
    print("=" * 100)
    for sym in ["DOGE", "XRP", "FIL", "ADA"]:
        df = load_recent(sym)
        ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        marker = "📈" if ret > 0 else "📉"
        print(f"  {marker} {sym}: ${df['close'].iloc[0]:.4f} → ${df['close'].iloc[-1]:.4f} = {ret:+.1%}")

    print()
    print("=" * 100)
    print("  STRATEGY TEST ON UNSEEN DATA:")
    print("=" * 100)
    print(f"\n{'Strategy':<32}{'Coin':<6}{'Tier':<22}{'Trades':>7}{'Return':>9}{'Final $':>11}{'Status':>11}")
    print('-' * 100)

    total_profit = 0
    survived = 0
    failed = 0

    for strat_name, sym, tier, rules in STRATEGIES_TO_TEST:
        df = load_recent(sym)
        trades, equity = bt.run(df, rules, init_cash=INIT_CASH)
        if trades.empty:
            print(f"{strat_name:<32}{sym:<6}{tier:<22}{'0':>7}{'-':>9}{'$10,000':>11}{'NO TRADES':>11}")
            continue

        s = m.compute(trades, equity)
        if "error" in s:
            print(f"{strat_name:<32}{sym:<6}{tier:<22}{'-':>7}{'ERR':>9}")
            continue

        ret = s["total_return"]
        n = s["total_trades"]
        final = s["final_equity"]

        if ret > 0:
            status = "✓ WORKS"
            survived += 1
        else:
            status = "✗ FAILS"
            failed += 1

        total_profit += (final - INIT_CASH)

        print(f"{strat_name:<32}{sym:<6}{tier:<22}{n:>7}{ret:>+8.1%}  ${final:>7,.0f}  {status:>10}")

    print('-' * 100)
    print(f"\nSurvived OOS test: {survived} of {len(STRATEGIES_TO_TEST)} | Failed: {failed}")
    print(f"Total profit if $2K в каждую = ${total_profit / len(STRATEGIES_TO_TEST) * 5 / 5 * 2 / INIT_CASH * 10000:,.0f} on $10K total")
    print(f"\nNote: 43 days is SHORT — few trades fire. Most short-EMA strategies need 20+ bars to even start signaling.")


if __name__ == "__main__":
    main()
