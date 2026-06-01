"""3-fold walk-forward validation — much stricter than single IS/OOS split.

A strategy is "robust" only if it's profitable in MULTIPLE non-overlapping periods.
Anything that wins in 1 of 3 folds = noise. Wins in 2/3 = maybe. Wins in 3/3 = real signal.
"""
import data
import backtest as bt
import metrics as m
from screen_alts import CANDIDATES, ALTS, INIT_CASH


def split_n(df, n_folds=3):
    """Return list of n equal chronological folds."""
    fold_size = len(df) // n_folds
    return [df.iloc[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]


def run(df, rules):
    if len(df) < 50:
        return None
    try:
        trades, equity = bt.run(df, rules, init_cash=INIT_CASH)
        if trades.empty or len(equity) < 2:
            return None
        s = m.compute(trades, equity)
        return None if "error" in s else s
    except Exception:
        return None


def main():
    print("3-fold walk-forward validation на alt-crypto\n")
    print(f"Каждая coin's 1 year → 3 × 4-month folds (independent periods)\n")

    results = []

    for strat, rules in CANDIDATES.items():
        for sym in ALTS:
            df = data.load(sym)
            folds = split_n(df, 3)

            fold_stats = []
            for fold_df in folds:
                s = run(fold_df, rules)
                fold_stats.append(s)

            # Count profitable folds
            profitable = sum(
                1 for s in fold_stats
                if s and s["total_trades"] >= 3 and s["total_return"] > 0
            )

            total_return_compound = 1.0
            for s in fold_stats:
                if s and s["total_trades"] >= 3:
                    total_return_compound *= (1 + s["total_return"])

            results.append({
                "strategy": strat,
                "symbol": sym,
                "profitable_folds": profitable,
                "compound_return": total_return_compound - 1,
                "fold_returns": [
                    s["total_return"] if s else None for s in fold_stats
                ],
                "fold_trades": [
                    s["total_trades"] if s else 0 for s in fold_stats
                ],
            })

    # Robust: profitable in ALL 3 folds
    robust = [r for r in results if r["profitable_folds"] == 3]
    robust.sort(key=lambda r: r["compound_return"], reverse=True)

    # Semi-robust: 2 of 3
    semi = [r for r in results if r["profitable_folds"] == 2]
    semi.sort(key=lambda r: r["compound_return"], reverse=True)

    print(f"=" * 100)
    print(f"  TIER 1 (3/3 folds profitable — REAL SIGNAL): {len(robust)}")
    print(f"  TIER 2 (2/3 folds profitable — maybe): {len(semi)}")
    print(f"  TIER 3 (0-1/3 folds — noise/curve-fit): {len(results) - len(robust) - len(semi)}")
    print(f"=" * 100)

    if robust:
        print(f"\n  TIER 1 — 3/3 PROFITABLE FOLDS (max {min(5, len(robust))}):\n")
        print(f"{'Strategy':<38}{'Coin':<6}{'Fold 1':>10}{'Fold 2':>10}{'Fold 3':>10}{'Compound':>11}{'Final $':>11}")
        print('-' * 100)
        for r in robust[:5]:
            fr = r["fold_returns"]
            final = INIT_CASH * (1 + r["compound_return"])
            print(f"{r['strategy']:<38}{r['symbol']:<6}"
                  f"{fr[0]:>+9.1%}{fr[1]:>+9.1%}{fr[2]:>+9.1%}"
                  f"{r['compound_return']:>+10.1%}  ${final:>8,.0f}")
    else:
        print("\n  ⚠ ZERO strategies profitable in ALL 3 folds.")

    if semi:
        print(f"\n  TIER 2 — 2/3 FOLDS (borderline, max 10):\n")
        print(f"{'Strategy':<38}{'Coin':<6}{'Fold 1':>10}{'Fold 2':>10}{'Fold 3':>10}{'Compound':>11}")
        print('-' * 100)
        for r in semi[:10]:
            fr = r["fold_returns"]
            print(f"{r['strategy']:<38}{r['symbol']:<6}"
                  f"{fr[0]:>+9.1%}{fr[1]:>+9.1%}{fr[2]:>+9.1%}"
                  f"{r['compound_return']:>+10.1%}")


if __name__ == "__main__":
    main()
