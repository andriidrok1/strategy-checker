"""Honest screening with In-Sample / Out-of-Sample split.

Methodology (real quant practice):
1. Split each symbol's data 50/50 → IS (in-sample, develop on this) + OOS (validate)
2. Screen all strategies on IS — find top performers
3. Test those SAME strategies on OOS data they never saw
4. A strategy is "real" only if it works on BOTH IS and OOS
5. Anything that works only on IS = curve-fit garbage

This is how professionals validate strategies. Most "backtested winners" fail OOS.
"""
import data
import backtest as bt
import metrics as m
from screen_alts import CANDIDATES, ALTS, INIT_CASH


def split_half(df):
    """Return (in_sample, out_of_sample) — 50/50 chronological split."""
    mid = len(df) // 2
    return df.iloc[:mid], df.iloc[mid:]


def run_strategy_on(df, rules):
    if len(df) < 50:
        return None
    try:
        trades, equity = bt.run(df, rules, init_cash=INIT_CASH)
        if trades.empty or len(equity) < 2:
            return None
        stats = m.compute(trades, equity)
        if "error" in stats:
            return None
        return stats
    except Exception:
        return None


def main():
    print("Honest screening with In-Sample / Out-of-Sample validation\n")
    print(f"Split: each coin's 1-year data → 6mo IS (develop) + 6mo OOS (validate)")
    print(f"Strategies: {len(CANDIDATES)} × Symbols: {len(ALTS)} = {len(CANDIDATES) * len(ALTS)} combos\n")

    is_results = []

    # === STEP 1: Screen on IN-SAMPLE only ===
    for strat, rules in CANDIDATES.items():
        for sym in ALTS:
            df = data.load(sym)
            df_is, _ = split_half(df)
            s = run_strategy_on(df_is, rules)
            if s is None:
                continue
            is_results.append({
                "strategy": strat,
                "symbol": sym,
                "is_trades": s["total_trades"],
                "is_return": s["total_return"],
                "is_sharpe": s["sharpe_ratio"],
                "is_dd": s["max_drawdown"],
                "rules": rules,
            })

    # IS criteria: meaningfully positive, reasonable sample
    is_winners = [
        r for r in is_results
        if r["is_trades"] >= 8
        and r["is_return"] > 0.10
        and r["is_sharpe"] > 0.5
    ]
    is_winners.sort(key=lambda r: r["is_return"], reverse=True)

    print(f"STEP 1: {len(is_winners)} of {len(is_results)} combos passed IS criteria\n")

    if not is_winners:
        print("No IS winners. Strategies don't even work on first half.")
        return

    # === STEP 2: Test ONLY those winners on OUT-OF-SAMPLE ===
    print("STEP 2: Testing IS winners on UNSEEN OOS data...\n")

    validated = []
    failed = []

    for w in is_winners:
        df = data.load(w["symbol"])
        _, df_oos = split_half(df)
        oos = run_strategy_on(df_oos, w["rules"])

        if oos is None:
            failed.append({**w, "oos_status": "no trades"})
            continue

        # Track OOS metrics regardless
        entry = {
            **w,
            "oos_trades": oos["total_trades"],
            "oos_return": oos["total_return"],
            "oos_sharpe": oos["sharpe_ratio"],
            "oos_dd": oos["max_drawdown"],
            "oos_final": oos["final_equity"],
        }

        # Validation: OOS must also be profitable with positive Sharpe
        if oos["total_trades"] >= 5 and oos["total_return"] > 0 and oos["sharpe_ratio"] > 0:
            validated.append(entry)
        else:
            failed.append({**entry, "oos_status": "failed"})

    print(f"{'='*120}")
    print(f"  RESULTS — Did winners survive OOS?")
    print(f"  IS winners: {len(is_winners)} | Survived OOS: {len(validated)} | Failed OOS: {len(failed)}")
    print(f"{'='*120}")

    if not validated:
        print("\n  ⚠ ZERO strategies survived out-of-sample validation.")
        print("  This means EVERY strategy that looked good on IS was curve-fit noise.")
        print("  Honest conclusion: на этом alt-crypto dataset нет robust edge.\n")
    else:
        validated.sort(key=lambda r: r["oos_return"], reverse=True)
        print(f"\n  TOP STRATEGIES THAT SURVIVED OOS ({min(5, len(validated))} of {len(is_winners)}):\n")
        print(f"{'Strategy':<38}{'Coin':<6}{'IS ret':>9}{'OOS ret':>10}{'IS Shrp':>9}{'OOS Shrp':>10}{'OOS Final':>12}")
        print('-' * 120)
        for v in validated[:5]:
            print(f"{v['strategy']:<38}{v['symbol']:<6}"
                  f"{v['is_return']:>+8.1%}{v['oos_return']:>+9.1%}"
                  f"{v['is_sharpe']:>+8.2f}{v['oos_sharpe']:>+9.2f}"
                  f"  ${v['oos_final']:>8,.0f}")

    # === STEP 3: Show what failed ===
    print(f"\n{'='*120}")
    print(f"  WHAT FAILED OOS (curve-fit reveal):")
    print(f"{'='*120}\n")
    print(f"{'Strategy':<38}{'Coin':<6}{'IS ret':>9}{'OOS ret':>10}{'IS Shrp':>9}{'OOS Shrp':>10}")
    print('-' * 100)
    for f in failed[:15]:
        oos_ret = f.get("oos_return", float("nan"))
        oos_shr = f.get("oos_sharpe", float("nan"))
        print(f"{f['strategy']:<38}{f['symbol']:<6}"
              f"{f['is_return']:>+8.1%}{oos_ret:>+9.1%}"
              f"{f['is_sharpe']:>+8.2f}{oos_shr:>+9.2f}")


if __name__ == "__main__":
    main()
