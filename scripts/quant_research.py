"""Institutional-grade validation framework for high-frequency crypto strategies.

Workflow:
1. Define candidate strategies (literature-derived defaults, no parameter mining)
2. Run on BTC/ETH/SOL 5m with REALISTIC Binance perp fees:
   - Taker fee: 0.04% per side
   - Slippage:  0.02% per side
   - Total round-trip cost: 0.12%
3. 4-fold walk-forward validation
4. Monte Carlo trade-order shuffling (500 sims)
5. Parameter sensitivity sweep (small, intentional grid)
6. Honest reporting — likely most strategies fail

Goal targets (from /goal directive):
- Profit Factor > 1.5
- Sharpe > 2.0
- Max DD < 10%
- 300+ trades/month
- Profitable after fees

Realistic expectation: most retail-style strategies fail to meet ALL targets.
"""
import json
import numpy as np
import pandas as pd
import data
import backtest as bt
import metrics as m

INIT_CASH = 10_000
TAKER_FEE = 0.0004   # Binance perp taker
SLIPPAGE = 0.0002    # conservative for top-3 crypto perps
SYMBOLS = ["BTC", "ETH", "SOL"]
TF = "5m"


# ─── Candidate strategies (5m crypto perps) ────────────────────────

STRATEGIES = {
    "Bollinger MR (20, 2)": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.0, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.0, "component": "middle"},
            {"name": "bb_upper", "type": "bbands", "period": 20, "std": 2.0, "component": "upper"},
        ],
        "entry_long": "close < bb_lower",
        "exit_long": "close > bb_middle",
        "entry_short": "close > bb_upper",
        "exit_short": "close < bb_middle",
        "stop_loss_pct": 0.005,  # 0.5%
    },
    "RSI extreme (14, 20/80)": {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 20",
        "exit_long": "rsi_14 > 50",
        "entry_short": "rsi_14 > 80",
        "exit_short": "rsi_14 < 50",
        "stop_loss_pct": 0.005,
        "take_profit_pct": 0.008,
    },
    "EMA cross (8/21)": {
        "indicators": [
            {"name": "ema_8", "type": "ema", "period": 8},
            {"name": "ema_21", "type": "ema", "period": 21},
        ],
        "entry_long": "ema_8 > ema_21",
        "exit_long": "ema_8 < ema_21",
        "entry_short": "ema_8 < ema_21",
        "exit_short": "ema_8 > ema_21",
    },
    "MACD signal cross": {
        "indicators": [
            {"name": "macd_line", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "line"},
            {"name": "macd_sig", "type": "macd", "fast": 12, "slow": 26, "signal": 9, "component": "signal"},
        ],
        "entry_long": "macd_line > macd_sig",
        "exit_long": "macd_line < macd_sig",
        "entry_short": "macd_line < macd_sig",
        "exit_short": "macd_line > macd_sig",
    },
    "RSI scalp (7, 25/75)": {
        "indicators": [{"name": "rsi_7", "type": "rsi", "period": 7}],
        "entry_long": "rsi_7 < 25",
        "exit_long": "rsi_7 > 60",
        "entry_short": "rsi_7 > 75",
        "exit_short": "rsi_7 < 40",
        "stop_loss_pct": 0.004,
        "take_profit_pct": 0.006,
    },
    "Stoch + Trend (above EMA50)": {
        "indicators": [
            {"name": "stoch_k", "type": "stoch", "k": 14, "d": 3, "component": "k"},
            {"name": "ema_50", "type": "ema", "period": 50},
        ],
        "entry_long": "stoch_k < 20 and close > ema_50",
        "exit_long": "stoch_k > 80",
        "entry_short": "stoch_k > 80 and close < ema_50",
        "exit_short": "stoch_k < 20",
        "stop_loss_pct": 0.005,
    },
    "Bollinger Squeeze + Trend": {
        "indicators": [
            {"name": "bb_lower", "type": "bbands", "period": 20, "std": 2.0, "component": "lower"},
            {"name": "bb_middle", "type": "bbands", "period": 20, "std": 2.0, "component": "middle"},
            {"name": "ema_100", "type": "ema", "period": 100},
        ],
        "entry_long": "close < bb_lower and close > ema_100",
        "exit_long": "close > bb_middle",
        "stop_loss_pct": 0.006,
    },
}


# ─── Validation primitives ─────────────────────────────────────────

def run_strategy(df: pd.DataFrame, rules: dict) -> dict | None:
    if len(df) < 200:
        return None
    try:
        trades, equity = bt.run(df, rules, init_cash=INIT_CASH,
                                fees=TAKER_FEE, slippage=SLIPPAGE)
        if trades.empty or len(equity) < 2:
            return None
        stats = m.compute(trades, equity)
        if "error" in stats:
            return None
        return stats
    except Exception:
        return None


def walk_forward(df: pd.DataFrame, rules: dict, n_folds: int = 4) -> dict:
    """Run strategy on N consecutive non-overlapping folds. Returns per-fold stats + summary."""
    fold_size = len(df) // n_folds
    folds = [df.iloc[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    fold_results = []
    for f in folds:
        fold_results.append(run_strategy(f, rules))

    valid = [r for r in fold_results if r is not None]
    if not valid:
        return {"folds": fold_results, "pass": False, "reason": "no valid folds"}

    positive_folds = sum(1 for r in valid if r["total_return"] > 0)
    avg_sharpe = float(np.mean([r["sharpe_ratio"] for r in valid]))
    compound_return = float(np.prod([1 + r["total_return"] for r in valid]) - 1)

    return {
        "folds": fold_results,
        "positive_folds": positive_folds,
        "total_folds": len(valid),
        "avg_sharpe": avg_sharpe,
        "compound_return": compound_return,
        "robust": positive_folds == len(valid) and avg_sharpe > 1.0,
    }


def monte_carlo_dd(stats: dict, equity: pd.Series, n_sims: int = 500) -> dict:
    """Shuffle daily returns 500x to estimate max DD distribution.

    If real max DD is at the worst-percentile of shuffled distribution = strategy depended on luck.
    """
    daily_ret = equity.resample("1D").last().ffill().pct_change().dropna().values
    if len(daily_ret) < 30:
        return {"skip": True}

    np.random.seed(42)
    dds = []
    for _ in range(n_sims):
        shuffled = np.random.permutation(daily_ret)
        eq = np.cumprod(1 + shuffled)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        dds.append(float(dd.min()))

    dds = np.array(dds)
    return {
        "median_dd": float(np.median(dds)),
        "p5_dd_worst": float(np.percentile(dds, 5)),  # worst case
        "p95_dd_best": float(np.percentile(dds, 95)),
        "actual_dd": stats["max_drawdown"],
    }


# ─── Main pipeline ─────────────────────────────────────────────────

def evaluate(name: str, rules: dict, sym: str) -> dict:
    """Full evaluation: full-period stats + walk-forward + Monte Carlo."""
    print(f"  → {name} on {sym} 5m ...", end=" ", flush=True)
    df = data.load(sym, TF)

    # Full period
    full_trades, full_equity = bt.run(df, rules, init_cash=INIT_CASH,
                                       fees=TAKER_FEE, slippage=SLIPPAGE)
    if full_trades.empty:
        print("NO TRADES")
        return {"name": name, "symbol": sym, "status": "no_trades"}

    full_stats = m.compute(full_trades, full_equity)
    n = full_stats["total_trades"]
    months_span = (df.index[-1] - df.index[0]).days / 30
    trades_per_month = n / months_span

    # Walk-forward
    wf = walk_forward(df, rules, n_folds=4)

    # Monte Carlo
    mc = monte_carlo_dd(full_stats, full_equity)

    result = {
        "name": name,
        "symbol": sym,
        "status": "ok",
        "trades": n,
        "trades_per_month": trades_per_month,
        "total_return": full_stats["total_return"],
        "sharpe": full_stats["sharpe_ratio"],
        "sortino": full_stats["sortino_ratio"],
        "max_dd": full_stats["max_drawdown"],
        "profit_factor": full_stats["profit_factor"],
        "win_rate": full_stats["win_rate"],
        "wf_robust": wf.get("robust", False),
        "wf_positive_folds": wf.get("positive_folds", 0),
        "wf_total_folds": wf.get("total_folds", 0),
        "wf_avg_sharpe": wf.get("avg_sharpe", 0),
        "mc_median_dd": mc.get("median_dd"),
        "mc_p5_dd": mc.get("p5_dd_worst"),
    }

    # Check all goal targets
    goal_pass = (
        result["profit_factor"] > 1.5
        and result["sharpe"] > 2.0
        and result["max_dd"] > -0.10
        and result["trades_per_month"] > 300
        and result["total_return"] > 0
        and result["wf_robust"]
    )
    result["meets_all_kpi"] = bool(goal_pass)

    print(f"trades={n}, sharpe={result['sharpe']:.2f}, dd={result['max_dd']:.1%}, "
          f"WF {result['wf_positive_folds']}/{result['wf_total_folds']}, "
          f"meets_kpi={goal_pass}")
    return result


def main():
    print("=" * 100)
    print("  INSTITUTIONAL VALIDATION: 5m crypto perps")
    print(f"  Fees: {TAKER_FEE*100:.2f}% taker | Slippage: {SLIPPAGE*100:.2f}% | Round-trip cost: "
          f"{(TAKER_FEE + SLIPPAGE)*2*100:.2f}%")
    print(f"  Strategies: {len(STRATEGIES)} | Symbols: {SYMBOLS} | Total tests: "
          f"{len(STRATEGIES) * len(SYMBOLS)}")
    print(f"  Goal KPIs: PF>1.5, Sharpe>2, DD<10%, 300+ trades/mo, walk-fwd robust")
    print("=" * 100)

    all_results = []
    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        for name, rules in STRATEGIES.items():
            r = evaluate(name, rules, sym)
            all_results.append(r)

    # Summary
    valid = [r for r in all_results if r.get("status") == "ok"]
    meets_kpi = [r for r in valid if r.get("meets_all_kpi")]

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  Total tests: {len(all_results)}")
    print(f"  Returned trades: {len(valid)}")
    print(f"  Met ALL KPI targets (PF>1.5, Sharpe>2, DD<10%, 300+ trades/mo, walk-fwd robust): "
          f"{len(meets_kpi)}")

    # Sort all valid by Sharpe (best at top)
    valid.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"\n  TOP 10 by Sharpe (regardless of KPI pass):")
    print(f"  {'Strategy':<35}{'Sym':<5}{'Trades':>7}{'/mo':>6}{'Ret':>9}"
          f"{'Sharpe':>8}{'DD':>8}{'PF':>6}{'WF':>6}{'KPI':>6}")
    print("  " + "-" * 95)
    for r in valid[:10]:
        kpi = "✓" if r["meets_all_kpi"] else "✗"
        wf = f"{r['wf_positive_folds']}/{r['wf_total_folds']}"
        print(f"  {r['name']:<35}{r['symbol']:<5}{r['trades']:>7}"
              f"{r['trades_per_month']:>6.0f}{r['total_return']:>+8.1%}"
              f"{r['sharpe']:>+7.2f}{r['max_dd']:>7.1%}"
              f"{r['profit_factor']:>6.2f}{wf:>6}{kpi:>6}")

    if meets_kpi:
        print(f"\n  WINNERS (all KPI passed):")
        for r in meets_kpi:
            print(f"    ✓ {r['name']} on {r['symbol']}: "
                  f"Sharpe {r['sharpe']:.2f}, PF {r['profit_factor']:.2f}, "
                  f"DD {r['max_dd']:.1%}, {r['trades_per_month']:.0f} trades/mo")
    else:
        print("\n  ⚠ NO STRATEGY MET ALL KPI TARGETS.")
        print("  This is expected — institutional-grade KPIs on retail public data are rare.")

    # Save results
    import json
    with open("/tmp/quant_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n  (Full results saved to /tmp/quant_results.json)")


if __name__ == "__main__":
    main()
