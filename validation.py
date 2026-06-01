"""Advanced validation tests:
1. Rule Significance Test (RST) — shuffle entry signals, prove strategy beats random
2. Candle Monte Carlo — bootstrap synthetic price paths, test robustness
3. Multi-year out-of-sample — test each calendar year independently
4. Iteration loop helpers — orchestrate LLM-driven strategy variants
"""
import json
import numpy as np
import pandas as pd
import vectorbt as vbt

import backtest as bt
import metrics as m
import llm


INIT_CASH = 10_000
RNG = np.random.default_rng(42)


# ─── 1. Rule Significance Test ──────────────────────────────────────

def rule_significance_test(df: pd.DataFrame, rules: dict, n_shuffles: int = 500) -> dict:
    """Shuffle entry signals randomly N times. Strategy has real edge if it beats
    most shuffles. Returns p-value + summary.

    Skipped for pattern mode (entry events are timestamp-specific, shuffling is meaningless).
    """
    if rules.get("pattern"):
        return {"skip": True, "reason": "pattern mode — shuffle test not applicable"}

    try:
        df_ind = bt.add_indicators(df, rules.get("indicators", []))
        real_entries = bt._eval(df_ind, rules.get("entry_long", ""))
        real_exits = bt._eval(df_ind, rules.get("exit_long", ""))
        if rules.get("entry_short"):
            real_short_e = bt._eval(df_ind, rules.get("entry_short", "") or "")
            real_short_x = bt._eval(df_ind, rules.get("exit_short", "") or "")
        else:
            real_short_e = real_short_x = None

        n_entries = int(real_entries.sum())
        if n_entries < 5:
            return {"skip": True, "reason": f"only {n_entries} entries — too few to test"}

        # Real return for comparison
        real_trades, real_equity = bt.run(df, rules, init_cash=INIT_CASH)
        if real_trades.empty:
            return {"skip": True, "reason": "no real trades"}
        real_return = float(real_equity.iloc[-1] / INIT_CASH - 1)

        # Run N shuffled backtests
        n_bars = len(df)
        rng = np.random.default_rng(42)
        shuffled_returns = []

        for _ in range(n_shuffles):
            shuf_e = pd.Series(False, index=df.index)
            idx = rng.choice(n_bars, n_entries, replace=False)
            shuf_e.iloc[idx] = True

            kwargs = dict(
                close=df_ind["close"],
                entries=shuf_e,
                exits=real_exits,
                init_cash=INIT_CASH,
                fees=0.001,
                slippage=0.0005,
            )
            if real_short_e is not None and real_short_e.any():
                n_shorts = int(real_short_e.sum())
                shuf_s = pd.Series(False, index=df.index)
                idx_s = rng.choice(n_bars, n_shorts, replace=False)
                shuf_s.iloc[idx_s] = True
                kwargs["short_entries"] = shuf_s
                kwargs["short_exits"] = real_short_x

            if rules.get("stop_loss_pct") is not None:
                kwargs["sl_stop"] = rules["stop_loss_pct"]
            if rules.get("take_profit_pct") is not None:
                kwargs["tp_stop"] = rules["take_profit_pct"]

            try:
                pf = vbt.Portfolio.from_signals(**kwargs)
                ret = float(pf.value().iloc[-1] / INIT_CASH - 1)
                shuffled_returns.append(ret)
            except Exception:
                continue

        if len(shuffled_returns) < 50:
            return {"skip": True, "reason": f"only {len(shuffled_returns)} shuffles succeeded"}

        arr = np.array(shuffled_returns)
        beats = int((real_return > arr).sum())
        p_value = 1 - (beats / len(arr))
        passed = p_value < 0.05

        return {
            "skip": False,
            "real_return": real_return,
            "shuffled_mean": float(arr.mean()),
            "shuffled_median": float(np.median(arr)),
            "shuffled_p95": float(np.percentile(arr, 95)),
            "shuffled_p5": float(np.percentile(arr, 5)),
            "beats_random": beats,
            "n_shuffles": len(arr),
            "p_value": float(p_value),
            "passed": passed,
            "verdict": (
                f"Real return {real_return:+.1%} vs {len(arr)} random shuffles. "
                f"Beats {beats}/{len(arr)} ({(beats/len(arr))*100:.0f}%). "
                f"p={p_value:.4f} → {'SIGNIFICANT' if passed else 'NOT SIGNIFICANT'}"
            ),
        }
    except Exception as e:
        return {"skip": True, "reason": f"error: {e}"}


# ─── 2. Candle Monte Carlo ──────────────────────────────────────────

def candle_monte_carlo(df: pd.DataFrame, rules: dict, n_sims: int = 100) -> dict:
    """Bootstrap synthetic OHLC paths from real returns, backtest each. Tests
    whether strategy is robust to alternative market histories with same stats.
    """
    try:
        real_trades, real_equity = bt.run(df, rules, init_cash=INIT_CASH)
        if real_trades.empty:
            return {"skip": True, "reason": "no real trades"}

        real_stats = m.compute(real_trades, real_equity)
        real_return = real_stats["total_return"]
        real_sharpe = real_stats["sharpe_ratio"]
        real_dd = real_stats["max_drawdown"]

        # Compute log returns of close
        log_ret = np.log(df["close"]).diff().dropna().values
        if len(log_ret) < 30:
            return {"skip": True, "reason": "not enough history"}

        # Compute typical intra-bar range as fraction of close
        body_range = (df["high"] - df["low"]) / df["close"]
        typical_range = float(body_range.mean())

        rng = np.random.default_rng(42)
        sim_returns = []
        sim_sharpes = []
        sim_dds = []

        for _ in range(n_sims):
            # Bootstrap log returns with replacement
            sim_log = rng.choice(log_ret, len(log_ret), replace=True)
            sim_close = df["close"].iloc[0] * np.exp(np.cumsum(sim_log))
            sim_close = np.insert(sim_close, 0, df["close"].iloc[0])

            # Synthetic OHLC from close (close ± half typical range, jittered)
            jitter_h = rng.uniform(0.3, 0.7, len(sim_close)) * typical_range
            jitter_l = rng.uniform(0.3, 0.7, len(sim_close)) * typical_range
            sim_high = sim_close * (1 + jitter_h)
            sim_low = sim_close * (1 - jitter_l)
            sim_open = np.concatenate([[sim_close[0]], sim_close[:-1]])

            sim_df = pd.DataFrame({
                "open": sim_open,
                "high": sim_high,
                "low": sim_low,
                "close": sim_close,
                "volume": df["volume"].iloc[:len(sim_close)].values,
            }, index=df.index[:len(sim_close)])

            try:
                tr, eq = bt.run(sim_df, rules, init_cash=INIT_CASH)
                if tr.empty:
                    continue
                s = m.compute(tr, eq)
                if "error" in s:
                    continue
                sim_returns.append(s["total_return"])
                sim_sharpes.append(s["sharpe_ratio"])
                sim_dds.append(s["max_drawdown"])
            except Exception:
                continue

        if len(sim_returns) < 20:
            return {"skip": True, "reason": f"only {len(sim_returns)} sims succeeded"}

        ret_arr = np.array(sim_returns)
        sharpe_arr = np.array(sim_sharpes)
        dd_arr = np.array(sim_dds)

        # Position of real result in simulation distribution
        real_percentile = float((ret_arr < real_return).sum() / len(ret_arr) * 100)

        # Robustness verdict: real result in middle 50% = robust, in tails = either lucky or unlucky
        if 30 <= real_percentile <= 70:
            robust = "ROBUST"
            robust_msg = "Real result within middle 50% of simulations — strategy not overly path-dependent."
        elif real_percentile > 90:
            robust = "LIKELY_LUCKY"
            robust_msg = (
                f"Real result better than {real_percentile:.0f}% of simulations — "
                "suspicious, may be path-dependent edge."
            )
        elif real_percentile < 10:
            robust = "BAD_LUCK"
            robust_msg = (
                f"Real result worse than {100-real_percentile:.0f}% of simulations — "
                "strategy could be better in average market."
            )
        else:
            robust = "ACCEPTABLE"
            robust_msg = "Real result reasonably positioned within simulations."

        return {
            "skip": False,
            "real_return": real_return,
            "real_sharpe": real_sharpe,
            "real_dd": real_dd,
            "sim_return_median": float(np.median(ret_arr)),
            "sim_return_p95": float(np.percentile(ret_arr, 95)),
            "sim_return_p5": float(np.percentile(ret_arr, 5)),
            "sim_sharpe_median": float(np.median(sharpe_arr)),
            "sim_dd_p5": float(np.percentile(dd_arr, 5)),  # worst case
            "real_percentile": real_percentile,
            "n_sims": len(ret_arr),
            "robust_label": robust,
            "robust_msg": robust_msg,
        }
    except Exception as e:
        return {"skip": True, "reason": f"error: {e}"}


# ─── 3. Multi-Year Out-of-Sample ────────────────────────────────────

def multi_year_oos(df: pd.DataFrame, rules: dict) -> dict:
    """Run strategy on each calendar year independently. Reveals regime breakdowns."""
    years = sorted(set(df.index.year))
    if len(years) < 2:
        return {"skip": True, "reason": "less than 2 years of data"}

    by_year = {}
    for year in years:
        ydf = df[df.index.year == year]
        if len(ydf) < 30:
            continue
        try:
            tr, eq = bt.run(ydf, rules, init_cash=INIT_CASH)
            bh_ret = float(ydf["close"].iloc[-1] / ydf["close"].iloc[0] - 1)
            if tr.empty:
                by_year[year] = {
                    "trades": 0, "return": 0.0, "sharpe": 0.0,
                    "max_dd": 0.0, "bh_return": bh_ret, "no_trades": True,
                }
                continue
            s = m.compute(tr, eq)
            if "error" in s:
                continue
            by_year[year] = {
                "trades": s["total_trades"],
                "return": s["total_return"],
                "sharpe": s["sharpe_ratio"],
                "max_dd": s["max_drawdown"],
                "bh_return": bh_ret,
                "beats_bh": s["total_return"] > bh_ret,
            }
        except Exception:
            continue

    if not by_year:
        return {"skip": True, "reason": "no valid years"}

    profitable = sum(1 for v in by_year.values() if v.get("return", 0) > 0 and not v.get("no_trades"))
    valid_years = [v for v in by_year.values() if not v.get("no_trades")]
    total = len(valid_years)

    if total == 0:
        verdict = "NO_DATA"
    elif profitable == total:
        verdict = "ROBUST_ACROSS_YEARS"
    elif profitable >= total * 0.7:
        verdict = "MOSTLY_WORKS"
    elif profitable >= total * 0.4:
        verdict = "INCONSISTENT"
    else:
        verdict = "REGIME_DEPENDENT"

    return {
        "skip": False,
        "by_year": by_year,
        "profitable_years": profitable,
        "total_years": total,
        "verdict": verdict,
    }


# ─── 4. Strategy Iteration Loop ─────────────────────────────────────

ITERATION_PROMPT_TEMPLATE = """The following trading strategy was tested and produced these results:

ORIGINAL STRATEGY:
{original_text}

PARSED RULES:
{rules_json}

BACKTEST RESULTS:
- Total return: {total_return:+.1%}
- Sharpe ratio: {sharpe:.2f}
- Max drawdown: {max_dd:.1%}
- Win rate: {win_rate:.0%}
- Total trades: {total_trades}

VALIDATION:
- Rule significance: {rst_status}
- Walk-forward: {wf_status}

Suggest ONE specific modification to improve risk-adjusted return (Sharpe). Examples:
- Tighter or looser entry threshold
- Add a trend filter (e.g., above/below moving average)
- Add ATR-based stop loss
- Change time-frame parameter
- Add volume confirmation

The new strategy will be re-parsed by an automated system, so you MUST answer in
EXACTLY this format — two separate lines, each with its label, nothing else:

REASONING: <one sentence explaining the single change>
STRATEGY: <the COMPLETE new strategy as one self-contained English sentence>

The STRATEGY line must restate the full strategy (entry + exit + the new modification),
not just the change. Do not output JSON. Do not add any other lines."""


def suggest_iteration(original_text: str, rules: dict, stats: dict,
                      rst_passed: bool | None, wf_tier: int | None) -> dict:
    """Ask LLM to propose ONE modification to improve strategy."""
    try:
        rst_status = "SIGNIFICANT" if rst_passed else ("NOT SIGNIFICANT" if rst_passed is False else "SKIPPED")
        wf_status = {1: "ROBUST", 2: "BORDERLINE", 3: "CURVE-FIT", None: "UNKNOWN"}.get(wf_tier, "UNKNOWN")

        prompt = ITERATION_PROMPT_TEMPLATE.format(
            original_text=original_text,
            rules_json=json.dumps(rules, indent=2),
            total_return=stats.get("total_return", 0),
            sharpe=stats.get("sharpe_ratio", 0),
            max_dd=stats.get("max_drawdown", 0),
            win_rate=stats.get("win_rate", 0),
            total_trades=stats.get("total_trades", 0),
            rst_status=rst_status,
            wf_status=wf_status,
        )

        response = llm.client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a quant trader proposing strategy improvements."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()

        # Parse labeled lines: "REASONING:" / "STRATEGY:"
        reasoning = ""
        strategy = ""
        leftover = []
        for line in raw.split("\n"):
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("REASONING:"):
                reasoning = s.split(":", 1)[1].strip()
            elif up.startswith("STRATEGY:"):
                strategy = s.split(":", 1)[1].strip()
            elif s.startswith("//"):
                reasoning = reasoning or s[2:].strip()
            else:
                leftover.append(s)

        # Fallbacks if the model ignored the format
        if not strategy:
            # Prefer any non-reasoning leftover text, else the whole response
            strategy = " ".join(leftover).strip() if leftover else raw
            # Strip a leading reasoning fragment if it leaked in
            if reasoning and strategy.startswith(reasoning):
                strategy = strategy[len(reasoning):].strip()

        if not strategy:
            return {"error": "AI returned an empty strategy — try iterating again."}

        return {"reasoning": reasoning, "strategy": strategy}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import data, time as _t

    print("=== Smoke test all validation functions ===\n")
    df = data.load("SPY")
    rules = {
        "indicators": [{"name": "rsi_14", "type": "rsi", "period": 14}],
        "entry_long": "rsi_14 < 35",
        "exit_long": "rsi_14 > 65",
        "stop_loss_pct": 0.05,
    }

    t0 = _t.time()
    print("1. Rule Significance Test (500 shuffles)...")
    rst = rule_significance_test(df, rules, n_shuffles=500)
    print(f"   done in {_t.time()-t0:.1f}s")
    print(f"   {rst}\n")

    t0 = _t.time()
    print("2. Candle Monte Carlo (100 sims)...")
    mc = candle_monte_carlo(df, rules, n_sims=100)
    print(f"   done in {_t.time()-t0:.1f}s")
    print(f"   {mc}\n")

    t0 = _t.time()
    print("3. Multi-Year OOS...")
    oos = multi_year_oos(df, rules)
    print(f"   done in {_t.time()-t0:.1f}s")
    print(f"   verdict: {oos.get('verdict')}, years: {oos.get('total_years')}, profitable: {oos.get('profitable_years')}")
    if oos.get("by_year"):
        for y, v in sorted(oos["by_year"].items()):
            if v.get("no_trades"):
                print(f"   {y}: no trades")
            else:
                print(f"   {y}: ret={v['return']:+.1%} sharpe={v['sharpe']:+.2f} dd={v['max_dd']:.1%} trades={v['trades']}")
