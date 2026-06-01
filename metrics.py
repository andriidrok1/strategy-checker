"""Backtest metrics + verdict logic."""
import numpy as np
import pandas as pd


def compute(trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Compute backtest metrics from vectorbt trades + equity series."""
    if trades.empty or len(equity) < 2:
        return {"error": "no trades"}

    pnls = trades["PnL"] if "PnL" in trades.columns else trades.get("net_pnl", pd.Series(dtype=float))

    total_trades = len(trades)
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    win_rate = len(winners) / total_trades if total_trades else 0.0

    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(abs(losers.mean())) if len(losers) else 0.0

    profit_factor = (
        float(winners.sum() / abs(losers.sum())) if len(losers) and losers.sum() != 0
        else float("inf") if len(winners) else 0.0
    )

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Returns — daily resample handles weekends/gaps correctly
    daily = equity.resample("1D").last().ffill().pct_change().dropna()

    # Annualized Sharpe (252 for stocks, 365 for crypto — use 252 as conservative default)
    sharpe = float(daily.mean() / (daily.std() + 1e-10) * np.sqrt(252)) if len(daily) > 1 else 0.0

    downside = daily[daily < 0]
    sortino = (
        float(daily.mean() / (downside.std() + 1e-10) * np.sqrt(252))
        if len(downside) > 1 else 0.0
    )

    # Max drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    recovery_factor = float(abs(total_return / max_dd)) if max_dd != 0 else 0.0

    return {
        "total_trades": int(total_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "recovery_factor": recovery_factor,
        "final_equity": float(equity.iloc[-1]),
    }


def verdict(stats: dict) -> dict:
    """Categorize strategy quality. Returns label, color, reason.

    Tiering by sample size:
    - < 10 trades: truly meaningless
    - 10-29 trades: borderline — show metrics with caveat
    - 30+ trades: full Works/Mixed/Doesn't verdict
    """
    if "error" in stats:
        return {"label": "No trades", "color": "gray",
                "reason": "strategy never triggered on this data"}

    n = stats.get("total_trades", 0)
    if n < 10:
        return {"label": "No data", "color": "gray",
                "reason": f"only {n} trades — completely meaningless. try a different symbol or loosen entry conditions."}

    ret = stats["total_return"]
    sharpe = stats["sharpe_ratio"]
    max_dd = stats["max_drawdown"]

    # Determine base label by metrics
    if ret < 0 or sharpe < 0:
        base_label, base_color = "Doesn't work", "red"
        base_reason = f"return {ret:.1%}, Sharpe {sharpe:.2f} — loses money"
    elif sharpe > 1.0 and ret > 0 and max_dd > -0.25:
        base_label, base_color = "Works", "green"
        base_reason = f"return {ret:.1%}, Sharpe {sharpe:.2f}, max DD {max_dd:.1%} — solid"
    else:
        base_label, base_color = "Mixed", "yellow"
        base_reason = f"profitable but weak — Sharpe {sharpe:.2f}, max DD {max_dd:.1%}"

    # Add caveat for borderline samples
    if n < 30:
        return {
            "label": f"{base_label} (small sample)",
            "color": base_color,
            "reason": f"{base_reason}. ⚠ only {n} trades — take with grain of salt, run on more data or longer timeframe.",
        }

    return {"label": base_label, "color": base_color, "reason": base_reason}
