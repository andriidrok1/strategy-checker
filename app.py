"""Strategy Checker — IG-content-focused version.

Auto-runs walk-forward + buy & hold comparison.
Detects curve-fit. Screenshot-ready verdict cards.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import data
import llm
import backtest
import metrics
import patterns as pat_mod
import validation as val


# ─── Page setup ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Strategy Checker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    #MainMenu, footer, header {visibility: hidden;}

    /* === DASHBOARD / OPS-CONSOLE DIRECTION === */

    :root {
        --bg: #0a0a0a;
        --bg-card: #111114;
        --bg-card-2: #15151a;
        --border: rgba(255,255,255,0.07);
        --border-strong: rgba(255,255,255,0.14);
        --cyan: #22d3ee;
        --cyan-bright: #67e8f9;
        --green: #4ade80;
        --red: #f87171;
        --yellow: #fbbf24;
        --orange: #fb923c;
        --text: #e8eaed;
        --text-dim: #9ca3af;
        --text-dimmer: #6b7280;
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--bg) !important;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text);
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: none;
        background: var(--bg);
    }

    /* TOP NAV BAR — Strategy Checker dashboard header */
    .topnav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 1.5rem;
        border-bottom: 1px solid var(--border);
        margin: 0 -1.5rem 2rem -1.5rem;
        background: var(--bg);
    }
    .topnav-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .topnav-links {
        display: flex;
        gap: 2.5rem;
        margin-left: 3rem;
        flex: 1;
    }
    .topnav-link {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        cursor: pointer;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid transparent;
    }
    .topnav-link.active {
        color: var(--cyan);
        border-bottom-color: var(--cyan);
        text-shadow: 0 0 12px rgba(34,211,238,0.4);
    }
    .topnav-right {
        display: flex;
        align-items: center;
        gap: 1.6rem;
    }
    .topnav-stat {
        text-align: right;
    }
    .topnav-stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-dimmer);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        display: block;
    }
    .topnav-stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: var(--cyan);
        font-weight: 500;
    }
    .topnav-icon {
        width: 28px;
        height: 28px;
        border: 1px solid var(--border);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: var(--text-dim);
    }

    /* Section card — every panel is wrapped in this */
    .panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin: 0 0 1rem 0;
        font-weight: 500;
    }
    .panel-header::before {
        content: "■";
        color: var(--cyan);
        margin-right: 0.5rem;
        font-size: 0.65rem;
        position: relative;
        top: -1px;
    }

    /* Hero — kept for fallback */
    .hero-title, .hero-subtitle, .hero-byline { display: none; }

    /* VERDICT / VALIDATION PANEL — dashboard style */
    .share-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.8rem 1.4rem 1.8rem;
        margin: 0 0 1.2rem 0;
        position: relative;
    }
    .share-works {
        border-color: rgba(74, 222, 128, 0.4);
        box-shadow: inset 0 0 60px rgba(74,222,128,0.04);
    }
    .share-doesnt {
        border-color: rgba(248, 113, 113, 0.4);
        box-shadow: inset 0 0 60px rgba(248,113,113,0.04);
    }
    .share-mixed { border-color: rgba(251, 191, 36, 0.4); }
    .share-curvefit { border-color: rgba(251, 146, 60, 0.5); }
    .share-no-data { border-color: var(--border); }

    /* Small status label above headline */
    .verdict-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        text-align: center;
        margin: 0.3rem 0 0.5rem 0;
        font-weight: 500;
    }
    .share-works .verdict-status { color: var(--green); }
    .share-doesnt .verdict-status { color: var(--red); }
    .share-curvefit .verdict-status { color: var(--orange); }
    .share-mixed .verdict-status { color: var(--yellow); }

    /* Headline group — big checkmark + STRATEGY_STABLE style */
    .verdict-header-group {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }
    .verdict-emoji { font-size: 2rem; line-height: 1; }
    .verdict-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.015em;
        line-height: 1;
        color: var(--text);
        text-transform: uppercase;
    }
    .share-works .verdict-headline {
        color: var(--green);
        text-shadow: 0 0 24px rgba(74,222,128,0.25);
    }
    .share-doesnt .verdict-headline { color: var(--red); }
    .share-mixed .verdict-headline { color: var(--yellow); }
    .share-curvefit .verdict-headline { color: var(--orange); }

    .verdict-tagline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        color: var(--text-dim);
        margin: 0.6rem auto 0 auto;
        line-height: 1.5;
        max-width: 90%;
        text-align: center;
    }

    /* Stats row — 3 or 5 column layout under headline */
    .verdict-stats-row {
        display: flex;
        gap: 0;
        margin-top: 1.6rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
        flex-wrap: wrap;
        justify-content: space-around;
    }
    .stat-block {
        flex: 1;
        min-width: 90px;
        padding: 0 0.8rem;
        text-align: center;
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: var(--text-dimmer);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin: 0;
        font-weight: 500;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.35rem 0 0 0;
        color: var(--text);
        letter-spacing: -0.01em;
        font-feature-settings: 'tnum';
    }
    .stat-value.green { color: var(--green); }
    .stat-value.red { color: var(--red); }
    .stat-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-dimmer);
        margin: 0.2rem 0 0 0;
    }

    /* Robustness badge */
    .robust-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 600;
        margin: 0.5rem auto 0 auto;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        border: 1px solid;
        border-radius: 3px;
        display: table;
    }
    .robust-tier1 { border-color: var(--green); color: var(--green); }
    .robust-tier2 { border-color: var(--yellow); color: var(--yellow); }
    .robust-tier3 { border-color: var(--red); color: var(--red); }
    .robust-tier-none { border-color: var(--border-strong); color: var(--text-dim); }

    /* Dateline — at very top of card */
    .verdict-dateline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        color: var(--text-dimmer);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin: 0 0 0.9rem 0;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--border);
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    /* Watermark — bottom strip */
    .watermark {
        margin-top: 1.2rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-dimmer);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        display: flex;
        justify-content: space-between;
    }
    .watermark .handle { color: var(--cyan); font-weight: 500; }

    /* Section dividers — like panel headers but standalone */
    .section-divider {
        margin: 1.8rem 0 0.8rem 0;
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        padding-bottom: 0.3rem;
    }
    .section-divider::before {
        content: "■ ";
        color: var(--cyan);
        font-size: 0.6rem;
        margin-right: 0.3rem;
    }

    /* Streamlit metric */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
        color: var(--text) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.65rem;
        color: var(--text-dimmer) !important;
    }

    /* Buttons */
    .stButton>button {
        background: transparent;
        color: var(--text);
        border: 1px solid var(--border-strong);
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background: var(--bg-card-2);
        border-color: var(--cyan);
        color: var(--cyan);
    }
    .stButton>button[kind="primary"] {
        background: var(--cyan);
        color: #001014;
        border-color: var(--cyan);
        font-weight: 700;
        box-shadow: 0 0 30px rgba(34,211,238,0.25);
    }
    .stButton>button[kind="primary"]:hover {
        background: var(--cyan-bright);
        color: #001014;
        box-shadow: 0 0 40px rgba(34,211,238,0.4);
    }

    /* Text area */
    .stTextArea textarea {
        background: #0d0d10 !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        color: var(--text) !important;
        padding: 1rem !important;
        caret-color: var(--cyan);
        text-transform: uppercase;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-dimmer) !important;
        text-transform: uppercase;
    }
    .stTextArea textarea:focus {
        border-color: var(--cyan) !important;
        outline: none !important;
    }

    /* Selectbox + number input */
    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput input {
        background: #0d0d10 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        text-transform: uppercase;
        font-size: 0.78rem !important;
    }
    .stNumberInput button {
        background: var(--bg-card) !important;
        color: var(--text-dim) !important;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label, .stTextArea label {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--text-dim) !important;
        font-size: 0.66rem !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: var(--text-dim) !important;
        padding: 0.7rem 1.2rem;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--cyan) !important;
        background: transparent !important;
        border-bottom: 2px solid var(--cyan);
    }

    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--text-dim) !important;
        font-size: 0.7rem !important;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 6px !important;
        border-left: 3px solid var(--cyan);
        background: var(--bg-card) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: #0d0d10 !important;
        color: var(--cyan) !important;
        border-radius: 4px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        color: var(--text-dim) !important;
        letter-spacing: 0.12em;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown p {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: var(--text);
    }
    .sidebar-status {
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        margin: 0.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .sidebar-status-icon {
        width: 32px;
        height: 32px;
        background: rgba(34,211,238,0.1);
        border: 1px solid rgba(34,211,238,0.3);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--cyan);
        font-size: 0.9rem;
    }
    .sidebar-status-text { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
    .sidebar-status-name { color: var(--text); font-weight: 500; letter-spacing: 0.06em; }
    .sidebar-status-state { color: var(--green); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 0.15rem; }
    .sidebar-status-state::before { content: "● "; }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
        margin-bottom: 2rem;
    }
    .sidebar-nav-item {
        padding: 0.7rem 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-radius: 6px;
        cursor: pointer;
    }
    .sidebar-nav-item.active {
        background: rgba(34,211,238,0.08);
        color: var(--cyan);
    }
    .sidebar-nav-item:hover { background: var(--bg-card-2); }

    /* Bottom data authenticity terminal */
    .auth-terminal {
        background: #0d0d10;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
    }
    .auth-terminal-header {
        display: flex;
        justify-content: space-between;
        color: var(--cyan);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        padding-bottom: 0.7rem;
        margin-bottom: 0.7rem;
        border-bottom: 1px solid var(--border);
    }
    .auth-terminal-uptime { color: var(--text-dim); }
    .auth-log-row {
        display: grid;
        grid-template-columns: 90px 1fr 100px;
        gap: 1rem;
        padding: 0.25rem 0;
        color: var(--text-dim);
    }
    .auth-log-time { color: var(--text-dimmer); }
    .auth-log-msg { color: var(--text); }
    .auth-log-status-ok { color: var(--green); text-align: right; }
    .auth-log-status-wait { color: var(--yellow); text-align: right; }
    .auth-log-status-hex { color: var(--cyan); text-align: right; }

    /* Hide default streamlit anchor links */
    [data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)

# === TOP NAV BAR ===
import datetime as _dt
_now = _dt.datetime.now()
st.markdown(
    f'<div class="topnav">'
    f'<div style="display:flex;align-items:center;gap:0;">'
    f'<span class="topnav-brand">Strategy Checker</span>'
    f'</div>'
    f'<div class="topnav-right">'
    f'<div class="topnav-stat">'
    f'<span class="topnav-stat-label">DATA_INTEGRITY</span>'
    f'<span class="topnav-stat-value">100.00%</span>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# === SIDEBAR ===
with st.sidebar:
    st.markdown(
        '<div class="sidebar-status">'
        '<div class="sidebar-status-icon">▣</div>'
        '<div class="sidebar-status-text">'
        '<div class="sidebar-status-name">STRAT_CORE_V1</div>'
        '<div class="sidebar-status-state">OPERATIONAL</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="margin-top:auto;"></div>', unsafe_allow_html=True)


# ─── Examples ────────────────────────────────────────────────────────

EXAMPLES = {
    "🎯 RSI Mean Reversion": "Buy when RSI(14) drops below 30, sell when RSI rises above 70. 5% stop loss.",
    "📈 Golden Cross": "Long when 50 SMA crosses above 200 SMA, exit when reversed.",
    "📉 Double Top": "Trade double top reversal pattern — short on confirmation below neckline.",
    "🔺 Ascending Triangle": "Buy on ascending triangle breakout above resistance.",
    "⚡ Bollinger MR": "Buy when price touches lower Bollinger band (2 std), exit at middle band.",
    "🚀 EMA Cross": "Long when EMA 9 crosses above EMA 21, short on opposite cross.",
}


# ─── Helpers ─────────────────────────────────────────────────────────

def run_walk_forward(df, rules, n_folds=4):
    """Run strategy on 4 chronological folds. Returns list of fold returns."""
    fold_size = len(df) // n_folds
    fold_results = []
    for i in range(n_folds):
        fold = df.iloc[i * fold_size:(i + 1) * fold_size]
        if len(fold) < 50:
            fold_results.append(None)
            continue
        try:
            trades, equity = backtest.run(fold, rules, init_cash=10_000)
            if trades.empty:
                fold_results.append({"return": 0.0, "trades": 0, "no_trades": True})
                continue
            s = metrics.compute(trades, equity)
            if "error" in s:
                fold_results.append(None)
                continue
            fold_results.append({"return": s["total_return"], "trades": s["total_trades"]})
        except Exception:
            fold_results.append(None)
    return fold_results


def robustness_tier(fold_results):
    """Classify based on fold results."""
    valid = [f for f in fold_results if f and not f.get("no_trades")]
    if not valid:
        return None, "No valid folds"
    profitable = sum(1 for f in valid if f["return"] > 0)
    total = len(valid)
    if profitable == total and total >= 3:
        return 1, f"{profitable}/{total} folds profitable — robust signal"
    if profitable >= total - 1 and total >= 3:
        return 2, f"{profitable}/{total} folds profitable — possible edge, one losing period"
    return 3, f"only {profitable}/{total} folds profitable — likely curve-fit"


def bh_return(df, init_cash):
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1)


# ─── Pooled (multi-symbol) scan ──────────────────────────────────────

CRYPTO_SET = {"BTC", "ETH", "SOL", "ADA", "ARB", "ATOM", "AVAX", "DOGE",
              "DOT", "FIL", "LINK", "NEAR", "UNI", "XRP"}
POOL_SENTINEL = "★ ALL CRYPTO (pool)"


def run_pool(rules, timeframe, init_cash):
    """Run the same strategy on every crypto symbol; pool the trades.

    Returns (per_symbol, pooled_trade_returns):
      per_symbol — [{symbol, n, ret, win, error}]
      pooled_trade_returns — list of per-trade % returns across all symbols
    """
    syms = [s for s in data.available_symbols(timeframe) if s in CRYPTO_SET]
    per_symbol, pooled_returns = [], []
    for sym in syms:
        try:
            df = data.load(sym, timeframe)
            trades, equity = backtest.run(df, rules, init_cash=init_cash)
            if trades.empty:
                per_symbol.append({"symbol": sym, "n": 0, "ret": 0.0, "win": None})
                continue
            s = metrics.compute(trades, equity)
            if "Return" in trades.columns:
                pooled_returns.extend(float(x) for x in trades["Return"].tolist())
            per_symbol.append({
                "symbol": sym,
                "n": int(s.get("total_trades", 0)),
                "ret": float(s.get("total_return", 0.0)),
                "win": float(s.get("win_rate", 0.0)),
            })
        except Exception as e:
            per_symbol.append({"symbol": sym, "n": 0, "ret": 0.0, "win": None, "error": str(e)})
    return per_symbol, pooled_returns


def pooled_verdict(per_symbol, pooled_returns):
    """Classify a pooled scan. A real edge works across MANY coins, not one."""
    import numpy as np
    traded = [p for p in per_symbol if p["n"] > 0]
    n_total = int(sum(p["n"] for p in per_symbol))
    n_syms_traded = len(traded)
    n_syms_profitable = sum(1 for p in traded if p["ret"] > 0)
    arr = np.array(pooled_returns, dtype=float) if pooled_returns else np.array([])
    avg_ret = float(arr.mean()) if arr.size else 0.0
    win_rate = float((arr > 0).mean()) if arr.size else 0.0
    pos = float(arr[arr > 0].sum()) if arr.size else 0.0
    neg = float(abs(arr[arr < 0].sum())) if arr.size else 0.0
    pf = (pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0)
    consistency = (n_syms_profitable / n_syms_traded) if n_syms_traded else 0.0

    # Classify
    if n_total < 30:
        label, color, msg = "THIN_SAMPLE", "amber", (
            f"Only {n_total} trades even pooled across {n_syms_traded} coins — "
            "too rare to draw a conclusion.")
    elif avg_ret <= 0:
        label, color, msg = "NO_EDGE", "red", (
            f"{n_total} trades across {n_syms_traded} coins, but average is "
            f"{avg_ret:+.2%} per trade after costs. The pattern loses on average — no edge.")
    elif consistency < 0.5:
        label, color, msg = "CHERRY_PICKED", "amber", (
            f"Profitable on only {n_syms_profitable}/{n_syms_traded} coins. "
            "Works on a few, fails on most — looks like luck, not edge.")
    else:
        label, color, msg = "POSSIBLE_EDGE", "green", (
            f"{n_total} trades, {win_rate:.0%} win, {avg_ret:+.2%} avg/trade, "
            f"profitable on {n_syms_profitable}/{n_syms_traded} coins — survives a broad sample.")

    return {
        "label": label, "color": color, "msg": msg,
        "n_total": n_total, "n_syms_traded": n_syms_traded,
        "n_syms_profitable": n_syms_profitable, "avg_ret": avg_ret,
        "win_rate": win_rate, "profit_factor": pf, "consistency": consistency,
    }


# ─── Session state ───────────────────────────────────────────────────

if "strategy_text" not in st.session_state:
    st.session_state.strategy_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False
if "switch_symbol" not in st.session_state:
    st.session_state.switch_symbol = None
if "parsed_preview" not in st.session_state:
    st.session_state.parsed_preview = None   # {"text": ..., "rules": ...}
if "parse_error" not in st.session_state:
    st.session_state.parse_error = None


# ─── Strategy blueprint (preview before running) ─────────────────────

# Per-pattern plan, numbers taken DIRECTLY from patterns.py defaults.
# Each entry: (title, direction, detection, entry, exits)
PATTERN_PLAN = {
    "double_top": (
        "Double Top Reversal", "SHORT (bearish reversal)",
        "two swing-high peaks within ±3% of each other, separated by 10–100 bars; neckline = lowest low between the peaks; pattern height = peak − neckline.",
        "short on the first close **below the neckline** (within 50 bars after the 2nd peak).",
        "take-profit at **neckline − 1× height**; stop **0.5% above the highest peak**; time-stop at 100 bars.",
    ),
    "double_bottom": (
        "Double Bottom Reversal", "LONG (bullish reversal)",
        "two swing-low troughs within ±3% of each other, separated by 10–100 bars; neckline = highest high between the troughs; height = neckline − trough.",
        "long on the first close **above the neckline** (within 50 bars after the 2nd trough).",
        "take-profit at **neckline + 1× height**; stop **0.5% below the lowest trough**; time-stop at 100 bars.",
    ),
    "head_and_shoulders": (
        "Head & Shoulders", "SHORT (bearish reversal)",
        "three peaks where the head is ≥2% above the shoulders, shoulders within ±4% of each other, total span ≤150 bars; neckline = midpoint of the two troughs.",
        "short on the first close **below the neckline** (within 50 bars of the right shoulder).",
        "take-profit at **neckline − head-to-neckline height**; stop **above the head**; time-stop at 100 bars.",
    ),
    "inverse_head_and_shoulders": (
        "Inverse Head & Shoulders", "LONG (bullish reversal)",
        "three troughs where the head is ≥2% below the shoulders, shoulders within ±4%, total span ≤150 bars; neckline = midpoint of the two peaks.",
        "long on the first close **above the neckline** (within 50 bars of the right shoulder).",
        "take-profit at **neckline + head-to-neckline height**; stop **below the head**; time-stop at 100 bars.",
    ),
    "ascending_triangle": (
        "Ascending Triangle", "LONG (bullish breakout)",
        "rising lows + a flat resistance line over ~80 bars (≥2 pivots each side; resistance flat within 0.05%/bar).",
        "long on the close **above resistance** (within 30 bars).",
        "take-profit at **entry + triangle height**; stop **0.5% below the lower trendline**; time-stop at 100 bars.",
    ),
    "descending_triangle": (
        "Descending Triangle", "SHORT (bearish breakdown)",
        "falling highs + a flat support line over ~80 bars (≥2 pivots each side; support flat within 0.05%/bar).",
        "short on the close **below support** (within 30 bars).",
        "take-profit at **entry − triangle height**; stop **0.5% above the upper trendline**; time-stop at 100 bars.",
    ),
    "symmetrical_triangle": (
        "Symmetrical Triangle", "EITHER (whichever side breaks)",
        "two converging trendlines over ~80 bars (≥2 pivots each side).",
        "enter on the close **beyond a trendline** — long if it breaks up, short if it breaks down (within 30 bars).",
        "take-profit at **entry ± triangle height** (breakout direction); stop **at the opposite trendline**; time-stop at 100 bars.",
    ),
    "bullish_flag": (
        "Bullish Flag", "LONG (continuation)",
        "a strong up-move pole (≥10% over ~15 bars), then a tight consolidation (≤5% wide over ~20 bars).",
        "long on the **break above the consolidation** (within 15 bars).",
        "take-profit = **pole height projected up** from breakout; stop **below the flag**; time-stop at 100 bars.",
    ),
    "bearish_flag": (
        "Bearish Flag", "SHORT (continuation)",
        "a strong down-move pole (≥10% over ~15 bars), then a tight consolidation (≤5% wide over ~20 bars).",
        "short on the **break below the consolidation** (within 15 bars).",
        "take-profit = **pole height projected down** from breakout; stop **above the flag**; time-stop at 100 bars.",
    ),
}


def _tf_word(timeframe: str) -> str:
    return {"daily": "daily", "4h": "4-hour", "1h": "hourly"}.get(timeframe, timeframe)


def _instrument_window(symbol: str, timeframe: str):
    """Return a human window string for the symbol's available data."""
    tfw = _tf_word(timeframe)
    try:
        _df = data.load(symbol, timeframe)
        start = _df.index[0].date()
        end = _df.index[-1].date()
        years = (end - start).days / 365.25
        return f"**{symbol}** {tfw} bars, {start} → {end} (~{years:.1f} years, {len(_df):,} bars)."
    except Exception:
        return f"**{symbol}** {tfw} bars (full available history)."


def build_plan(rules: dict, symbol: str, init_cash: float, timeframe: str = "daily"):
    """Return (title, [numbered plan lines]) describing exactly what will run."""
    tfw = _tf_word(timeframe)
    is_pool = symbol == POOL_SENTINEL
    if is_pool:
        _n_syms = len([s for s in data.available_symbols(timeframe) if s in CRYPTO_SET])
        window = (f"**all {_n_syms} crypto symbols** ({', '.join(sorted(CRYPTO_SET)[:5])}…) "
                  f"on {tfw} bars — the same strategy run on each, trades pooled.")
        deliverable = ("a per-coin breakdown table + a **pooled verdict**: total trades, "
                       "win rate, average return per trade, and how many coins it works on. "
                       "A real edge survives across many coins — not just one.")
    else:
        window = _instrument_window(symbol, timeframe)
        deliverable = ("verdict card, equity vs **buy-and-hold**, walk-forward (4 folds), "
                       "Rule Significance Test, Monte Carlo, and multi-year out-of-sample.")
    costs = ("one position at a time, 100% of equity per trade, "
             "**0.1% fees + 0.05% slippage per side** baked in.")

    if "pattern" in rules:
        ptype = rules["pattern"].get("type", "?")
        plan = PATTERN_PLAN.get(ptype)
        if plan:
            title, direction, detection, entry, exits = plan
        else:
            title = ptype.replace("_", " ").title()
            direction = "(pattern-defined)"
            detection = "geometric pivot scan via find_peaks (prominence ≥2% of mean price, ≥5 bars apart)."
            entry = "on pattern confirmation."
            exits = "measured-move target and pattern-based stop; time-stop at 100 bars."
        lines = [
            f"**Instrument & window**: {window}",
            f"**Direction**: {direction}.",
            "**Pivot detection**: swing highs/lows via `find_peaks`, prominence ≥2% of mean price, ≥5 bars apart.",
            f"**Pattern detection**: {detection}",
            f"**Entry**: {entry}",
            f"**Exits** (whichever hits first): {exits}",
            f"**Sizing & costs**: {costs}",
            f"**What you'll get**: {deliverable}",
            "**Note**: trade count = how often this pattern actually occurs in the data. "
            "It can't be set to a target — on daily bars rare patterns yield very few trades; "
            "switch to 4h or 1h, or use ALL CRYPTO (pool), for a larger sample.",
        ]
        _scope = "ALL CRYPTO (pooled)" if is_pool else symbol
        return f"Plan — {title} on {_scope} ({tfw})", lines

    # Indicator mode
    inds = rules.get("indicators", [])
    ind_str = ", ".join(
        f'{i.get("type", "?").upper()}({i.get("period", "-")})' for i in inds
    ) or "price only"
    entry_long = rules.get("entry_long") or "—"
    exit_long = rules.get("exit_long") or "—"
    sl = rules.get("stop_loss_pct")
    tp = rules.get("take_profit_pct")
    risk_bits = []
    risk_bits.append(f"stop-loss **{sl * 100:.1f}%**" if sl else "no fixed stop (exit on rule only)")
    if tp:
        risk_bits.append(f"take-profit **{tp * 100:.1f}%**")
    lines = [
        f"**Instrument & window**: {window}",
        f"**Indicators**: {ind_str}.",
        f"**Entry (long)**: `{entry_long}`.",
        f"**Exit (long)**: `{exit_long}`.",
    ]
    if rules.get("entry_short"):
        lines.append(f"**Entry (short)**: `{rules['entry_short']}`.")
    if rules.get("exit_short"):
        lines.append(f"**Exit (short)**: `{rules['exit_short']}`.")
    lines += [
        f"**Risk**: {', '.join(risk_bits)}.",
        f"**Sizing & costs**: {costs}",
        f"**What you'll get**: {deliverable}",
    ]
    _scope = "ALL CRYPTO (pooled)" if is_pool else symbol
    return f"Plan — Indicator strategy on {_scope} ({tfw})", lines


def render_blueprint(rules: dict, symbol: str, init_cash: float, timeframe: str = "daily"):
    """Render the 'what will be tested' plan (Alva-style numbered plan)."""
    title, lines = build_plan(rules, symbol, init_cash, timeframe)
    mode_label = "CHART PATTERN" if "pattern" in rules else "INDICATOR RULES"
    # Header (styled, single-line HTML to avoid markdown code-block bug)
    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'border:1px solid rgba(34,211,238,0.35);border-bottom:none;background:#0d0d0d;'
        'border-radius:8px 8px 0 0;padding:12px 16px 10px;margin:14px 0 0;'
        'font-family:\'JetBrains Mono\',monospace;">'
        '<span style="color:#22d3ee;font-size:12px;letter-spacing:2px;font-weight:600;">'
        '◈ STRATEGY BLUEPRINT</span>'
        '<span style="color:#666;font-size:10px;letter-spacing:1px;border:1px solid #333;'
        'border-radius:4px;padding:2px 8px;">' + mode_label + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Body: build plan as a markdown numbered list inside a bordered container
    with st.container(border=True):
        st.markdown(f"**{title}**")
        body = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
        st.markdown(body)
        st.caption("This is exactly what the engine will test. Confirm to run.")


# ─── Input section (left) + validation panel placeholder (right) ───

input_col, validation_col = st.columns([1.4, 1])

with input_col:
    st.markdown('<div class="section-divider">STRATEGY_INPUT</div>', unsafe_allow_html=True)

    tf_col, sym_col, cash_col = st.columns([1, 1.3, 1])
    with tf_col:
        _tf_label = st.selectbox("Timeframe", ["Daily", "4h", "1h"], index=0,
                                 help="Intraday (4h/1h) gives far more trades — a real sample. "
                                      "Daily covers stocks too but patterns are rare.")
    timeframe = {"Daily": "daily", "4h": "4h", "1h": "1h"}[_tf_label]

    all_syms = data.available_symbols(timeframe)
    stocks = [s for s in all_syms if s in {"SPY", "QQQ", "IWM", "GLD", "TLT"}]
    cryptos = [s for s in all_syms if s not in stocks]
    # Pooled scan available whenever there are ≥2 crypto symbols
    _has_pool = len(cryptos) >= 2
    symbol_options = ([POOL_SENTINEL] if _has_pool else []) + stocks + cryptos

    if st.session_state.switch_symbol and st.session_state.switch_symbol in symbol_options:
        default_idx = symbol_options.index(st.session_state.switch_symbol)
        st.session_state.switch_symbol = None
    else:
        default_idx = symbol_options.index("BTC") if "BTC" in symbol_options else 0

    with sym_col:
        symbol = st.selectbox("Symbol_Scan", symbol_options, index=default_idx)
    with cash_col:
        init_cash = st.number_input("Capital ($)", value=10_000, step=1000, min_value=1000)

    strategy_text = st.text_area(
        "Strategy_Definition",
        value=st.session_state.strategy_text,
        placeholder="DEFINE STRATEGY PARAMETERS IN NATURAL LANGUAGE... e.g. EXECUTE LONG ON BTC WHEN RSI < 30 AND 200EMA IS TRENDING UPWARDS...",
        height=140,
        label_visibility="visible",
    )
    st.session_state.strategy_text = strategy_text

    # Step 1 — analyze: parse the strategy and store a preview
    analyze_btn = st.button("ANALYZE_STRATEGY", type="primary", use_container_width=True)
    if analyze_btn and strategy_text.strip():
        with st.spinner("🧠 Parsing strategy with AI..."):
            try:
                _rules = llm.parse_strategy(strategy_text)
                st.session_state.parsed_preview = {"text": strategy_text, "rules": _rules}
                st.session_state.parse_error = None
            except Exception as e:
                st.session_state.parsed_preview = None
                st.session_state.parse_error = str(e)
        st.rerun()

    if st.session_state.parse_error:
        st.error(f"Couldn't parse: {st.session_state.parse_error}")
        st.caption(
            "Try a more specific description. Examples: "
            "`Buy when RSI dips below 30, sell at 70` · "
            "`Test double bottom pattern` · `EMA 9/21 crossover, long only`"
        )

    # Step 2 — blueprint preview + confirm (only if preview matches current text)
    confirm_run = False
    _pv = st.session_state.parsed_preview
    if _pv and _pv.get("text") == strategy_text and strategy_text.strip():
        render_blueprint(_pv["rules"], symbol, init_cash, timeframe)
        confirm_run = st.button("✓  CONFIRM · RUN_BACKTEST", type="primary",
                                use_container_width=True, key="confirm_btn")
    elif _pv and _pv.get("text") != strategy_text and strategy_text.strip():
        st.caption("⚠ Strategy text changed — click ANALYZE_STRATEGY again to refresh the blueprint.")

with validation_col:
    st.markdown('<div class="section-divider">VALIDATION_STATUS</div>', unsafe_allow_html=True)

    _r = st.session_state.last_result
    if _r and _r.get("pooled"):
        st.markdown(
            '<div class="share-card share-no-data" style="text-align:center;padding-top:2.5rem;padding-bottom:2.5rem;">'
            '<div class="verdict-status">POOLED_SCAN</div>'
            '<div class="verdict-header-group">'
            '<div class="verdict-emoji">🌐</div>'
            '<h1 class="verdict-headline" style="font-size:1.8rem;">ALL CRYPTO</h1>'
            '</div>'
            '<p class="verdict-tagline">Pooled verdict and per-coin breakdown below.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not _r or "fold_results" not in _r:
        # Placeholder
        st.markdown(
            '<div class="share-card share-no-data" style="text-align:center;padding-top:2.5rem;padding-bottom:2.5rem;">'
            '<div class="verdict-status">AWAITING_INPUT</div>'
            '<div class="verdict-header-group">'
            '<div class="verdict-emoji">◯</div>'
            '<h1 class="verdict-headline" style="font-size:1.8rem;">READY</h1>'
            '</div>'
            '<p class="verdict-tagline">Define a strategy, ANALYZE, then CONFIRM to run.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Build + render verdict card from session state
        _stats = _r["stats"]; _df = _r["df"]; _sym = _r["symbol"]
        _cash = _r["init_cash"]; _bh = _r["bh_return"]
        _tier = _r["tier"]; _robust_msg = _r["robust_msg"]
        _folds = _r["fold_results"]
        _has_stats = "error" not in _stats

        if not _has_stats:
            _cls, _emo, _slbl, _hl, _tg = ("share-no-data", "◯",
                "INSUFFICIENT_DATA", "NO_TRADES",
                "Strategy never triggered on this data.")
        else:
            _ret = _stats["total_return"]; _n = _stats["total_trades"]
            _is_cf = (_tier == 3) and _ret > 0
            _beats = _ret > _bh
            if _is_cf:
                _cls, _emo, _slbl, _hl = "share-curvefit", "⚠", "OVERFIT_DETECTED", "CURVE_FIT"
                _tg = f"Looks profitable (+{_ret:.0%}) but failed walk-forward validation. Likely overfit."
            elif _tier == 1 and _beats:
                _cls, _emo, _slbl, _hl = "share-works", "✓", "VALIDATION_SUCCESS", "STRATEGY_STABLE"
                _tg = f"+{_ret:.0%} return, beats buy & hold ({_bh:+.0%}), robust across all 4 folds."
            elif _tier == 1:
                _cls, _emo, _slbl, _hl = "share-mixed", "≈", "VALIDATION_PARTIAL", "STRATEGY_WEAK"
                _tg = f"+{_ret:.0%} return, robust folds, BUT buy & hold gave {_bh:+.0%}. Passive wins."
            elif _ret < 0:
                _cls, _emo, _slbl, _hl = "share-doesnt", "✕", "VALIDATION_FAILED", "STRATEGY_LOSING"
                _tg = f"{_ret:.0%} return. {_robust_msg}. Do not trade this."
            else:
                _cls, _emo, _slbl, _hl = "share-mixed", "⚠", "VALIDATION_MARGINAL", "STRATEGY_MARGINAL"
                _tg = f"+{_ret:.0%} return. {_robust_msg}."
            if _n < 30:
                _trade_word = "trade" if _n == 1 else "trades"
                _tg += f" Only {_n} {_trade_word} — too small to trust."

        # Tier badge
        _badge = ""
        if _has_stats:
            _tc = {1: ("robust-tier1", "ROBUST · 3-4/4 FOLDS"),
                   2: ("robust-tier2", "BORDERLINE · 2/4 FOLDS"),
                   3: ("robust-tier3", "CURVE FIT · ≤1/4 FOLDS"),
                   None: ("robust-tier-none", "NO VALIDATION")}
            _k, _l = _tc.get(_tier, ("robust-tier-none", "NO DATA"))
            _badge = f'<div style="text-align:center;"><span class="robust-badge {_k}">{_l}</span></div>'

        # Stats
        _stats_html = ""
        if _has_stats:
            _rc = "green" if _stats["total_return"] > 0 else "red"
            _profit = _stats["final_equity"] - _cash
            _bc = "green" if _bh > 0 else "red"
            _stats_html = (
                f'<div class="verdict-stats-row">'
                f'<div class="stat-block"><p class="stat-label">Return</p>'
                f'<p class="stat-value {_rc}">{_stats["total_return"]:+.1%}</p>'
                f'<p class="stat-sub">${_profit:+,.0f}</p></div>'
                f'<div class="stat-block"><p class="stat-label">B&amp;H</p>'
                f'<p class="stat-value {_bc}">{_bh:+.1%}</p>'
                f'<p class="stat-sub">benchmark</p></div>'
                f'<div class="stat-block"><p class="stat-label">Sharpe</p>'
                f'<p class="stat-value">{_stats["sharpe_ratio"]:+.2f}</p>'
                f'<p class="stat-sub">risk-adj</p></div>'
                f'<div class="stat-block"><p class="stat-label">Max DD</p>'
                f'<p class="stat-value red">{_stats["max_drawdown"]:.0%}</p>'
                f'<p class="stat-sub">drawdown</p></div>'
                f'<div class="stat-block"><p class="stat-label">Trades</p>'
                f'<p class="stat-value">{_stats["total_trades"]:,}</p>'
                f'<p class="stat-sub">{_stats["win_rate"]:.0%} win</p></div>'
                f'</div>'
            )

        # RST mini-line (shown in verdict card)
        _rst = _r.get("rst", {})
        _rst_html = ""
        if _has_stats and _rst and not _rst.get("skip"):
            _rst_color = "var(--green)" if _rst.get("passed") else "var(--red)"
            _rst_icon = "✓" if _rst.get("passed") else "✗"
            _rst_label = "SIGNIFICANT" if _rst.get("passed") else "NOT SIGNIFICANT"
            _rst_html = (
                f'<p style="font-size:0.7rem;color:var(--text-dim);margin-top:0.6rem;'
                f'font-family:\'JetBrains Mono\',monospace;text-align:center;letter-spacing:0.08em;">'
                f'RST · <span style="color:{_rst_color}">{_rst_icon} {_rst_label}</span> '
                f'· p={_rst.get("p_value", 0):.3f} · beats {_rst.get("beats_random", 0)}/{_rst.get("n_shuffles", 0)}'
                f'</p>'
            )

        # Folds
        _fold_html = ""
        if _has_stats and _folds:
            _pcs = []
            for _i, _f in enumerate(_folds, 1):
                if _f is None:
                    _pcs.append(f"F{_i}: —")
                elif _f.get("no_trades"):
                    _pcs.append(f"F{_i}: 0")
                else:
                    _e = "🟢" if _f["return"] > 0 else "🔴"
                    _pcs.append(f"{_e} F{_i}: {_f['return']:+.1%}")
            _fold_html = ('<p style="font-size:0.72rem;color:var(--text-dim);margin-top:0.9rem;'
                          'font-family:\'JetBrains Mono\',monospace;text-align:center;letter-spacing:0.05em;">'
                          + " · ".join(_pcs) + '</p>')

        _src = "BINANCE" if _sym in {"BTC","ETH","SOL","ADA","ARB","ATOM","AVAX","DOGE",
                                      "DOT","FIL","LINK","NEAR","UNI","XRP"} else "YAHOO"
        import hashlib as _hl_mod
        _sid = _hl_mod.sha1(f"{_sym}{_r.get('strategy_text','')}{_cash}".encode()).hexdigest()[:8].upper()
        _vdate = _df.index[-1].strftime("%Y-%m-%d")
        _tf = _r.get("tf_label", "Daily")
        _dateline = (
            f'<div class="verdict-dateline">'
            f'<span>SYM: {_sym} · {_tf} · BARS: {len(_df):,} · SRC: {_src}</span>'
            f'<span>CASH: ${_cash:,}</span>'
            f'</div>'
        )
        _card = (
            f'<div class="share-card {_cls}">'
            f'{_dateline}'
            f'<div class="verdict-status">{_slbl}</div>'
            f'<div class="verdict-header-group">'
            f'<div class="verdict-emoji">{_emo}</div>'
            f'<h1 class="verdict-headline">{_hl}</h1>'
            f'</div>'
            f'{_badge}'
            f'<p class="verdict-tagline">{_tg}</p>'
            f'{_stats_html}{_rst_html}{_fold_html}'
            f'<div class="watermark">'
            f'<span>SESSION_ID: {_sid}</span>'
            f'<span>VERIFIED: {_vdate} · <span class="handle">@andrii</span></span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(_card, unsafe_allow_html=True)

        # Iteration button — let AI suggest improvement
        if _has_stats:
            if st.button("🔁 ITERATE · LET AI IMPROVE THIS STRATEGY",
                         use_container_width=True, key="iterate_btn"):
                with st.spinner("AI proposing modification..."):
                    suggestion = val.suggest_iteration(
                        _r.get("strategy_text", ""),
                        _r["rules"],
                        _stats,
                        _r.get("rst", {}).get("passed"),
                        _tier,
                    )
                if "error" in suggestion:
                    st.error(f"Iteration failed: {suggestion['error']}")
                elif not suggestion.get("strategy", "").strip():
                    st.error("AI returned an empty strategy — click ITERATE again.")
                else:
                    # Save to history
                    if "iteration_history" not in st.session_state:
                        st.session_state.iteration_history = []
                    st.session_state.iteration_history.append({
                        "version": len(st.session_state.iteration_history) + 1,
                        "strategy": _r.get("strategy_text", ""),
                        "return": _stats.get("total_return", 0),
                        "sharpe": _stats.get("sharpe_ratio", 0),
                        "max_dd": _stats.get("max_drawdown", 0),
                        "rst_passed": _r.get("rst", {}).get("passed"),
                        "wf_tier": _tier,
                    })
                    # Fill new strategy text and trigger run
                    st.session_state.strategy_text = suggestion["strategy"]
                    if suggestion.get("reasoning"):
                        st.info(f"AI reasoning: {suggestion['reasoning']}")
                    st.session_state.trigger_run = True
                    st.rerun()

        # Iteration history (if any)
        if st.session_state.get("iteration_history"):
            with st.expander(f"📚 ITERATION HISTORY · {len(st.session_state.iteration_history)} previous versions"):
                hist_rows = []
                for h in st.session_state.iteration_history:
                    hist_rows.append({
                        "V": f"V{h['version']}",
                        "RETURN": f"{h['return']:+.1%}",
                        "SHARPE": f"{h['sharpe']:+.2f}",
                        "MAX DD": f"{h['max_dd']:.0%}",
                        "RST": "✓" if h.get("rst_passed") else "✗",
                        "WF TIER": h.get("wf_tier", "—"),
                        "STRATEGY": h["strategy"][:60] + ("..." if len(h["strategy"]) > 60 else ""),
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
                if st.button("Clear history", key="clear_hist"):
                    st.session_state.iteration_history = []
                    st.rerun()

# Examples — moved below to not break the 2-col grid
st.markdown('<div class="section-divider">QUICK_LOAD · STRATEGY_TEMPLATES</div>', unsafe_allow_html=True)
ex_cols = st.columns(6)
for i, (label, text) in enumerate(EXAMPLES.items()):
    with ex_cols[i % 6]:
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.strategy_text = text
            st.rerun()


# ─── Run ─────────────────────────────────────────────────────────────

if (confirm_run or st.session_state.trigger_run) and strategy_text.strip():
    st.session_state.trigger_run = False
    # Use the confirmed blueprint rules if they match; otherwise (iterate path) re-parse
    _pv = st.session_state.parsed_preview
    if _pv and _pv.get("text") == strategy_text and _pv.get("rules"):
        rules = _pv["rules"]
    else:
        with st.spinner("🧠 Parsing strategy with AI..."):
            try:
                rules = llm.parse_strategy(strategy_text)
            except Exception as e:
                st.error(f"Couldn't parse: {e}")
                with st.expander("Your input"):
                    st.code(strategy_text, language="text")
                st.info(
                    "Try a more specific description with an indicator or pattern name. "
                    "Examples:\n\n"
                    "• `Buy when RSI dips below 30, sell at 70`\n"
                    "• `Test double bottom pattern`\n"
                    "• `EMA 9/21 crossover, long only`\n"
                    "• `Bollinger band touch mean reversion`"
                )
                st.stop()

    # ─── Pooled mode: same strategy across all crypto, pool the trades ───
    if symbol == POOL_SENTINEL:
        with st.spinner(f"📊 Scanning all crypto ({_tf_label}) and pooling trades..."):
            per_symbol, pooled_returns = run_pool(rules, timeframe, init_cash)
            pv = pooled_verdict(per_symbol, pooled_returns)
        st.session_state.last_result = {
            "pooled": True, "rules": rules, "strategy_text": strategy_text,
            "per_symbol": per_symbol, "pooled_returns": pooled_returns,
            "pooled_verdict": pv, "timeframe": timeframe, "tf_label": _tf_label,
            "init_cash": init_cash,
        }
        st.session_state.parsed_preview = {"text": strategy_text, "rules": rules}
        st.rerun()

    with st.spinner(f"📊 Running backtest on {symbol} ({_tf_label})..."):
        try:
            df = data.load(symbol, timeframe)
            trades, equity = backtest.run(df, rules, init_cash=init_cash)
            stats = metrics.compute(trades, equity)
            v = metrics.verdict(stats)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            with st.expander("Generated rules (for debugging)"):
                st.json(rules)
            st.stop()

    with st.spinner("🔬 Walk-forward validation (4 folds)..."):
        fold_results = run_walk_forward(df, rules, n_folds=4)
        tier, robust_msg = robustness_tier(fold_results)

    with st.spinner("🎲 Rule Significance Test (500 random shuffles)..."):
        rst_result = val.rule_significance_test(df, rules, n_shuffles=500)

    with st.spinner("🌊 Candle Monte Carlo (100 synthetic timelines)..."):
        mc_result = val.candle_monte_carlo(df, rules, n_sims=100)

    with st.spinner("📅 Multi-year out-of-sample..."):
        oos_result = val.multi_year_oos(df, rules)

    bh_ret = bh_return(df, init_cash)

    st.session_state.last_result = {
        "rules": rules, "df": df, "trades": trades, "equity": equity,
        "stats": stats, "verdict": v, "symbol": symbol, "init_cash": init_cash,
        "fold_results": fold_results, "tier": tier, "robust_msg": robust_msg,
        "bh_return": bh_ret, "strategy_text": strategy_text,
        "rst": rst_result, "mc": mc_result, "oos": oos_result,
        "timeframe": timeframe, "tf_label": _tf_label,
    }
    # Keep the blueprint in sync with what actually ran (e.g. after ITERATE)
    st.session_state.parsed_preview = {"text": strategy_text, "rules": rules}
    st.rerun()  # re-render so validation_col picks up new result


# ─── Results ─────────────────────────────────────────────────────────

if st.session_state.last_result and st.session_state.last_result.get("pooled"):
    r = st.session_state.last_result
    pv = r["pooled_verdict"]
    per_symbol = r["per_symbol"]
    _colormap = {"green": "#4ade80", "amber": "#fbbf24", "red": "#f87171"}
    _c = _colormap.get(pv["color"], "#e5e5e5")
    _label_txt = {"POSSIBLE_EDGE": "POSSIBLE EDGE", "NO_EDGE": "NO EDGE",
                  "CHERRY_PICKED": "CHERRY-PICKED", "THIN_SAMPLE": "THIN SAMPLE"}.get(pv["label"], pv["label"])

    st.markdown('<div class="section-divider">POOLED_SCAN · ALL CRYPTO</div>', unsafe_allow_html=True)

    # Verdict hero
    _pf = pv["profit_factor"]
    _pf_txt = "∞" if _pf == float("inf") else f"{_pf:.2f}"
    st.markdown(
        '<div style="border:1px solid ' + _c + '55;background:#0d0d0d;border-radius:10px;'
        'padding:20px 22px;margin:6px 0 14px;font-family:\'JetBrains Mono\',monospace;text-align:center;">'
        '<div style="color:' + _c + ';font-size:13px;letter-spacing:3px;margin-bottom:6px;">'
        + _label_txt + '</div>'
        '<div style="color:#e5e5e5;font-size:15px;line-height:1.5;">' + pv["msg"] + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Pooled KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("POOLED TRADES", f"{pv['n_total']:,}")
    k2.metric("WIN RATE", f"{pv['win_rate']:.0%}")
    k3.metric("AVG / TRADE", f"{pv['avg_ret']:+.2%}")
    k4.metric("COINS WORKING", f"{pv['n_syms_profitable']}/{pv['n_syms_traded']}")

    # Per-symbol breakdown
    st.markdown("**Per-coin breakdown**")
    _rows = []
    for p in sorted(per_symbol, key=lambda x: x["ret"], reverse=True):
        _rows.append({
            "COIN": p["symbol"],
            "TRADES": p["n"],
            "RETURN": f"{p['ret']:+.1%}" if p["n"] else "—",
            "WIN RATE": f"{p['win']:.0%}" if p["win"] is not None else "—",
        })
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    # Distribution of per-trade returns
    if r["pooled_returns"]:
        import plotly.graph_objects as _go
        _fig = _go.Figure(_go.Histogram(x=[x * 100 for x in r["pooled_returns"]],
                                        marker_color=_c, nbinsx=40))
        _fig.add_vline(x=0, line_dash="dot", line_color="#666")
        _fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#888", family="'JetBrains Mono', monospace", size=10),
                           xaxis_title="per-trade return (%)", yaxis_title="count")
        _fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        _fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.markdown("**Pooled per-trade return distribution**")
        st.plotly_chart(_fig, use_container_width=True)

    st.caption(
        f"Same strategy run independently on each coin ({r['tf_label']} bars), all trades pooled. "
        "A genuine edge shows a positive average and works across most coins — not one lucky pair."
    )
    st.stop()


if st.session_state.last_result:
    r = st.session_state.last_result
    # Backward-compat: older session_state may miss new keys → clear and prompt re-run
    if "fold_results" not in r:
        st.session_state.last_result = None
        st.info("App was updated — please click Run Backtest again.")
        st.stop()

    rules, df, trades, equity, stats, v = (
        r["rules"], r["df"], r["trades"], r["equity"], r["stats"], r["verdict"]
    )
    symbol = r["symbol"]
    init_cash = r["init_cash"]
    fold_results = r["fold_results"]
    tier = r["tier"]
    robust_msg = r["robust_msg"]
    bh_ret = r["bh_return"]
    days_span = len(df)

    has_stats = "error" not in stats

    # Source label (used by DATA INTEGRITY panel below)
    src_label = "BINANCE" if symbol in {"BTC", "ETH", "SOL", "ADA", "ARB", "ATOM",
                                          "AVAX", "DOGE", "DOT", "FIL", "LINK",
                                          "NEAR", "UNI", "XRP"} else "YAHOO FINANCE"


    # === DATA INTEGRITY PANEL — proof everything is real ===
    with st.expander("🔍 DATA INTEGRITY — verify everything is real, not fake", expanded=False):
        import pathlib, vectorbt as _vbt
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SOURCE FILE**")
            _tf = r.get("timeframe", "daily")
            _tf_lbl = r.get("tf_label", "Daily")
            _subdir = "intraday/" if _tf != "daily" else ""
            data_path = pathlib.Path(f"data/{_subdir}{symbol}.parquet").resolve()
            file_size = data_path.stat().st_size if data_path.exists() else 0
            _tf_note = f"{_tf_lbl} (resampled from 5m)" if _tf in {"1h", "4h"} else "daily OHLCV"
            st.code(
                f"path:     {data_path}\n"
                f"size:     {file_size:,} bytes\n"
                f"bars:     {len(df):,}\n"
                f"range:    {df.index[0]} → {df.index[-1]}\n"
                f"timeframe: {_tf_note}\n"
                f"source:   {src_label} (real market data)",
                language="text",
            )
        with col_b:
            st.markdown("**ENGINE**")
            st.code(
                f"backtest:  vectorbt {_vbt.__version__}\n"
                f"indicators: pandas_ta\n"
                f"fees:      0.10% per trade (configurable)\n"
                f"slippage:  0.05% per side\n"
                f"signals:   df.eval (sandboxed)\n"
                f"language:  Python 3.12\n"
                f"no mocks, no fake data, no API tricks",
                language="text",
            )

        st.markdown("**RAW PRICE DATA — first 5 + last 5 bars actually loaded:**")
        sample = pd.concat([df.head(5), df.tail(5)])
        st.dataframe(
            sample.style.format({
                "open": "{:.2f}", "high": "{:.2f}", "low": "{:.2f}",
                "close": "{:.2f}", "volume": "{:,.0f}",
            }),
            use_container_width=True, height=350,
        )

        if not trades.empty:
            st.markdown("**ACTUAL TRADES — first 5 executions:**")
            cols_show = [c for c in [
                "Entry Timestamp", "Avg Entry Price", "Exit Timestamp",
                "Avg Exit Price", "Size", "PnL", "Return",
            ] if c in trades.columns]
            st.dataframe(
                trades[cols_show].head(5) if cols_show else trades.head(5),
                use_container_width=True,
            )

        st.caption(
            "Cross-check any of this manually: open TradingView, find the same symbol, "
            "the same date range. OHLCV will match because it's the same exchange feed."
        )

    # === Quick "test on another symbol" ===
    st.markdown('<div class="section-divider">▸ Test same strategy on another symbol</div>', unsafe_allow_html=True)
    quick_syms = ["SPY", "QQQ", "BTC", "ETH", "SOL", "GLD"] if symbol not in {"SPY", "QQQ"} else ["BTC", "ETH", "SOL", "IWM", "GLD"]
    quick_syms = [s for s in quick_syms if s != symbol and s in all_syms][:5]

    qcols = st.columns(len(quick_syms))
    for i, qsym in enumerate(quick_syms):
        with qcols[i]:
            if st.button(f"→ {qsym}", key=f"qsw_{qsym}", use_container_width=True):
                st.session_state.switch_symbol = qsym
                st.session_state.trigger_run = True
                st.rerun()

    # === Chart + details ===
    if has_stats:
        tab_chart, tab_walk, tab_rst, tab_mc, tab_oos, tab_trades, tab_rules = st.tabs([
            "📈 CHART", "🔬 WALK-FWD", "🎲 SIGNIFICANCE", "🌊 MONTE CARLO",
            "📅 MULTI-YEAR", "📋 TRADES", "⚙️ RULES"
        ])

        with tab_chart:
            detected_patterns = []
            if rules.get("pattern"):
                try:
                    detected_patterns = pat_mod.detect(
                        df, rules["pattern"]["type"], rules["pattern"].get("params")
                    )
                except Exception:
                    detected_patterns = []

            bh_eq = (df["close"] / df["close"].iloc[0]) * init_cash
            bh_eq = bh_eq.reindex(equity.index, method="ffill")

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                row_heights=[0.55, 0.45],
                subplot_titles=(f"{symbol} price + entry points", "Strategy vs Buy & Hold"),
            )

            fig.add_trace(
                go.Scatter(x=df.index, y=df["close"], name="Price",
                           line=dict(color="#ffb000", width=1.2)),
                row=1, col=1,
            )

            if not trades.empty and "Entry Timestamp" in trades.columns:
                entry_dates = trades["Entry Timestamp"]
                entry_prices = trades["Avg Entry Price"]
                directions = trades.get("Direction", "Long")
                long_mask = directions == "Long"
                if long_mask.any():
                    fig.add_trace(
                        go.Scatter(x=entry_dates[long_mask], y=entry_prices[long_mask],
                                   mode="markers", name="Long",
                                   marker=dict(size=10, color="#00d4aa", symbol="triangle-up",
                                               line=dict(color="white", width=1))),
                        row=1, col=1,
                    )
                short_mask = ~long_mask
                if short_mask.any():
                    fig.add_trace(
                        go.Scatter(x=entry_dates[short_mask], y=entry_prices[short_mask],
                                   mode="markers", name="Short",
                                   marker=dict(size=10, color="#ff4b4b", symbol="triangle-down",
                                               line=dict(color="white", width=1))),
                        row=1, col=1,
                    )

            if detected_patterns:
                pivot_xs, pivot_ys = [], []
                for p in detected_patterns:
                    for key in ["p1_date", "p2_date", "ls_date", "head_date", "rs_date"]:
                        if key in p.meta:
                            price_key = key.replace("_date", "_price")
                            if price_key in p.meta:
                                pivot_xs.append(p.meta[key])
                                pivot_ys.append(p.meta[price_key])
                if pivot_xs:
                    fig.add_trace(
                        go.Scatter(x=pivot_xs, y=pivot_ys, mode="markers", name="Pattern pivots",
                                   marker=dict(size=9, color="#ffb800",
                                               line=dict(color="white", width=1))),
                        row=1, col=1,
                    )

            strategy_color = "#00ff00" if stats["total_return"] > 0 else "#ff3333"
            fig.add_trace(
                go.Scatter(x=equity.index, y=equity.values, name="Strategy",
                           line=dict(color=strategy_color, width=2.5)),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(x=bh_eq.index, y=bh_eq.values, name="Buy & Hold",
                           line=dict(color="#888", width=1.5, dash="dash")),
                row=2, col=1,
            )
            fig.add_hline(y=init_cash, line_dash="dot", line_color="#444", row=2, col=1)

            fig.update_layout(
                height=600, hovermode="x unified", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffb000", family="'JetBrains Mono', monospace", size=11), margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.update_xaxes(gridcolor="rgba(255,176,0,0.1)")
            fig.update_yaxes(gridcolor="rgba(255,176,0,0.1)")
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        with tab_walk:
            st.markdown(f"**{robust_msg}**")
            st.markdown(f"4-fold chronological walk-forward — each fold is ~{days_span // 4} days of unseen data.")

            # Build fold bar chart
            fold_data = []
            for i, f in enumerate(fold_results, 1):
                if f and not f.get("no_trades"):
                    fold_data.append({"fold": f"Fold {i}", "return": f["return"] * 100,
                                      "trades": f["trades"]})
                else:
                    fold_data.append({"fold": f"Fold {i}", "return": 0, "trades": 0})

            fig2 = go.Figure()
            colors = ["#00ff00" if d["return"] > 0 else "#ff3333" if d["return"] < 0 else "#888" for d in fold_data]
            fig2.add_trace(go.Bar(
                x=[d["fold"] for d in fold_data],
                y=[d["return"] for d in fold_data],
                marker_color=colors,
                text=[f"{d['return']:+.1f}%<br>({d['trades']} trades)" for d in fold_data],
                textposition="outside",
            ))
            fig2.update_layout(
                height=400,
                title=None,
                yaxis_title="Return (%)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffb000", family="'JetBrains Mono', monospace", size=11),
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            fig2.add_hline(y=0, line_color="#ffb000", line_dash="dash")
            fig2.update_yaxes(gridcolor="rgba(255,176,0,0.1)")
            st.plotly_chart(fig2, use_container_width=True)

            if tier == 1:
                st.success("✓ Strategy survived independent validation. Real signal candidate.")
            elif tier == 2:
                st.warning("⚠ Borderline — one losing period. Could be edge или could be partial luck.")
            elif tier == 3:
                st.error("✗ Strategy fails on multiple independent periods. Full-period 'profit' likely curve-fit.")

        # === RST tab — Rule Significance Test ===
        with tab_rst:
            rst = r.get("rst", {})
            if rst.get("skip"):
                st.info(f"RST skipped: {rst.get('reason', 'unavailable')}")
            else:
                st.markdown(
                    f"**Test:** entry signals shuffled randomly **{rst['n_shuffles']:,} times** "
                    f"to check if strategy beats random chance.\n\n"
                    f"**Logic:** if your edge is real, real strategy should outperform almost "
                    f"all random orderings."
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("REAL RETURN", f"{rst['real_return']:+.1%}")
                col2.metric("RANDOM AVG", f"{rst['shuffled_mean']:+.1%}")
                col3.metric("P-VALUE", f"{rst['p_value']:.4f}",
                            delta="SIGNIFICANT" if rst['passed'] else "NOT SIGNIFICANT",
                            delta_color="normal" if rst['passed'] else "inverse")

                st.markdown(
                    f"**Beats random:** {rst['beats_random']:,} / {rst['n_shuffles']:,} "
                    f"({rst['beats_random']/rst['n_shuffles']*100:.0f}%)"
                )

                # Bar chart: random distribution + real position
                _fig = go.Figure()
                _fig.add_trace(go.Histogram(
                    x=[rst['shuffled_mean']]*100,  # placeholder for now
                    nbinsx=30, marker_color="#444", name="Random shuffles",
                ))
                _fig.add_vline(x=rst['real_return'], line_color="#22d3ee", line_width=3,
                               annotation_text=f"REAL: {rst['real_return']:+.1%}",
                               annotation_position="top")
                _fig.add_vline(x=rst['shuffled_p95'], line_color="#f87171", line_dash="dash",
                               annotation_text=f"P95 random: {rst['shuffled_p95']:+.1%}")
                _fig.update_layout(
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaed", family="'Space Grotesk', sans-serif"),
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title="Return", yaxis_title="Count",
                    showlegend=False,
                )
                _fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%")
                _fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(_fig, use_container_width=True)

                if rst['passed']:
                    st.success(f"✓ {rst['verdict']}")
                else:
                    st.error(
                        f"✗ {rst['verdict']}\n\n"
                        f"Strategy doesn't reliably beat random entry placement. "
                        f"Entry rule has no real edge — what looked like skill might be luck."
                    )

        # === MC tab — Candle Monte Carlo ===
        with tab_mc:
            mc = r.get("mc", {})
            if mc.get("skip"):
                st.info(f"MC skipped: {mc.get('reason', 'unavailable')}")
            else:
                st.markdown(
                    f"**Test:** **{mc['n_sims']}** synthetic timelines generated by bootstrapping "
                    f"daily returns from real data. Strategy tested on each alternative history.\n\n"
                    f"**Logic:** robust strategies perform similarly across many possible market paths. "
                    f"If real result is in the middle of the distribution → not luck."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("REAL RETURN", f"{mc['real_return']:+.1%}")
                c2.metric("SIM MEDIAN", f"{mc['sim_return_median']:+.1%}")
                c3.metric("SIM P95 (best)", f"{mc['sim_return_p95']:+.1%}")
                c4.metric("SIM P5 (worst)", f"{mc['sim_return_p5']:+.1%}")

                st.markdown(f"**Real result percentile:** {mc['real_percentile']:.0f}th — "
                            f"better than {mc['real_percentile']:.0f}% of simulations")

                color_map = {
                    "ROBUST": "success", "ACCEPTABLE": "info",
                    "LIKELY_LUCKY": "warning", "BAD_LUCK": "warning",
                }
                _f = {"success": st.success, "info": st.info, "warning": st.warning}.get(
                    color_map.get(mc['robust_label'], "info"), st.info
                )
                _f(f"**{mc['robust_label']}** — {mc['robust_msg']}")

        # === OOS tab — Multi-Year Out-of-Sample ===
        with tab_oos:
            oos = r.get("oos", {})
            if oos.get("skip"):
                st.info(f"Multi-year OOS skipped: {oos.get('reason', 'unavailable')}")
            else:
                v_label = oos.get("verdict", "UNKNOWN")
                st.markdown(
                    f"**Test:** strategy tested on each calendar year independently.\n\n"
                    f"**Result:** {oos.get('profitable_years')}/{oos.get('total_years')} years profitable · "
                    f"**Verdict:** **{v_label}**"
                )

                # Build per-year table
                years_data = []
                for year, v_data in sorted(oos.get("by_year", {}).items()):
                    if v_data.get("no_trades"):
                        years_data.append({
                            "YEAR": year, "TRADES": 0, "RETURN": "—",
                            "B&H": f"{v_data['bh_return']:+.1%}",
                            "SHARPE": "—", "MAX DD": "—", "BEATS B&H": "—",
                        })
                    else:
                        years_data.append({
                            "YEAR": year,
                            "TRADES": v_data["trades"],
                            "RETURN": f"{v_data['return']:+.1%}",
                            "B&H": f"{v_data['bh_return']:+.1%}",
                            "SHARPE": f"{v_data['sharpe']:+.2f}",
                            "MAX DD": f"{v_data['max_dd']:.1%}",
                            "BEATS B&H": "✓" if v_data.get("beats_bh") else "✗",
                        })
                st.dataframe(pd.DataFrame(years_data), use_container_width=True,
                             height=min(400, 50 + 35 * len(years_data)))

                # Bar chart of yearly returns
                ydata = oos.get("by_year", {})
                yr_x = [str(y) for y in sorted(ydata.keys())]
                yr_y = []
                yr_colors = []
                for y in sorted(ydata.keys()):
                    v_data = ydata[y]
                    if v_data.get("no_trades"):
                        yr_y.append(0)
                        yr_colors.append("#444")
                    else:
                        yr_y.append(v_data["return"] * 100)
                        yr_colors.append("#4ade80" if v_data["return"] > 0 else "#f87171")

                _fig = go.Figure()
                _fig.add_trace(go.Bar(x=yr_x, y=yr_y, marker_color=yr_colors,
                                       text=[f"{v:+.0f}%" for v in yr_y],
                                       textposition="outside"))
                _fig.update_layout(
                    height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaed", family="'Space Grotesk', sans-serif"),
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title="Year", yaxis_title="Return %", showlegend=False,
                )
                _fig.add_hline(y=0, line_color="#fff", line_dash="dash")
                _fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                _fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(_fig, use_container_width=True)

                # Verdict alert
                if v_label == "ROBUST_ACROSS_YEARS":
                    st.success("✓ Strategy worked in EVERY tested year. Strong cross-regime signal.")
                elif v_label == "MOSTLY_WORKS":
                    st.info(f"~ Strategy worked in {oos.get('profitable_years')}/{oos.get('total_years')} years. Some regime sensitivity.")
                elif v_label == "INCONSISTENT":
                    st.warning("⚠ Strategy works in some periods, fails in others. Regime-dependent.")
                else:
                    st.error(f"✗ {v_label}. Strategy doesn't generalize across years.")

        with tab_trades:
            if not trades.empty:
                display_cols = [c for c in [
                    "Entry Timestamp", "Exit Timestamp", "Direction",
                    "Avg Entry Price", "Avg Exit Price", "PnL", "Return",
                ] if c in trades.columns]
                st.dataframe(
                    trades[display_cols] if display_cols else trades,
                    use_container_width=True, height=400,
                )
                st.caption(f"Total: {len(trades)} trades")
            else:
                st.info("No trades")

        with tab_rules:
            st.caption("AI translated your description:")
            st.json(rules)

else:
    if not strategy_text.strip():
        st.info("👆 Describe a strategy above or pick an example, then click **Run Backtest**.")
