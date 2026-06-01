# Strategy Checker

> Describe a trading strategy in plain English. Get a code-tested verdict with proper validation.

A web tool that takes natural-language trading strategy descriptions, parses them into executable rules via AI, and runs rigorous backtests with statistical validation — designed to expose curve-fit strategies and trading bullshit, not generate alpha.

**Built by [@andriidrok1](https://github.com/andriidrok1) — 17 yo, Ukrainian, currently learning quant.**

---

## Why this exists

Most trading content on YouTube / Twitter / TikTok shows backtests without proper validation. A strategy "with 95% win rate on SPY 2024" might just be lucky on one period.

This tool runs **five layers of validation automatically:**

1. **Backtest** with realistic fees + slippage (vectorbt engine)
2. **Walk-forward (4-fold)** — strategy must work in independent time periods
3. **Rule Significance Test (RST)** — shuffles entry signals 500 times, real strategy must beat random orderings
4. **Candle Monte Carlo** — bootstraps 100 synthetic price paths from real returns
5. **Multi-year out-of-sample** — tests each calendar year independently

Plus a **multi-coin pooled scan**: run one pattern across all crypto at once. A genuine edge survives across many coins — not one lucky pair.

If a strategy fails any of these, you'll see it.

---

## How it flows

1. **Type** a strategy in plain English and pick a symbol + timeframe (Daily / 4h / 1h).
2. **Analyze** — the AI parses it and shows a **Strategy Blueprint**: the exact rules, direction, entry/exit triggers, stops, and data window that will be tested. No black box — you confirm what runs.
3. **Confirm** — the engine runs the backtest + all validation layers and returns a verdict card.

---

## What you can do with it

- Test classic strategies: golden cross, RSI mean reversion, Bollinger band touches, EMA crossovers
- Test chart patterns: double top/bottom, head and shoulders, ascending/descending/symmetrical triangles, bullish/bearish flags
- Run on **daily, 4-hour, or hourly** bars (intraday gives far larger trade samples)
- **Pool a pattern across all 14 crypto symbols** for a statistically meaningful sample + a "real edge or cherry-picked?" verdict
- Compare strategy performance against a buy-and-hold benchmark
- Iterate strategies with AI-suggested improvements
- Generate screenshot-ready verdict cards for content

---

## Tech stack

- **Backend:** Python · pandas · numpy · vectorbt · pandas-ta · scipy
- **UI:** Streamlit
- **AI parsing:** OpenAI API (gpt-4o-mini for cost efficiency)
- **Data:** Yahoo Finance (stocks) · Binance public API (crypto)

---

## Quick start

### 1. Clone + install

```bash
git clone https://github.com/andriidrok1/strategy-checker.git
cd strategy-checker
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Open .env and paste your OpenAI key
```

You can get a key at https://platform.openai.com/api-keys. Costs are minimal (gpt-4o-mini, ~$0.001 per strategy parse).

### 3. Fetch market data (one-time, ~2 minutes)

```bash
python setup_data.py
```

This downloads:
- 5 US ETFs (SPY, QQQ, IWM, GLD, TLT) from Yahoo Finance — 12 years of daily bars
- 14 crypto pairs from Binance public API — 2 years of daily bars
- 14 crypto pairs — 2 years of **hourly (1h)** bars into `data/intraday/` (the 4h and 1h app modes resample from this; pooled scan uses it too)

No API keys required for data fetching. Public market data only. The intraday fetch takes the longest (a few minutes).

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

### 5. Keep data fresh (optional)

```bash
python update_data.py
```

Incrementally updates crypto data to current time. Stocks remain at historical snapshot.

---

## Available symbols & timeframes

**Stocks (12 years, daily only):** SPY, QQQ, IWM, GLD, TLT

**Crypto (2 years; daily + intraday):** BTC, ETH, SOL, ADA, ARB, ATOM, AVAX, DOGE, DOT, FIL, LINK, NEAR, UNI, XRP

**Timeframes:** `Daily` · `4h` · `1h`. Intraday (4h/1h) is crypto-only and is resampled on the fly from the stored hourly bars — so daily BTC gives ~775 bars while 4h gives ~4,600 and 1h ~18,500. More bars → more trades → a real sample.

**Pooled scan:** pick `★ ALL CRYPTO (pool)` as the symbol to run one strategy across every crypto pair and pool the trades.

To add more symbols, edit `setup_data.py` and re-run.

---

## Architecture

```
strategy-checker/
├── app.py              # Streamlit UI: blueprint, single-symbol + pooled scan, result rendering
├── data.py             # Parquet loader
├── backtest.py         # vectorbt engine + indicator/expression evaluation
├── metrics.py          # Sharpe, Sortino, drawdown, etc.
├── patterns.py         # 9 chart pattern detectors (pivot-based geometry)
├── validation.py       # RST, Monte Carlo, multi-year OOS, iteration
├── llm.py              # OpenAI NL → strategy JSON
├── setup_data.py       # One-time data fetch (yfinance + ccxt)
├── update_data.py      # Incremental crypto data refresh
└── scripts/            # Research / exploration scripts (not for app users)
```

---

## How strategy parsing works

You type something like:

> "Buy when RSI(14) drops below 30 and price is above 200 SMA. Sell when RSI > 70. 5% stop loss."

The AI returns structured JSON:

```json
{
  "indicators": [
    {"name": "rsi_14", "type": "rsi", "period": 14},
    {"name": "sma_200", "type": "sma", "period": 200}
  ],
  "entry_long": "rsi_14 < 30 and close > sma_200",
  "exit_long": "rsi_14 > 70",
  "stop_loss_pct": 0.05
}
```

Before anything runs, the parsed rules are shown back to you as a **Strategy Blueprint** (direction, entry/exit triggers, stops, data window) so you can confirm the AI understood you correctly. Only after you confirm does `backtest.py` compute indicators via `pandas-ta`, evaluate expressions in a sandboxed `df.eval()`, and run the portfolio simulation with vectorbt.

For chart patterns, the parsing returns `{"pattern": {"type": "double_top"}}` and `patterns.py` detects pivots geometrically using `scipy.signal.find_peaks` + custom rules.

---

## Supported indicators

`rsi`, `sma`, `ema`, `atr`, `macd`, `stoch`, `bbands`, `adx`

## Supported chart patterns

`double_top`, `double_bottom`, `head_and_shoulders`, `inverse_head_and_shoulders`, `ascending_triangle`, `descending_triangle`, `symmetrical_triangle`, `bullish_flag`, `bearish_flag`

---

## Validation methodology

### Rule Significance Test (RST)

Shuffles entry signals randomly N times (default 500). For each shuffle, runs the same backtest with random entry timing. If the real strategy beats most random shuffles, p-value < 0.05 = significant edge. If not, your "edge" may just be luck.

This is the most ruthless check — many "profitable" strategies fail RST.

### Walk-forward (4-fold)

Splits data into 4 chronological folds. Strategy tested on each independently. Robust signal = profitable in all 4. Curve-fit = profitable in only 1.

### Candle Monte Carlo

Bootstraps log-returns from real data to generate 100 synthetic OHLC price paths. Runs strategy on each. Real result should be in the middle 50% of simulations — not in extreme tails (which indicates path-dependent luck).

### Multi-year OOS

Splits data by calendar year. Tests strategy on each year independently. Reveals regime breakdowns: many strategies work in 2024 (bull market) but blow up in 2022 (bear).

### Multi-coin pooled scan

Pick `★ ALL CRYPTO (pool)` as the symbol. The same strategy runs independently on every crypto pair and all trades are pooled into one sample. The verdict is based on **breadth, not luck**:

- average return per trade ≤ 0 → **NO EDGE** (the pattern loses on average)
- positive but profitable on < half the coins → **CHERRY-PICKED** (worked on a few, failed on most)
- positive *and* profitable across most coins with ≥ 30 trades → **POSSIBLE EDGE**

This is the most communicable check: a real edge shows up across many markets, not one.

### Iteration loop

After seeing results, click "ITERATE" to have the AI propose ONE specific improvement (tighter threshold, add filter, etc). Auto-fills new strategy text and re-runs.

---

## Limitations

- **Indicator-based and pattern-based only** — discretionary "smart money" strategies can't be tested
- **`df.eval` doesn't support transition events** — "EMA9 crosses above EMA21" is interpreted as regime (`>`) not single bar transition
- **No live trading** — backtest only
- **One timeframe per strategy** — you pick Daily/4h/1h, but a single strategy can't mix timeframes (no MTF rules)
- **Intraday is crypto-only** — stocks are daily-bar only
- **Generic fees (0.10%) + slippage (0.05%)** — not per-market specific

---

## Disclaimer

**This is not financial advice. Backtest results do not predict future performance. Real markets include execution costs, latency, funding rates, liquidations, slippage, and behavior changes that backtests cannot fully model.**

Past performance is not indicative of future results. Trading carries substantial risk of loss. The author is not a licensed financial advisor and accepts no liability for decisions made based on this tool's output.

This software is provided AS-IS under the MIT license. Use it to learn, to explore, to challenge claims you see online — not to risk capital you can't afford to lose.

---

## Contributing

Bug reports, feature ideas, and pull requests welcome. Open an issue first for larger changes.

---

## License

MIT. See [LICENSE](LICENSE).
