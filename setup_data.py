"""One-time data setup. Fetches OHLCV for all supported symbols.

- Stocks (SPY, QQQ, IWM, GLD, TLT): yfinance, daily, 12 years
- Crypto (BTC, ETH, SOL + 11 alts): Binance public API via ccxt, daily

After install:
    pip install -r requirements.txt
    python setup_data.py
    streamlit run app.py
"""
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

STOCKS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
STOCK_START = "2012-09-06"

CRYPTOS = {
    "BTC":  "BTC/USDT", "ETH": "ETH/USDT", "SOL":  "SOL/USDT",
    "ADA":  "ADA/USDT", "ARB": "ARB/USDT", "ATOM": "ATOM/USDT",
    "AVAX": "AVAX/USDT", "DOGE": "DOGE/USDT", "DOT": "DOT/USDT",
    "FIL":  "FIL/USDT", "LINK": "LINK/USDT", "NEAR": "NEAR/USDT",
    "UNI":  "UNI/USDT", "XRP":  "XRP/USDT",
}
# How many days of crypto history to fetch (Binance allows 1000 daily bars per call)
CRYPTO_LOOKBACK_DAYS = 730


def fetch_stocks():
    """Download daily OHLCV via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠ yfinance not installed. Run: pip install yfinance")
        return

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for sym in STOCKS:
        out = DATA_DIR / f"{sym}.parquet"
        if out.exists():
            print(f"  ↺ {sym}: exists, skip (delete file to re-fetch)")
            continue
        print(f"  → {sym}: fetching {STOCK_START} → {today} from Yahoo ...")
        df = yf.download(sym, start=STOCK_START, end=today, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"    ⚠ empty response")
            continue
        # Yahoo returns capitalized columns + sometimes MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index.name = "date"
        df.to_parquet(out)
        print(f"    ✓ {len(df)} bars  → {out.name}")


def fetch_cryptos():
    """Download daily OHLCV via ccxt (Binance public API)."""
    try:
        import ccxt
    except ImportError:
        print("  ⚠ ccxt not installed. Run: pip install ccxt")
        return

    binance = ccxt.binance()
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - CRYPTO_LOOKBACK_DAYS * 24 * 60 * 60 * 1000

    for sym, market in CRYPTOS.items():
        out = DATA_DIR / f"{sym}.parquet"
        if out.exists():
            print(f"  ↺ {sym}: exists, skip (delete file to re-fetch)")
            continue
        print(f"  → {sym} ({market}): fetching ~{CRYPTO_LOOKBACK_DAYS}d daily ...")
        all_bars = []
        cur = since_ms
        while cur < now_ms:
            try:
                batch = binance.fetch_ohlcv(market, timeframe="1d", since=cur, limit=1000)
            except Exception as e:
                print(f"    ⚠ {e}")
                break
            if not batch:
                break
            all_bars.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cur:
                break
            cur = last_ts + 24 * 60 * 60 * 1000
            time.sleep(0.15)

        if not all_bars:
            print(f"    ⚠ no data")
            continue
        df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="first")]
        df.to_parquet(out)
        print(f"    ✓ {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})  → {out.name}")


def main():
    print(f"=== Strategy Checker · data setup ===\n")
    print("Fetching stocks (Yahoo Finance)...")
    fetch_stocks()
    print("\nFetching crypto (Binance public API)...")
    fetch_cryptos()
    print(f"\nDone. Files in {DATA_DIR}/\n")
    print("Next: `streamlit run app.py`")


if __name__ == "__main__":
    main()
