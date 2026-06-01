"""Fetch most recent crypto OHLCV from Binance public API.

Pulls everything FROM end of our existing parquets TO now.
Saves as data/{SYMBOL}_recent.parquet (separate from main files) for OOS testing.
"""
import time
from pathlib import Path
import pandas as pd
import ccxt

DATA_DIR = Path(__file__).parent / "data"

# Map our internal name → Binance market pair
SYMBOLS = {
    "BTC":  "BTC/USDT",
    "ETH":  "ETH/USDT",
    "SOL":  "SOL/USDT",
    "ADA":  "ADA/USDT",
    "ARB":  "ARB/USDT",
    "ATOM": "ATOM/USDT",
    "AVAX": "AVAX/USDT",
    "DOGE": "DOGE/USDT",
    "DOT":  "DOT/USDT",
    "FIL":  "FIL/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "UNI":  "UNI/USDT",
    "XRP":  "XRP/USDT",
}

BINANCE = ccxt.binance()


def fetch_5m_since(market: str, since_ms: int) -> pd.DataFrame:
    """Fetch all 5-minute OHLCV bars from since_ms to now."""
    all_rows = []
    cur = since_ms
    now_ms = int(time.time() * 1000)
    limit = 1000  # max per request

    while cur < now_ms:
        try:
            batch = BINANCE.fetch_ohlcv(market, timeframe="5m", since=cur, limit=limit)
        except Exception as e:
            print(f"    ⚠ {market}: {e}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cur:
            break  # no progress
        cur = last_ts + 5 * 60 * 1000
        time.sleep(0.15)  # avoid rate limit

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    return df


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()


def main():
    for internal, market in SYMBOLS.items():
        existing_path = DATA_DIR / f"{internal}.parquet"
        if not existing_path.exists():
            print(f"  ⚠ {internal}: no existing file, skip")
            continue
        existing = pd.read_parquet(existing_path)
        last_date = existing.index[-1]
        # Start fetching from day after last bar
        since_dt = last_date + pd.Timedelta(days=1)
        since_ms = int(since_dt.timestamp() * 1000)

        print(f"  {internal} ({market}): fetching 5m bars from {since_dt.date()} → now ...")
        raw = fetch_5m_since(market, since_ms)
        if raw.empty:
            print(f"    no new bars")
            continue

        daily = to_daily(raw)
        out_path = DATA_DIR / f"{internal}_recent.parquet"
        daily.to_parquet(out_path)
        print(f"    ✓ {len(daily)} new daily bars: {daily.index[0].date()} → {daily.index[-1].date()} → {out_path.name}")


if __name__ == "__main__":
    main()
