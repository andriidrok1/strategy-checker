"""Update all crypto parquets to latest data from Binance.

For each crypto symbol:
1. Read existing intraday (5m) parquet
2. Fetch new 5m bars from last_existing_bar+1 → now via Binance public API
3. Concat + dedupe, save updated intraday parquet
4. Resample intraday → daily, overwrite main daily parquet
5. Report new bar count and date range

Stock data (SPY/QQQ/IWM/GLD/TLT) is NOT updated — those are static historical.
"""
import time
from pathlib import Path
import pandas as pd
import ccxt

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INTRADAY_DIR = DATA_DIR / "intraday"

CRYPTOS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
    "ADA": "ADA/USDT", "ARB": "ARB/USDT", "ATOM": "ATOM/USDT",
    "AVAX": "AVAX/USDT", "DOGE": "DOGE/USDT", "DOT": "DOT/USDT",
    "FIL": "FIL/USDT", "LINK": "LINK/USDT", "NEAR": "NEAR/USDT",
    "UNI": "UNI/USDT", "XRP": "XRP/USDT",
}

BINANCE = ccxt.binance()


def fetch_5m_since(market: str, since_ms: int, label: str) -> pd.DataFrame:
    """Fetch all 5m bars from since_ms to now in chunks of 1000."""
    all_rows = []
    cur = since_ms
    now_ms = int(time.time() * 1000)
    limit = 1000

    while cur < now_ms:
        try:
            batch = BINANCE.fetch_ohlcv(market, timeframe="5m", since=cur, limit=limit)
        except Exception as e:
            print(f"    ⚠ {market}: {e}, stopping")
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cur:
            break
        cur = last_ts + 5 * 60 * 1000
        time.sleep(0.12)
        if len(all_rows) % 5000 == 0:
            last_dt = pd.to_datetime(last_ts, unit="ms")
            print(f"    {label}: fetched {len(all_rows):,} bars, at {last_dt}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df.index.name = "datetime"
    return df[~df.index.duplicated(keep="first")]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()


def update_symbol(sym: str, market: str):
    intraday_path = INTRADAY_DIR / f"{sym}.parquet"
    daily_path = DATA_DIR / f"{sym}.parquet"

    if not intraday_path.exists():
        print(f"  ⚠ {sym}: no existing intraday parquet, skip")
        return

    existing = pd.read_parquet(intraday_path)
    last_ts = existing.index[-1]
    since_dt = last_ts + pd.Timedelta(minutes=5)
    since_ms = int(since_dt.timestamp() * 1000)

    bars_to_fetch_days = (pd.Timestamp.utcnow().tz_localize(None) - since_dt).days
    print(f"  {sym} ({market}): {len(existing):,} existing, "
          f"last bar {last_ts}, fetching ~{bars_to_fetch_days}d ...")

    new_bars = fetch_5m_since(market, since_ms, sym)

    if new_bars.empty:
        print(f"    no new bars (data is current)")
        return

    print(f"    fetched {len(new_bars):,} new bars ({new_bars.index[0]} → {new_bars.index[-1]})")

    # Merge intraday
    merged = pd.concat([existing, new_bars])
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    merged.to_parquet(intraday_path)
    added = len(merged) - len(existing)
    print(f"    intraday: {len(existing):,} → {len(merged):,} bars (+{added:,})")

    # Resample → daily
    daily = to_daily(merged)
    daily.to_parquet(daily_path)
    print(f"    daily:    {len(daily):,} bars, range {daily.index[0].date()} → {daily.index[-1].date()}")


def main():
    print(f"Updating {len(CRYPTOS)} crypto symbols to current time ({pd.Timestamp.utcnow()})...\n")
    for sym, market in CRYPTOS.items():
        update_symbol(sym, market)
        print()

    # Cleanup obsolete _recent files
    for old in DATA_DIR.glob("*_recent.parquet"):
        print(f"  cleanup: rm {old.name}")
        old.unlink()

    print(f"\nDone. All crypto data current as of {pd.Timestamp.utcnow()}.")


if __name__ == "__main__":
    main()
