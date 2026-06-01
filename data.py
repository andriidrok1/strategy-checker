"""Data loader. Reads normalized parquets from data/ (daily) or data/intraday/ (5m).

Intraday timeframes 1h / 4h are resampled on the fly from the stored 5m bars.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
INTRADAY_DIR = DATA_DIR / "intraday"

# Timeframes that are derived by resampling the stored 5m intraday bars
RESAMPLE_RULES = {"1h": "1h", "4h": "4h"}


def available_symbols(timeframe: str = "daily") -> list[str]:
    """Symbols available for a timeframe. daily → data/; intraday (5m/1h/4h) → intraday/."""
    folder = DATA_DIR if timeframe == "daily" else INTRADAY_DIR
    return sorted(p.stem for p in folder.glob("*.parquet") if "_recent" not in p.stem)


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser timeframe."""
    agg = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return agg


def load(symbol: str, timeframe: str = "daily") -> pd.DataFrame:
    """Load OHLCV for symbol. timeframe: 'daily', '5m', '1h', or '4h'."""
    if timeframe in RESAMPLE_RULES:
        base = load(symbol, "5m")
        return _resample(base, RESAMPLE_RULES[timeframe])

    folder = INTRADAY_DIR if timeframe == "5m" else DATA_DIR
    path = folder / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No {timeframe} data for {symbol}. Available: {available_symbols(timeframe)}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    print("Available symbols:", available_symbols())
    for sym in available_symbols():
        df = load(sym)
        print(f"  {sym}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}, "
              f"last close = {df['close'].iloc[-1]:.2f}")
