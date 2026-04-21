from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from src import config

CACHE_DIR = Path(os.environ.get("BACKTEST_CACHE_DIR", ".backtest_cache"))
_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str, start: date, end: date) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.parquet"


def _fetch_alpaca(ticker: str, start: date, end: date,
                   api_key: str, secret_key: str) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    resp = client.get_stock_bars(req)
    bars = resp[ticker]
    df = pd.DataFrame([
        {"open": float(b.open), "high": float(b.high), "low": float(b.low),
         "close": float(b.close), "volume": float(b.volume),
         "date": b.timestamp.date()}
        for b in bars
    ])
    df = df.set_index("date")
    return df[_COLUMNS]


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty DataFrame for {ticker}")
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = _COLUMNS
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"
    return df


def fetch_bars(
    ticker: str,
    start: date,
    end: date,
    api_key: str = "",
    secret_key: str = "",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for ticker between start and end (inclusive).
    Caches to parquet; uses Alpaca if credentials supplied, else yfinance.
    Raises ValueError if all sources fail.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker, start, end)

    if not force_refresh and cache.exists():
        return pd.read_parquet(cache)

    df: pd.DataFrame | None = None
    last_exc: Exception | None = None

    if api_key and secret_key:
        try:
            df = _fetch_alpaca(ticker, start, end, api_key, secret_key)
        except Exception as exc:
            last_exc = exc

    if df is None:
        try:
            df = _fetch_yfinance(ticker, start, end)
        except Exception as exc:
            last_exc = exc

    if df is None or df.empty:
        raise ValueError(f"All bar sources failed for {ticker}: {last_exc}")

    df = df[_COLUMNS]
    df.to_parquet(cache)
    return df
