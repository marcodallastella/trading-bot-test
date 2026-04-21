# src/alpaca_client.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from src import config
from src.logger import get_logger
from src.models import Position

log = get_logger(__name__)


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        base_url = config.ALPACA_BASE_URL_PAPER if paper else config.ALPACA_BASE_URL_LIVE
        self._trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            url_override=base_url,
        )
        self._data = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    def get_clock(self) -> dict:
        try:
            clock = self._trading.get_clock()
            return {
                "is_open": bool(clock.is_open),
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }
        except Exception as exc:
            log.error(f"get_clock error: {exc}")
            return {"is_open": False, "next_open": "", "next_close": ""}

    def get_account(self) -> dict:
        try:
            account = self._trading.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
            }
        except Exception as exc:
            log.error(f"get_account error: {exc}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0}

    def get_positions(self) -> list[Position]:
        try:
            raw = self._trading.get_all_positions()
            return [
                Position(
                    ticker=str(p.symbol),
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                )
                for p in raw
            ]
        except Exception as exc:
            log.error(f"get_positions error: {exc}")
            return []

    def get_quote(self, ticker: str) -> dict:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            resp = self._data.get_stock_latest_quote(req)
            quote = resp[ticker]
            return {
                "ask": float(quote.ask_price),
                "bid": float(quote.bid_price),
                "last": float(quote.ask_price),
            }
        except Exception as exc:
            log.error(f"get_quote({ticker}) error: {exc}")
            return {"ask": 0.0, "bid": 0.0, "last": 0.0}

    def get_bars(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        try:
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(days=lookback_days + 10)
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            resp = self._data.get_stock_bars(req)
            bars = resp[ticker]
            df = pd.DataFrame([
                {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "timestamp": b.timestamp,
                }
                for b in bars
            ])
            if df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df = df.set_index("timestamp").sort_index()
            df.index = pd.DatetimeIndex(df.index)
            return df.tail(lookback_days)
        except Exception as exc:
            log.error(f"get_bars({ticker}) error: {exc}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def submit_buy_order(self, ticker: str, notional: float,
                         time_in_force: str = "day") -> dict | None:
        try:
            tif = TimeInForce(time_in_force)
            req = MarketOrderRequest(
                symbol=ticker,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
            order = self._trading.submit_order(req)
            return {
                "id": str(order.id),
                "symbol": str(order.symbol),
                "notional": notional,
                "side": "buy",
                "status": str(order.status),
            }
        except Exception as exc:
            log.error(f"submit_buy_order({ticker}, {notional}) error: {exc}")
            return None

    def close_position(self, ticker: str) -> dict | None:
        try:
            order = self._trading.close_position(ticker)
            return {
                "id": str(order.id),
                "symbol": str(order.symbol),
                "side": "sell",
                "status": str(order.status),
            }
        except Exception as exc:
            log.error(f"close_position({ticker}) error: {exc}")
            return None
