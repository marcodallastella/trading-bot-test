from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pct: float = field(init=False)
    market_value: float = field(init=False)

    def __post_init__(self):
        if self.avg_entry_price > 0:
            self.unrealized_pct = round(
                (self.current_price - self.avg_entry_price) / self.avg_entry_price, 4
            )
        else:
            self.unrealized_pct = 0.0
        self.market_value = round(self.qty * self.current_price, 4)


@dataclass
class WashSaleEntry:
    sold_date: str    # "YYYY-MM-DD"
    expires: str      # "YYYY-MM-DD"


@dataclass
class TrailingStop:
    entry_price: float
    peak_price: float
    activated: str    # "YYYY-MM-DD"


@dataclass
class State:
    rolling_peak_equity: float
    rolling_peak_timestamp: str
    last_run_timestamp: str
    ticker_last_buy_date: dict[str, str]
    wash_sale_blacklist: dict[str, WashSaleEntry]
    trailing_stops_active: dict[str, TrailingStop]
    cumulative_realized_pnl: float
    drawdown_halt_active: bool


@dataclass
class BuySignal:
    ticker: str
    price: float
    pir: float
    signal_mult: float
    sma_50: float
    atr_pct: float
    vol_factor: float
    is_deep_dip: bool


@dataclass
class SellDecision:
    ticker: str
    action: str   # "sell_stop_loss" | "sell_trailing" | "activate_trailing" | "hold"
    unrealized_pct: float
    reason: str


@dataclass
class RegimeInfo:
    spy_price: float
    sma_200: float
    slope_20d_pct: float
    above_sma: bool
    slope_direction: str   # "rising" | "flat" | "falling" | "flat_or_falling"
    regime_mult: float
    description: str


@dataclass
class SizingResult:
    ticker: str
    kelly_base: float
    signal_mult: float
    vol_factor: float
    regime_mult: float
    position_size_pct: float
    buy_amount_usd: float
    capped: bool
    cap_reason: str
    skip: bool
    skip_reason: str


@dataclass
class LogbookRow:
    timestamp: str
    ticker: str
    price: str
    sma_50: str
    pir: str
    atr_pct: str
    signal_mult: str
    vol_factor: str
    regime_mult: str
    kelly_base: str
    final_pct: str
    action: str
    order_notional: str
    fill_price: str
    reason: str
