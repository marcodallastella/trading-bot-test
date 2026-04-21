import pytest
from dataclasses import fields
from src.models import (
    Position,
    State,
    BuySignal,
    SellDecision,
    RegimeInfo,
    SizingResult,
    WashSaleEntry,
    TrailingStop,
    LogbookRow,
)


def test_position_instantiation():
    pos = Position(ticker="AAPL", qty=10.5, avg_entry_price=150.00, current_price=155.00)
    assert pos.ticker == "AAPL"
    assert pos.qty == 10.5
    assert pos.avg_entry_price == 150.00
    assert pos.current_price == 155.00
    assert pos.unrealized_pct == round((155.00 - 150.00) / 150.00, 4)


def test_state_instantiation():
    wash_entry = WashSaleEntry(sold_date="2024-01-01", expires="2024-02-01")
    trailing = TrailingStop(entry_price=100.0, peak_price=120.0, activated="2024-01-15")
    state = State(
        rolling_peak_equity=10000.0,
        rolling_peak_timestamp="2024-01-01T00:00:00Z",
        last_run_timestamp="2024-01-02T00:00:00Z",
        ticker_last_buy_date={"AAPL": "2024-01-01"},
        wash_sale_blacklist={"TSLA": wash_entry},
        trailing_stops_active={"NVDA": trailing},
        cumulative_realized_pnl=500.0,
        drawdown_halt_active=False,
    )
    assert state.rolling_peak_equity == 10000.0
    assert state.drawdown_halt_active is False


def test_buy_signal_fields():
    signal = BuySignal(
        ticker="GOOGL",
        price=182.0,
        pir=0.25,
        signal_mult=1.2,
        sma_50=180.0,
        atr_pct=0.03,
        vol_factor=1.0,
        is_deep_dip=False,
    )
    assert signal.ticker == "GOOGL"
    assert signal.pir == 0.25
    assert signal.signal_mult == 1.2
    assert signal.sma_50 == 180.0
    assert signal.atr_pct == 0.03
    assert signal.vol_factor == 1.0


def test_sell_decision_valid_actions():
    valid_actions = ["sell_stop_loss", "sell_trailing", "activate_trailing", "hold"]
    for action in valid_actions:
        decision = SellDecision(ticker="AAPL", action=action, unrealized_pct=-0.05, reason="test reason")
        assert decision.action == action


def test_regime_info_multiplier_is_float():
    info = RegimeInfo(
        spy_price=520.0,
        sma_200=495.0,
        slope_20d_pct=0.018,
        above_sma=True,
        slope_direction="rising",
        regime_mult=1.0,
        description="bull",
    )
    assert isinstance(info.regime_mult, float)


def test_sizing_result_skip_fields():
    result = SizingResult(
        ticker="MSFT",
        kelly_base=0.08,
        signal_mult=1.0,
        vol_factor=1.0,
        regime_mult=1.0,
        position_size_pct=0.08,
        buy_amount_usd=80.0,
        capped=False,
        cap_reason="",
        skip=True,
        skip_reason="below min notional",
    )
    assert result.skip is True
    assert result.skip_reason == "below min notional"


def test_wash_sale_entry_instantiation():
    entry = WashSaleEntry(sold_date="2024-03-01", expires="2024-04-01")
    assert entry.sold_date == "2024-03-01"
    assert entry.expires == "2024-04-01"


def test_trailing_stop_instantiation():
    ts = TrailingStop(entry_price=100.0, peak_price=130.0, activated="2024-02-10")
    assert ts.entry_price == 100.0
    assert ts.peak_price == 130.0
    assert ts.activated == "2024-02-10"


def test_logbook_row_has_15_fields():
    field_names = [f.name for f in fields(LogbookRow)]
    assert len(field_names) == 15
    expected = [
        "timestamp", "ticker", "price", "sma_50", "pir", "atr_pct",
        "signal_mult", "vol_factor", "regime_mult", "kelly_base",
        "final_pct", "action", "order_notional", "fill_price", "reason",
    ]
    assert field_names == expected


def test_state_wash_sale_blacklist_holds_dataclass_objects():
    entry = WashSaleEntry(sold_date="2024-01-01", expires="2024-02-01")
    state = State(
        rolling_peak_equity=1000.0,
        rolling_peak_timestamp="2024-01-01T00:00:00Z",
        last_run_timestamp="2024-01-01T00:00:00Z",
        ticker_last_buy_date={},
        wash_sale_blacklist={"TSLA": entry},
        trailing_stops_active={},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=False,
    )
    value = state.wash_sale_blacklist["TSLA"]
    assert isinstance(value, WashSaleEntry), f"Expected WashSaleEntry, got {type(value)}"


def test_state_trailing_stops_active_holds_dataclass_objects():
    ts = TrailingStop(entry_price=100.0, peak_price=120.0, activated="2024-01-10")
    state = State(
        rolling_peak_equity=1000.0,
        rolling_peak_timestamp="2024-01-01T00:00:00Z",
        last_run_timestamp="2024-01-01T00:00:00Z",
        ticker_last_buy_date={},
        wash_sale_blacklist={},
        trailing_stops_active={"NVDA": ts},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=False,
    )
    value = state.trailing_stops_active["NVDA"]
    assert isinstance(value, TrailingStop), f"Expected TrailingStop, got {type(value)}"
