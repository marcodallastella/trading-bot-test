# tests/test_risk.py
import os
import pytest
from dataclasses import replace
from src.risk import (
    update_rolling_peak, compute_drawdown, update_drawdown_halt,
    is_kill_switch_active,
    is_wash_sale_blocked, add_wash_sale, clean_expired_wash_sales,
    should_activate_trailing_stop, activate_trailing_stop,
    update_trailing_stop_peak, check_trailing_stop_trigger,
    deactivate_trailing_stop,
)
from src.models import State, TrailingStop, WashSaleEntry


@pytest.fixture
def base_state() -> State:
    return State(
        rolling_peak_equity=100.0,
        rolling_peak_timestamp="2026-01-01",
        last_run_timestamp="2026-01-01T15:00:00Z",
        ticker_last_buy_date={},
        wash_sale_blacklist={},
        trailing_stops_active={},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=False,
    )


# ── rolling peak ──────────────────────────────────────────────────────────────

def test_update_rolling_peak_new_high(base_state):
    updated = update_rolling_peak(base_state, current_equity=110.0, as_of="2026-04-01")
    assert updated.rolling_peak_equity == 110.0
    assert updated.rolling_peak_timestamp == "2026-04-01"


def test_update_rolling_peak_below_peak_unchanged(base_state):
    updated = update_rolling_peak(base_state, current_equity=90.0, as_of="2026-04-01")
    assert updated.rolling_peak_equity == 100.0
    assert updated.rolling_peak_timestamp == "2026-01-01"


def test_update_rolling_peak_equal_unchanged(base_state):
    updated = update_rolling_peak(base_state, current_equity=100.0, as_of="2026-04-01")
    assert updated.rolling_peak_equity == 100.0
    assert updated.rolling_peak_timestamp == "2026-01-01"


def test_update_rolling_peak_resets_after_366_days(base_state):
    # base_state peak is 2026-01-01; 366 days later is 2027-01-02
    updated = update_rolling_peak(base_state, current_equity=80.0, as_of="2027-01-02")
    assert updated.rolling_peak_equity == 80.0
    assert updated.rolling_peak_timestamp == "2027-01-02"


def test_update_rolling_peak_does_not_reset_at_exactly_365_days(base_state):
    # 2026-01-01 + 365 days = 2027-01-01
    updated = update_rolling_peak(base_state, current_equity=80.0, as_of="2027-01-01")
    assert updated.rolling_peak_equity == 100.0


def test_update_rolling_peak_does_not_mutate_input(base_state):
    original_peak = base_state.rolling_peak_equity
    update_rolling_peak(base_state, current_equity=200.0, as_of="2026-04-01")
    assert base_state.rolling_peak_equity == original_peak


# ── drawdown ──────────────────────────────────────────────────────────────────

def test_compute_drawdown_below_peak(base_state):
    dd = compute_drawdown(base_state, current_equity=75.0)
    assert dd == 0.25


def test_compute_drawdown_at_peak(base_state):
    dd = compute_drawdown(base_state, current_equity=100.0)
    assert dd == 0.0


def test_compute_drawdown_above_peak(base_state):
    dd = compute_drawdown(base_state, current_equity=120.0)
    assert dd == 0.0


def test_compute_drawdown_rounds_to_4dp(base_state):
    dd = compute_drawdown(base_state, current_equity=66.666)
    assert dd == round((100.0 - 66.666) / 100.0, 4)


# ── drawdown halt ─────────────────────────────────────────────────────────────

def test_drawdown_halt_fires_at_exactly_25pct(base_state):
    updated = update_drawdown_halt(base_state, current_equity=75.0)
    assert updated.drawdown_halt_active is True


def test_drawdown_halt_fires_above_25pct(base_state):
    updated = update_drawdown_halt(base_state, current_equity=70.0)
    assert updated.drawdown_halt_active is True


def test_drawdown_halt_does_not_fire_below_25pct(base_state):
    updated = update_drawdown_halt(base_state, current_equity=75.01)
    assert updated.drawdown_halt_active is False


def test_drawdown_halt_already_active_stays_active_above_resume(base_state):
    halted = replace(base_state, drawdown_halt_active=True)
    updated = update_drawdown_halt(halted, current_equity=80.0)
    assert updated.drawdown_halt_active is True


def test_drawdown_halt_clears_below_15pct(base_state):
    halted = replace(base_state, drawdown_halt_active=True)
    updated = update_drawdown_halt(halted, current_equity=86.0)
    assert updated.drawdown_halt_active is False


def test_drawdown_halt_does_not_clear_at_exactly_15pct(base_state):
    halted = replace(base_state, drawdown_halt_active=True)
    updated = update_drawdown_halt(halted, current_equity=85.0)
    assert updated.drawdown_halt_active is True


def test_drawdown_halt_not_triggered_when_already_active(base_state):
    halted = replace(base_state, drawdown_halt_active=True)
    updated = update_drawdown_halt(halted, current_equity=70.0)
    assert updated.drawdown_halt_active is True


def test_drawdown_halt_does_not_mutate_input(base_state):
    update_drawdown_halt(base_state, current_equity=70.0)
    assert base_state.drawdown_halt_active is False


# ── kill switch ───────────────────────────────────────────────────────────────

def test_kill_switch_active_when_env_is_1(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "1")
    assert is_kill_switch_active() is True


def test_kill_switch_inactive_when_env_is_0(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "0")
    assert is_kill_switch_active() is False


def test_kill_switch_inactive_when_env_unset(monkeypatch):
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    assert is_kill_switch_active() is False


def test_kill_switch_inactive_for_unexpected_value(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "true")
    assert is_kill_switch_active() is False


# ── wash-sale ─────────────────────────────────────────────────────────────────

def test_wash_sale_blocks_buy_before_expiry(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    assert is_wash_sale_blocked(state, "AAPL", as_of="2026-01-15") is True


def test_wash_sale_blocks_buy_on_expiry_date(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    # expires = 2026-02-01; blocked on that date (today <= expires)
    assert is_wash_sale_blocked(state, "AAPL", as_of="2026-02-01") is True


def test_wash_sale_unblocks_day_after_expiry(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    assert is_wash_sale_blocked(state, "AAPL", as_of="2026-02-02") is False


def test_wash_sale_does_not_block_unknown_ticker(base_state):
    assert is_wash_sale_blocked(base_state, "MSFT", as_of="2026-04-01") is False


def test_add_wash_sale_sets_correct_expiry(base_state):
    state = add_wash_sale(base_state, ticker="NVDA", sold_date="2026-03-01")
    ws = state.wash_sale_blacklist["NVDA"]
    assert ws.sold_date == "2026-03-01"
    assert ws.expires == "2026-04-01"   # 31 days after 2026-03-01


def test_add_wash_sale_does_not_affect_other_tickers(base_state):
    state = add_wash_sale(base_state, ticker="NVDA", sold_date="2026-03-01")
    assert "AAPL" not in state.wash_sale_blacklist


def test_add_wash_sale_does_not_mutate_input(base_state):
    add_wash_sale(base_state, ticker="NVDA", sold_date="2026-03-01")
    assert "NVDA" not in base_state.wash_sale_blacklist


def test_clean_expired_wash_sales_removes_expired(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    cleaned = clean_expired_wash_sales(state, as_of="2026-02-02")
    assert "AAPL" not in cleaned.wash_sale_blacklist


def test_clean_expired_wash_sales_keeps_active_entries(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    cleaned = clean_expired_wash_sales(state, as_of="2026-01-15")
    assert "AAPL" in cleaned.wash_sale_blacklist


def test_clean_expired_wash_sales_mixed(base_state):
    state = add_wash_sale(base_state, ticker="AAPL", sold_date="2026-01-01")
    state = add_wash_sale(state, ticker="MSFT", sold_date="2026-03-01")
    # AAPL expires 2026-02-01 (expired by 2026-04-01), MSFT expires 2026-04-01 (still active on that date)
    cleaned = clean_expired_wash_sales(state, as_of="2026-04-01")
    assert "AAPL" not in cleaned.wash_sale_blacklist
    assert "MSFT" in cleaned.wash_sale_blacklist


# ── trailing stops ────────────────────────────────────────────────────────────

def test_should_activate_trailing_stop_at_exactly_20pct():
    assert should_activate_trailing_stop(0.20) is True


def test_should_activate_trailing_stop_above_20pct():
    assert should_activate_trailing_stop(0.35) is True


def test_should_not_activate_trailing_stop_below_20pct():
    assert should_activate_trailing_stop(0.1999) is False


def test_activate_trailing_stop_adds_entry(base_state):
    updated = activate_trailing_stop(
        base_state, ticker="PLTR",
        entry_price=22.40, current_price=28.15,
        activated_date="2026-04-12",
    )
    assert "PLTR" in updated.trailing_stops_active
    ts = updated.trailing_stops_active["PLTR"]
    assert isinstance(ts, TrailingStop)
    assert ts.entry_price == 22.40
    assert ts.peak_price == 28.15
    assert ts.activated == "2026-04-12"


def test_activate_trailing_stop_does_not_mutate_input(base_state):
    activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    assert "PLTR" not in base_state.trailing_stops_active


def test_update_trailing_stop_peak_raises_when_higher(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    updated = update_trailing_stop_peak(state, "PLTR", current_price=30.00)
    assert updated.trailing_stops_active["PLTR"].peak_price == 30.00


def test_update_trailing_stop_peak_unchanged_when_lower(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    updated = update_trailing_stop_peak(state, "PLTR", current_price=25.00)
    assert updated.trailing_stops_active["PLTR"].peak_price == 28.15


def test_update_trailing_stop_peak_unchanged_when_equal(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    updated = update_trailing_stop_peak(state, "PLTR", current_price=28.15)
    assert updated.trailing_stops_active["PLTR"].peak_price == 28.15


def test_update_trailing_stop_peak_noop_for_unknown_ticker(base_state):
    updated = update_trailing_stop_peak(base_state, "UNKNOWN", current_price=100.0)
    assert updated.trailing_stops_active == {}


def test_check_trailing_stop_triggers_at_exactly_10pct_from_peak(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    current = round(28.15 * 0.90, 4)  # exactly 10% below peak
    should_sell, dd = check_trailing_stop_trigger(state, "PLTR", current_price=current)
    assert should_sell is True
    assert dd == pytest.approx(0.10, abs=1e-4)


def test_check_trailing_stop_does_not_trigger_before_10pct(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    current = round(28.15 * 0.9001, 4)  # 9.99% below peak
    should_sell, dd = check_trailing_stop_trigger(state, "PLTR", current_price=current)
    assert should_sell is False


def test_check_trailing_stop_returns_false_for_unknown_ticker(base_state):
    should_sell, dd = check_trailing_stop_trigger(base_state, "UNKNOWN", current_price=50.0)
    assert should_sell is False
    assert dd == 0.0


def test_check_trailing_stop_returns_drawdown_fraction(base_state):
    state = activate_trailing_stop(base_state, "TEST", 80.0, 100.0, "2026-04-01")
    should_sell, dd = check_trailing_stop_trigger(state, "TEST", current_price=88.0)
    assert should_sell is True
    assert dd == pytest.approx(0.12, abs=1e-4)


def test_deactivate_trailing_stop_removes_ticker(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    assert "PLTR" in state.trailing_stops_active
    updated = deactivate_trailing_stop(state, "PLTR")
    assert "PLTR" not in updated.trailing_stops_active


def test_deactivate_trailing_stop_noop_for_unknown_ticker(base_state):
    updated = deactivate_trailing_stop(base_state, "UNKNOWN")
    assert updated.trailing_stops_active == {}


def test_deactivate_trailing_stop_does_not_affect_other_tickers(base_state):
    state = activate_trailing_stop(base_state, "PLTR", 22.40, 28.15, "2026-04-12")
    state = activate_trailing_stop(state, "NVDA", 100.0, 130.0, "2026-04-10")
    updated = deactivate_trailing_stop(state, "PLTR")
    assert "NVDA" in updated.trailing_stops_active
    assert "PLTR" not in updated.trailing_stops_active
