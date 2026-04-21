# src/risk.py
import os
from dataclasses import replace
from datetime import date, timedelta

from src import config
from src.logger import get_logger
from src.models import State, TrailingStop, WashSaleEntry

log = get_logger(__name__)


# ── Rolling peak and drawdown ─────────────────────────────────────────────────

def update_rolling_peak(state: State, current_equity: float, as_of: str) -> State:
    """
    Replace the stored rolling peak if current_equity is higher OR if the
    stored peak is older than ROLLING_PEAK_WINDOW_DAYS (stale high-water mark).
    Returns a new State; never mutates input.
    """
    as_of_date = date.fromisoformat(as_of)
    peak_date = date.fromisoformat(state.rolling_peak_timestamp)
    peak_age_days = (as_of_date - peak_date).days

    if current_equity > state.rolling_peak_equity or peak_age_days > config.ROLLING_PEAK_WINDOW_DAYS:
        return replace(
            state,
            rolling_peak_equity=round(current_equity, 4),
            rolling_peak_timestamp=as_of,
        )
    return state


def compute_drawdown(state: State, current_equity: float) -> float:
    """Return drawdown fraction from rolling peak. Always >= 0.0."""
    if state.rolling_peak_equity <= 0:
        return 0.0
    raw = (state.rolling_peak_equity - current_equity) / state.rolling_peak_equity
    return round(max(raw, 0.0), 4)


def update_drawdown_halt(state: State, current_equity: float) -> State:
    """
    Activate halt when drawdown >= DRAWDOWN_HALT_THRESHOLD.
    Clear halt only when drawdown < DRAWDOWN_RESUME_THRESHOLD (hysteresis gap).
    Returns new State; never mutates input.
    """
    dd = compute_drawdown(state, current_equity)

    if not state.drawdown_halt_active and dd >= config.DRAWDOWN_HALT_THRESHOLD:
        log.critical(
            f"DRAWDOWN HALT ACTIVATED: {dd:.1%} >= {config.DRAWDOWN_HALT_THRESHOLD:.0%} threshold"
        )
        return replace(state, drawdown_halt_active=True)

    if state.drawdown_halt_active and dd < config.DRAWDOWN_RESUME_THRESHOLD:
        log.info(
            f"DRAWDOWN HALT CLEARED: recovered to {dd:.1%} "
            f"(resume threshold {config.DRAWDOWN_RESUME_THRESHOLD:.0%})"
        )
        return replace(state, drawdown_halt_active=False)

    return state


# ── Kill switch ───────────────────────────────────────────────────────────────

def is_kill_switch_active() -> bool:
    """Returns True only when KILL_SWITCH env var is exactly '1'."""
    return os.environ.get("KILL_SWITCH", "0") == "1"


# ── Wash-sale guard ───────────────────────────────────────────────────────────

def is_wash_sale_blocked(state: State, ticker: str, as_of: str) -> bool:
    """Return True if ticker is in the wash-sale blacklist and the block has not expired."""
    if ticker not in state.wash_sale_blacklist:
        return False
    ws = state.wash_sale_blacklist[ticker]
    today = date.fromisoformat(as_of)
    expires = date.fromisoformat(ws.expires)
    return today <= expires


def add_wash_sale(state: State, ticker: str, sold_date: str) -> State:
    """Add ticker to the wash-sale blacklist for WASH_SALE_DAYS calendar days."""
    sold = date.fromisoformat(sold_date)
    expires = sold + timedelta(days=config.WASH_SALE_DAYS)
    entry = WashSaleEntry(sold_date=sold_date, expires=expires.isoformat())
    return replace(state, wash_sale_blacklist={**state.wash_sale_blacklist, ticker: entry})


def clean_expired_wash_sales(state: State, as_of: str) -> State:
    """Remove entries whose expiry date is strictly before as_of."""
    today = date.fromisoformat(as_of)
    active = {
        t: ws
        for t, ws in state.wash_sale_blacklist.items()
        if date.fromisoformat(ws.expires) >= today
    }
    return replace(state, wash_sale_blacklist=active)


# ── Trailing stops ────────────────────────────────────────────────────────────

def should_activate_trailing_stop(unrealized_pct: float) -> bool:
    """Return True when unrealized gain has reached the activation threshold."""
    return unrealized_pct >= config.TRAILING_STOP_ACTIVATION_PCT


def activate_trailing_stop(
    state: State,
    ticker: str,
    entry_price: float,
    current_price: float,
    activated_date: str,
) -> State:
    """Record a new trailing stop. peak_price is set to current_price at activation."""
    ts = TrailingStop(
        entry_price=round(entry_price, 4),
        peak_price=round(current_price, 4),
        activated=activated_date,
    )
    return replace(state, trailing_stops_active={**state.trailing_stops_active, ticker: ts})


def update_trailing_stop_peak(state: State, ticker: str, current_price: float) -> State:
    """Ratchet peak_price upward only — never downward."""
    if ticker not in state.trailing_stops_active:
        return state
    ts = state.trailing_stops_active[ticker]
    if current_price <= ts.peak_price:
        return state
    updated_ts = replace(ts, peak_price=round(current_price, 4))
    return replace(state, trailing_stops_active={**state.trailing_stops_active, ticker: updated_ts})


def check_trailing_stop_trigger(
    state: State, ticker: str, current_price: float
) -> tuple[bool, float]:
    """
    Returns (should_sell, drawdown_from_peak).
    Returns (False, 0.0) when no active trailing stop exists for ticker.
    """
    if ticker not in state.trailing_stops_active:
        return False, 0.0
    ts = state.trailing_stops_active[ticker]
    dd = round((ts.peak_price - current_price) / ts.peak_price, 4)
    return dd >= config.TRAILING_STOP_TRAIL_PCT, dd


def deactivate_trailing_stop(state: State, ticker: str) -> State:
    """Remove trailing stop entry. No-op if ticker is not present."""
    new_stops = {t: ts for t, ts in state.trailing_stops_active.items() if t != ticker}
    return replace(state, trailing_stops_active=new_stops)
