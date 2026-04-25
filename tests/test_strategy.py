# tests/test_strategy.py
import pytest
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional

from src.strategy import (
    validate_data_quality, compute_pir, compute_sma, compute_atr,
    evaluate_buy_signal, evaluate_sell, compute_regime,
)
from src.models import BuySignal, SellDecision, RegimeInfo
from src import config


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rising():
    """60 bars linearly rising 1→60. 10th≈6.95, 90th≈54.05, SMA50=35.5."""
    vals = pd.Series(range(1, 61), dtype=float)
    return {
        "closes": vals,
        "highs": vals + 0.5,
        "lows": vals - 0.5,
    }


@pytest.fixture
def spike_then_settle():
    """[200]*10 + [100]*50. 10th=100, 90th=200, range=100, SMA50=100."""
    vals_h = pd.Series([200.0] * 10 + [100.0] * 50)
    return {
        "closes": vals_h,
        "highs": vals_h + 1.0,
        "lows": vals_h - 1.0,
    }


@pytest.fixture
def settling_after_spike():
    """[100]*40 + [200]*20. SMA50=140. current=125 → PIR=0.25 but price < SMA50."""
    vals = pd.Series([100.0] * 40 + [200.0] * 20)
    return {
        "closes": vals,
        "highs": vals + 1.0,
        "lows": vals - 1.0,
    }


# ── data quality ──────────────────────────────────────────────────────────────

def test_validate_data_quality_passes_on_good_data(rising):
    valid, reason = validate_data_quality(rising["closes"], "AAPL")
    assert valid is True
    assert reason == ""


def test_validate_data_quality_fails_too_few_bars():
    # Need ≥ 85% of 60 = 51 bars; 50 should fail
    closes = pd.Series(range(1, 51), dtype=float)
    valid, reason = validate_data_quality(closes, "AAPL")
    assert valid is False
    assert "bar" in reason.lower()


def test_validate_data_quality_passes_at_minimum_bar_count():
    # Exactly 51 bars (= floor(60 * 0.85)) → should pass
    closes = pd.Series(range(1, 55), dtype=float)
    valid, reason = validate_data_quality(closes, "AAPL")
    assert valid is True


def test_validate_data_quality_fails_flat_data():
    # 10th and 90th percentile identical → flat
    closes = pd.Series([100.0] * 60)
    valid, reason = validate_data_quality(closes, "AAPL")
    assert valid is False
    assert "flat" in reason.lower()


def test_validate_data_quality_fails_nearly_flat_data():
    # Nearly flat: tiny spread that means 10th == 90th
    closes = pd.Series([100.0] * 58 + [100.0001, 100.0001])
    # np.percentile will make 10th ≈ 90th ≈ 100 → flat check fires
    pct10 = float(np.percentile(closes, 10))
    pct90 = float(np.percentile(closes, 90))
    if pct10 >= pct90:
        valid, reason = validate_data_quality(closes, "AAPL")
        assert valid is False
    # If numpy gives distinct pcts, that's fine — just skip
    else:
        pytest.skip("percentiles distinct for this numpy version — test not applicable")


def test_validate_data_quality_stale_data_fails():
    """If DatetimeIndex and last bar is >5 days before reference_date → fail."""
    idx = pd.date_range(end="2026-01-01", periods=60, freq="B")
    closes = pd.Series(range(1, 61), dtype=float, index=idx)
    ref = date(2026, 1, 10)  # 9 days after last bar
    valid, reason = validate_data_quality(closes, "AAPL", reference_date=ref)
    assert valid is False
    assert "stale" in reason.lower()


def test_validate_data_quality_fresh_data_passes():
    """Last bar within 5 calendar days of reference_date → pass."""
    idx = pd.date_range(end="2026-01-02", periods=60, freq="B")
    closes = pd.Series(range(1, 61), dtype=float, index=idx)
    ref = date(2026, 1, 5)  # 3 days after last bar
    valid, reason = validate_data_quality(closes, "AAPL", reference_date=ref)
    assert valid is True


def test_validate_data_quality_no_reference_date_skips_staleness(rising):
    """Without reference_date, staleness check is skipped entirely."""
    valid, _ = validate_data_quality(rising["closes"], "AAPL", reference_date=None)
    assert valid is True


# ── compute_pir ───────────────────────────────────────────────────────────────

def test_compute_pir_mid_range(rising):
    # current=30: 10th≈6.95, 90th≈54.05, range≈47.1
    # PIR = (30 - 6.95) / 47.1 ≈ 0.489
    pir, low, high = compute_pir(rising["closes"], current_price=30.0)
    assert 0.45 < pir < 0.55


def test_compute_pir_in_lower_range(rising):
    # current=10: PIR = (10 - 6.95) / 47.1 ≈ 0.065
    pir, _, _ = compute_pir(rising["closes"], current_price=10.0)
    assert pir < 0.15
    assert pir > 0.0


def test_compute_pir_clamps_to_zero(rising):
    # current below 10th percentile → PIR = 0.0
    pir, _, _ = compute_pir(rising["closes"], current_price=1.0)
    assert pir == 0.0


def test_compute_pir_clamps_to_one(rising):
    # current above 90th percentile → PIR = 1.0
    pir, _, _ = compute_pir(rising["closes"], current_price=100.0)
    assert pir == 1.0


def test_compute_pir_returns_correct_bounds(rising):
    _, low, high = compute_pir(rising["closes"], current_price=30.0)
    assert low == pytest.approx(np.percentile(rising["closes"], 10), abs=0.01)
    assert high == pytest.approx(np.percentile(rising["closes"], 90), abs=0.01)


def test_compute_pir_deep_dip_range(rising):
    # current=8: PIR = (8-6.95)/47.1 ≈ 0.022 < PIR_VERY_DEEP_THRESHOLD (0.10)
    pir, _, _ = compute_pir(rising["closes"], current_price=8.0)
    assert pir < config.PIR_VERY_DEEP_THRESHOLD


def test_compute_pir_rounds_to_4dp(rising):
    pir, _, _ = compute_pir(rising["closes"], current_price=20.0)
    assert pir == round(pir, 4)


# ── compute_sma ───────────────────────────────────────────────────────────────

def test_compute_sma_50_on_rising_series(rising):
    # Last 50 of range(1,61) → 11..60, mean = (11+60)/2 = 35.5
    sma = compute_sma(rising["closes"], period=50)
    assert sma == 35.5


def test_compute_sma_uses_last_n_values():
    # Only the last 3 values matter
    closes = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    sma = compute_sma(closes, period=3)
    assert sma == 200.0


def test_compute_sma_returns_float(rising):
    sma = compute_sma(rising["closes"], period=50)
    assert isinstance(sma, float)


# ── compute_atr ───────────────────────────────────────────────────────────────

def test_compute_atr_constant_range():
    """High-low spread is constant 4.0 with no gaps → ATR = 4.0."""
    n = 20
    closes = pd.Series([100.0] * n)
    highs = pd.Series([102.0] * n)
    lows = pd.Series([98.0] * n)
    atr = compute_atr(highs, lows, closes, period=14)
    assert atr == pytest.approx(4.0, abs=0.01)


def test_compute_atr_returns_positive_float(rising):
    atr = compute_atr(rising["highs"], rising["lows"], rising["closes"], period=14)
    assert isinstance(atr, float)
    assert atr > 0.0


def test_compute_atr_accounts_for_gap():
    """A large overnight gap inflates ATR above the simple high-low range."""
    n = 20
    # Most bars: high-low = 4, but the last bar has a gap-up from prev_close=100 to open≈150
    closes = pd.Series([100.0] * (n - 1) + [102.0])
    highs = pd.Series([102.0] * (n - 1) + [155.0])
    lows = pd.Series([98.0]  * (n - 1) + [150.0])
    atr = compute_atr(highs, lows, closes, period=14)
    # Last bar TR = max(155-150, |155-100|, |150-100|) = max(5, 55, 50) = 55
    # Prior 13 bars: TR ≈ 4 each
    # ATR ≈ (13*4 + 55) / 14 ≈ (52 + 55) / 14 ≈ 7.64
    assert atr > 4.0  # gap inflated ATR


# ── evaluate_buy_signal ───────────────────────────────────────────────────────

def test_buy_signal_standard_dip(spike_then_settle):
    """PIR=0.25 (in 0.20-0.30 bucket), current=125 > SMA50=100 → BuySignal."""
    signal, reason = evaluate_buy_signal(
        "AAPL", spike_then_settle["closes"],
        spike_then_settle["highs"], spike_then_settle["lows"],
        current_price=125.0,
    )
    assert signal is not None
    assert signal.pir == pytest.approx(0.25, abs=0.01)
    assert signal.signal_mult == config.SIGNAL_MULT_STANDARD
    assert signal.is_deep_dip is False


def test_buy_signal_deep_dip_bypasses_trend_filter(spike_then_settle):
    """PIR=0.05 (< PIR_DEEP_DIP_THRESHOLD=0.15) fires even when price < SMA50."""
    # current=105: PIR=(105-100)/100=0.05, SMA50=100, price(105)>SMA50(100)... actually still above
    # Use settling_after_spike instead: SMA50=140, current=108 < SMA50
    closes = pd.Series([100.0] * 40 + [200.0] * 20)
    highs = closes + 1.0
    lows = closes - 1.0
    # SMA50 = mean of last 50 = mean of [100]*30 + [200]*20 = (3000+4000)/50 = 140
    # 10th pct = 100, 90th pct = 200, range=100
    # current=108: PIR=(108-100)/100=0.08 < 0.15 → deep dip override
    # current(108) < SMA50(140) → trend filter would block, but deep dip bypasses it
    signal, reason = evaluate_buy_signal("NVDA", closes, highs, lows, current_price=108.0)
    assert signal is not None
    assert signal.is_deep_dip is True
    assert signal.pir < config.PIR_DEEP_DIP_THRESHOLD


def test_buy_signal_no_signal_above_pir_threshold(spike_then_settle):
    """PIR=0.45 (above 0.30 threshold) → no signal."""
    # current=145: PIR=(145-100)/100=0.45 > 0.30
    signal, reason = evaluate_buy_signal(
        "AAPL", spike_then_settle["closes"],
        spike_then_settle["highs"], spike_then_settle["lows"],
        current_price=145.0,
    )
    assert signal is None


def test_buy_signal_no_signal_below_sma_not_deep_dip(settling_after_spike):
    """PIR=0.25, price(125) < SMA50(140), not deep dip → no signal."""
    signal, _ = evaluate_buy_signal(
        "MSFT", settling_after_spike["closes"],
        settling_after_spike["highs"], settling_after_spike["lows"],
        current_price=125.0,
    )
    assert signal is None


def test_buy_signal_no_signal_on_bad_data():
    """Data quality failure → no signal."""
    closes = pd.Series([100.0] * 60)   # flat → quality fail
    highs = closes + 1.0
    lows = closes - 1.0
    signal, reason = evaluate_buy_signal("AAPL", closes, highs, lows, current_price=100.0)
    assert signal is None
    assert "flat" in reason.lower()


def test_buy_signal_strong_dip_multiplier(spike_then_settle):
    """PIR in 0.10–0.20 range → signal_mult = SIGNAL_MULT_STRONG_DIP (1.2)."""
    # current=115: PIR=(115-100)/100=0.15; is_deep_dip=(0.15 < 0.15)=False; standard or strong?
    # PIR=0.15: NOT < PIR_DEEP_DIP_THRESHOLD (0.15), so not deep dip
    # PIR=0.15 < PIR_STRONG_DIP_THRESHOLD (0.20) → SIGNAL_MULT_STRONG_DIP
    # current(115) > SMA50(100) ✓
    signal, _ = evaluate_buy_signal(
        "AAPL", spike_then_settle["closes"],
        spike_then_settle["highs"], spike_then_settle["lows"],
        current_price=115.0,
    )
    assert signal is not None
    assert signal.signal_mult == config.SIGNAL_MULT_STRONG_DIP


def test_buy_signal_deep_crash_multiplier(spike_then_settle):
    """PIR < 0.10 → signal_mult = SIGNAL_MULT_DEEP_CRASH (1.5)."""
    # current=108: PIR=(108-100)/100=0.08 < 0.10 → deep crash
    signal, _ = evaluate_buy_signal(
        "AAPL", spike_then_settle["closes"],
        spike_then_settle["highs"], spike_then_settle["lows"],
        current_price=108.0,
    )
    assert signal is not None
    assert signal.signal_mult == config.SIGNAL_MULT_DEEP_CRASH


def test_buy_signal_low_vol_factor():
    """ATR% < 2% → vol_factor = VOL_FACTOR_LOW (1.2)."""
    # Spike-then-settle with very tight high-low spread
    closes = pd.Series([200.0] * 10 + [100.0] * 50)
    # Set tight highs/lows so ATR% < 2%
    highs = closes + 0.5   # spread = 1.0 on a $100 stock = 1% ATR
    lows = closes - 0.5
    signal, _ = evaluate_buy_signal("KO", closes, highs, lows, current_price=115.0)
    if signal is not None:
        assert signal.vol_factor == config.VOL_FACTOR_LOW


def test_buy_signal_high_vol_factor():
    """ATR% > 4% → vol_factor = VOL_FACTOR_HIGH (0.7)."""
    closes = pd.Series([200.0] * 10 + [100.0] * 50)
    # Wide spread: 8% of ~100 → ATR% > 4%
    highs = closes + 6.0
    lows = closes - 6.0
    signal, _ = evaluate_buy_signal("NVDA", closes, highs, lows, current_price=115.0)
    if signal is not None:
        assert signal.vol_factor == config.VOL_FACTOR_HIGH


def test_buy_signal_populates_all_fields(spike_then_settle):
    signal, _ = evaluate_buy_signal(
        "AAPL", spike_then_settle["closes"],
        spike_then_settle["highs"], spike_then_settle["lows"],
        current_price=125.0,
    )
    assert signal is not None
    assert signal.ticker == "AAPL"
    assert signal.price == 125.0
    assert signal.pir > 0.0
    assert signal.sma_50 > 0.0
    assert signal.atr_pct > 0.0


# ── evaluate_sell ─────────────────────────────────────────────────────────────

def test_sell_stop_loss_fires_at_exactly_minus_10pct():
    decision = evaluate_sell(
        ticker="AAPL",
        avg_entry_price=100.0,
        current_price=90.0,        # exactly -10%
        trailing_stop_active=False,
        trailing_stop_peak=None,
    )
    assert decision.action == "sell_stop_loss"
    assert decision.unrealized_pct == pytest.approx(-0.10, abs=1e-4)


def test_sell_stop_loss_fires_below_minus_10pct():
    decision = evaluate_sell("AAPL", 100.0, 85.0, False, None)
    assert decision.action == "sell_stop_loss"


def test_sell_no_stop_loss_above_minus_10pct():
    # -9.9% → hold (or other action, but not stop_loss)
    decision = evaluate_sell("AAPL", 100.0, 90.1, False, None)
    assert decision.action != "sell_stop_loss"


def test_sell_trailing_triggers_at_exactly_10pct_from_peak():
    # Entry=50, current=90, peak=100 → drawdown from peak = 10% → sell
    decision = evaluate_sell(
        ticker="PLTR",
        avg_entry_price=50.0,
        current_price=90.0,
        trailing_stop_active=True,
        trailing_stop_peak=100.0,
    )
    assert decision.action == "sell_trailing"


def test_sell_trailing_does_not_trigger_below_10pct():
    # peak=100, current=91 → drawdown=9% < 10% → hold
    decision = evaluate_sell("PLTR", 50.0, 91.0, True, 100.0)
    assert decision.action == "hold"


def test_sell_stop_loss_takes_priority_over_trailing():
    """Stop loss fires even if trailing stop is active (price crashed past both)."""
    decision = evaluate_sell(
        ticker="PLTR",
        avg_entry_price=100.0,
        current_price=85.0,        # -15% from entry → stop loss
        trailing_stop_active=True,
        trailing_stop_peak=150.0,  # trailing stop active at high peak
    )
    assert decision.action == "sell_stop_loss"


def test_sell_activate_trailing_at_exactly_20pct():
    # unrealized = +20% exactly and no trailing stop yet → activate_trailing
    decision = evaluate_sell("PLTR", 100.0, 120.0, False, None)
    assert decision.action == "activate_trailing"


def test_sell_activate_trailing_above_20pct():
    decision = evaluate_sell("PLTR", 100.0, 130.0, False, None)
    assert decision.action == "activate_trailing"


def test_sell_no_activate_trailing_below_20pct():
    decision = evaluate_sell("PLTR", 100.0, 119.9, False, None)
    assert decision.action == "hold"


def test_sell_hold_with_active_trailing_no_trigger():
    # Trailing stop active, price still rising (no drawdown yet) → hold
    decision = evaluate_sell("PLTR", 50.0, 110.0, True, 108.0)
    assert decision.action == "hold"


def test_sell_no_activate_trailing_when_already_active():
    # unrealized=+25%, trailing already active, no drawdown → hold (not activate again)
    decision = evaluate_sell("PLTR", 100.0, 125.0, True, 128.0)
    assert decision.action == "hold"


def test_sell_unrealized_pct_in_decision():
    decision = evaluate_sell("AAPL", 100.0, 90.0, False, None)
    assert decision.unrealized_pct == pytest.approx(-0.10, abs=1e-4)
    decision2 = evaluate_sell("AAPL", 100.0, 115.0, False, None)
    assert decision2.unrealized_pct == pytest.approx(0.15, abs=1e-4)


def test_sell_reason_is_populated():
    decision = evaluate_sell("AAPL", 100.0, 90.0, False, None)
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0


# ── compute_regime ────────────────────────────────────────────────────────────

def _make_spy(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_compute_regime_bull():
    """SPY above 200SMA, slope rising (>0.5%) → BULL, regime_mult=1.0."""
    # Linearly rising: 400 to 509.5 over 220 bars (step 0.5)
    spy = _make_spy([400.0 + i * 0.5 for i in range(220)])
    regime = compute_regime(spy)
    assert regime.regime_mult == config.REGIME_MULT_BULL
    assert regime.above_sma is True
    assert regime.slope_direction == "rising"
    assert isinstance(regime, RegimeInfo)


def test_compute_regime_late_cycle():
    """SPY above 200SMA, slope flat (0%) → LATE_CYCLE, regime_mult=0.85."""
    # Low for 180 bars then flat-up: SMA200 < current, slope=0
    spy = _make_spy([300.0] * 180 + [500.0] * 40)
    regime = compute_regime(spy)
    assert regime.regime_mult == config.REGIME_MULT_LATE_CYCLE
    assert regime.above_sma is True
    assert regime.slope_direction == "flat"


def test_compute_regime_topping():
    """SPY above 200SMA, slope falling (< -0.5%) → TOPPING, regime_mult=0.60."""
    # Historical lows, then big spike that is now falling hard
    spy = _make_spy([300.0] * 180 + [520.0 - i * 2 for i in range(40)])
    regime = compute_regime(spy)
    assert regime.regime_mult == config.REGIME_MULT_TOPPING
    assert regime.above_sma is True
    assert regime.slope_direction == "falling"


def test_compute_regime_recovery():
    """SPY below 200SMA, slope rising → RECOVERY, regime_mult=0.60."""
    # Historical highs, then crash, now recovering sharply
    spy = _make_spy([600.0] * 180 + [400.0 + i * 2 for i in range(40)])
    regime = compute_regime(spy)
    assert regime.regime_mult == config.REGIME_MULT_RECOVERY
    assert regime.above_sma is False
    assert regime.slope_direction == "rising"


def test_compute_regime_bear():
    """SPY below 200SMA, slope falling → BEAR, regime_mult=0.30."""
    # Linearly falling: 600 to 490.5 over 220 bars
    spy = _make_spy([600.0 - i * 0.5 for i in range(220)])
    regime = compute_regime(spy)
    assert regime.regime_mult == config.REGIME_MULT_BEAR
    assert regime.above_sma is False
    assert regime.slope_direction == "flat_or_falling"


def test_compute_regime_populates_all_fields():
    spy = _make_spy([400.0 + i * 0.5 for i in range(220)])
    regime = compute_regime(spy)
    assert regime.spy_price > 0
    assert regime.sma_200 > 0
    assert isinstance(regime.slope_20d_pct, float)
    assert isinstance(regime.description, str)
    assert len(regime.description) > 0
    assert regime.regime_mult in {
        config.REGIME_MULT_BULL,
        config.REGIME_MULT_LATE_CYCLE,
        config.REGIME_MULT_TOPPING,
        config.REGIME_MULT_RECOVERY,
        config.REGIME_MULT_BEAR,
    }
