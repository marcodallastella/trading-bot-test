# tests/test_sizing.py
import pytest
from unittest.mock import patch
from src.sizing import get_equity_tier, compute_kelly_base, compute_position_size
from src.models import BuySignal, SizingResult
from src import config


# ── helpers ───────────────────────────────────────────────────────────────────

def make_signal(
    ticker: str = "AAPL",
    signal_mult: float = 1.0,
    vol_factor: float = 1.0,
    pir: float = 0.25,
    is_deep_dip: bool = False,
) -> BuySignal:
    return BuySignal(
        ticker=ticker,
        price=100.0,
        pir=pir,
        sma_50=90.0,
        atr_pct=0.03,
        signal_mult=signal_mult,
        vol_factor=vol_factor,
        is_deep_dip=is_deep_dip,
    )


# ── equity tier selection ─────────────────────────────────────────────────────

def test_get_equity_tier_100_usd():
    tier = get_equity_tier(100.0)
    assert tier["min"] == 0
    assert tier["max"] == 500
    assert tier["per_ticker_cap"] == 0.15
    assert tier["max_positions"] == 5


def test_get_equity_tier_at_500_boundary():
    # At exactly 500, should be the SECOND tier (min=500 is inclusive)
    tier = get_equity_tier(500.0)
    assert tier["min"] == 500
    assert tier["max"] == 2500


def test_get_equity_tier_just_below_500():
    tier = get_equity_tier(499.99)
    assert tier["min"] == 0
    assert tier["max"] == 500


def test_get_equity_tier_2500():
    tier = get_equity_tier(2500.0)
    assert tier["min"] == 2500


def test_get_equity_tier_large():
    tier = get_equity_tier(50_000.0)
    assert tier["min"] == 10000
    assert tier["per_ticker_cap"] == 0.05
    assert tier["max_positions"] == 10


# ── Kelly base clamping ───────────────────────────────────────────────────────

def test_compute_kelly_base_within_tier_range(monkeypatch):
    # Default BASE_KELLY_FRACTION=0.08 is within tier-1 [0.08, 0.12] → returns 0.08
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_kelly_base(100.0)
    assert result == 0.08


def test_compute_kelly_base_clamped_to_global_max(monkeypatch):
    # BASE_KELLY_FRACTION above global max of 0.12 → clamped to 0.12
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.20)
    result = compute_kelly_base(100.0)
    assert result == config.BASE_KELLY_FRACTION_MAX


def test_compute_kelly_base_clamped_to_global_min(monkeypatch):
    # BASE_KELLY_FRACTION below global min of 0.04 → clamped to 0.04
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.01)
    result = compute_kelly_base(100.0)
    assert result == config.BASE_KELLY_FRACTION_MIN


def test_compute_kelly_base_clamped_to_tier_max(monkeypatch):
    # In tier 4 (equity=$50k), kelly_max=0.06. BASE=0.10 → clamped to 0.06.
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.10)
    result = compute_kelly_base(50_000.0)
    assert result == 0.06


def test_compute_kelly_base_clamped_to_tier_min(monkeypatch):
    # In tier 2 (equity=$1k), kelly_min=0.06. BASE=0.04 → global floor=0.04,
    # tier floor=0.06; tier floor wins → 0.06
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.04)
    result = compute_kelly_base(1000.0)
    assert result == 0.06


def test_compute_kelly_base_returns_float(monkeypatch):
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_kelly_base(100.0)
    assert isinstance(result, float)


# ── core pipeline ─────────────────────────────────────────────────────────────

def test_sizing_basic_pipeline(monkeypatch):
    """Standard signal, bull regime, no existing exposure: raw formula only."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.skip is False
    assert result.kelly_base == 0.08
    assert result.signal_mult == 1.0
    assert result.vol_factor == 1.0
    assert result.regime_mult == 1.0
    assert result.position_size_pct == pytest.approx(0.08, abs=1e-4)
    assert result.buy_amount_usd == pytest.approx(8.0, abs=0.01)
    assert result.capped is False


def test_sizing_bear_regime_reduces_to_30pct(monkeypatch):
    """Gate C critical: BEAR regime_mult=0.30 reduces final size to 30% of kelly_base."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=config.REGIME_MULT_BEAR,   # 0.30
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.skip is False
    assert result.position_size_pct == pytest.approx(0.08 * 0.30, abs=1e-4)
    assert result.buy_amount_usd == pytest.approx(100.0 * 0.08 * 0.30, abs=0.01)


def test_sizing_deep_crash_multiplier(monkeypatch):
    """Deep crash (PIR<0.10): signal_mult=1.5 scales up the position."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="NVDA",
        total_equity=100.0,
        signal=make_signal(signal_mult=config.SIGNAL_MULT_DEEP_CRASH, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.skip is False
    assert result.position_size_pct == pytest.approx(0.08 * 1.5, abs=1e-4)


def test_sizing_high_vol_reduces_size(monkeypatch):
    """vol_factor=0.7 for noisy names reduces position size."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="COIN",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=config.VOL_FACTOR_HIGH),  # 0.7
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.buy_amount_usd == pytest.approx(100.0 * 0.08 * 0.7, abs=0.01)


def test_sizing_kelly_clamped_to_max_applies_to_buy_amount(monkeypatch):
    """Gate C critical: even if BASE_KELLY_FRACTION=0.20, buy_amount uses clamped 0.12."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.20)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.kelly_base == config.BASE_KELLY_FRACTION_MAX   # 0.12
    assert result.buy_amount_usd == pytest.approx(100.0 * 0.12, abs=0.01)


def test_sizing_result_populates_all_fields(monkeypatch):
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="MSFT",
        total_equity=100.0,
        signal=make_signal(),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.ticker == "MSFT"
    assert isinstance(result.kelly_base, float)
    assert isinstance(result.position_size_pct, float)
    assert isinstance(result.buy_amount_usd, float)
    assert isinstance(result.capped, bool)
    assert isinstance(result.cap_reason, str)
    assert isinstance(result.skip, bool)
    assert isinstance(result.skip_reason, str)


# ── cap enforcement ───────────────────────────────────────────────────────────

def test_sizing_per_ticker_cap_limits_buy_amount(monkeypatch):
    """Already holds $10 in AAPL; cap is $15; raw=$8 → $5 allowed, capped=True."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=10.0,    # $10 already in AAPL
        current_total_deployed_usd=10.0,
        current_position_count=1,
    )
    assert result.skip is False
    assert result.capped is True
    assert result.buy_amount_usd == pytest.approx(5.0, abs=0.01)  # 15-10=5
    assert "per-ticker" in result.cap_reason


def test_sizing_total_deployment_cap_limits_buy_amount(monkeypatch):
    """Gate C critical: $87 deployed out of $90 max → only $3 more allowed."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=87.0,     # 87 of 90 max deployed
        current_position_count=4,
    )
    assert result.skip is False
    assert result.capped is True
    assert result.buy_amount_usd == pytest.approx(3.0, abs=0.01)
    assert "deployment" in result.cap_reason


def test_sizing_skip_when_per_ticker_at_cap(monkeypatch):
    """Already at per-ticker limit → skip."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(),
        regime_mult=1.0,
        current_ticker_exposure_usd=15.0,    # = 15% of $100 → at cap
        current_total_deployed_usd=15.0,
        current_position_count=1,
    )
    assert result.skip is True
    assert "per-ticker" in result.skip_reason


def test_sizing_skip_when_total_deployed_at_cap(monkeypatch):
    """$90 of $100 already deployed → skip."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=90.0,     # = 90% → at cap
        current_position_count=3,
    )
    assert result.skip is True
    assert "deployment" in result.skip_reason


def test_sizing_skip_when_position_count_at_tier_max(monkeypatch):
    """5 positions open (tier-1 max) → skip."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=100.0,
        signal=make_signal(),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=60.0,
        current_position_count=5,            # = tier-1 max_positions
    )
    assert result.skip is True
    assert "max positions" in result.skip_reason


def test_sizing_skip_below_min_notional(monkeypatch):
    """After caps, final buy_usd < $1.00 → skip."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL",
        total_equity=5.0,      # $5 equity → 0.08 × $5 = $0.40 < $1 min
        signal=make_signal(signal_mult=1.0, vol_factor=1.0),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    assert result.skip is True
    assert "minimum" in result.skip_reason


def test_sizing_deep_crash_can_hit_per_ticker_cap(monkeypatch):
    """deep crash (1.5×) on low-vol (1.2×) at $100: 0.08×1.5×1.2=0.144=$14.40 < $15 cap → not capped."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="KO",
        total_equity=100.0,
        signal=make_signal(
            signal_mult=config.SIGNAL_MULT_DEEP_CRASH,   # 1.5
            vol_factor=config.VOL_FACTOR_LOW,            # 1.2
        ),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    # 0.08 * 1.5 * 1.2 * 1.0 = 0.144 = $14.40 — under $15 cap
    assert result.skip is False
    assert result.capped is False
    assert result.buy_amount_usd == pytest.approx(14.40, abs=0.01)


def test_sizing_per_ticker_cap_fires_when_deep_crash_exceeds_15pct(monkeypatch):
    """With larger kelly_base, deep crash × low vol can exceed 15% → capped to $15."""
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.12)  # max allowed
    result = compute_position_size(
        ticker="KO",
        total_equity=100.0,
        signal=make_signal(
            signal_mult=config.SIGNAL_MULT_DEEP_CRASH,   # 1.5
            vol_factor=config.VOL_FACTOR_LOW,            # 1.2
        ),
        regime_mult=1.0,
        current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0,
        current_position_count=0,
    )
    # 0.12 * 1.5 * 1.2 = 0.216 = $21.60 > $15 cap → capped to $15
    assert result.capped is True
    assert result.buy_amount_usd == pytest.approx(15.0, abs=0.01)


def test_sizing_skip_reason_empty_when_not_skipped(monkeypatch):
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL", total_equity=100.0, signal=make_signal(),
        regime_mult=1.0, current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0, current_position_count=0,
    )
    assert result.skip is False
    assert result.skip_reason == ""


def test_sizing_cap_reason_empty_when_not_capped(monkeypatch):
    monkeypatch.setattr(config, "BASE_KELLY_FRACTION", 0.08)
    result = compute_position_size(
        ticker="AAPL", total_equity=100.0, signal=make_signal(),
        regime_mult=1.0, current_ticker_exposure_usd=0.0,
        current_total_deployed_usd=0.0, current_position_count=0,
    )
    assert result.capped is False
    assert result.cap_reason == ""
