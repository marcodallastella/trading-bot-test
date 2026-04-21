# src/strategy.py
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.logger import get_logger
from src.models import BuySignal, RegimeInfo, SellDecision

log = get_logger(__name__)


def validate_data_quality(
    closes: pd.Series,
    ticker: str,
    reference_date: date | None = None,
) -> tuple[bool, str]:
    """
    Three checks:
    1. ≥ 90% of LOOKBACK_DAYS bars present.
    2. 10th and 90th percentile are distinct (not flat data).
    3. If reference_date provided and closes has DatetimeIndex: last bar ≤ 5 days stale.
    """
    min_bars = int(config.LOOKBACK_DAYS * config.DATA_QUALITY_MIN_BAR_RATIO)
    if len(closes) < min_bars:
        reason = f"{ticker}: {len(closes)} bars present, need ≥ {min_bars}"
        log.warning(f"DATA QUALITY FAIL: {reason}")
        return False, reason

    pct10 = float(np.percentile(closes, 10))
    pct90 = float(np.percentile(closes, 90))
    if pct10 >= pct90:
        reason = f"{ticker}: flat data (10th={pct10:.4f} >= 90th={pct90:.4f})"
        log.warning(f"DATA QUALITY FAIL: {reason}")
        return False, reason

    if reference_date is not None and isinstance(closes.index, pd.DatetimeIndex):
        last_bar_date = closes.index[-1].date()
        days_stale = (reference_date - last_bar_date).days
        if days_stale > 5:
            reason = f"{ticker}: stale data (last bar {last_bar_date}, {days_stale}d ago)"
            log.warning(f"DATA QUALITY FAIL: {reason}")
            return False, reason

    return True, ""


def compute_pir(closes: pd.Series, current_price: float) -> tuple[float, float, float]:
    """
    Position-in-range: where is current_price relative to the 10th–90th percentile
    of the lookback window? Returns (pir, low_bound, high_bound). pir ∈ [0, 1].
    Using 10th/90th percentile (not raw high/low) prevents a single wick from
    poisoning the range.
    """
    low_bound = float(np.percentile(closes, 10))
    high_bound = float(np.percentile(closes, 90))
    if high_bound <= low_bound:
        return 0.0, low_bound, high_bound
    raw = (current_price - low_bound) / (high_bound - low_bound)
    pir = round(float(np.clip(raw, 0.0, 1.0)), 4)
    return pir, low_bound, high_bound


def compute_sma(closes: pd.Series, period: int) -> float:
    """Simple moving average of the last `period` bars."""
    return round(float(closes.iloc[-period:].mean()), 4)


def compute_atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = config.ATR_LOOKBACK,
) -> float:
    """
    Average True Range over `period` bars.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    """
    prev_closes = closes.shift(1)
    tr = pd.concat(
        [highs - lows, (highs - prev_closes).abs(), (lows - prev_closes).abs()],
        axis=1,
    ).max(axis=1)
    return round(float(tr.iloc[-period:].mean()), 4)


def _signal_mult(pir: float) -> float:
    if pir < config.PIR_VERY_DEEP_THRESHOLD:
        return config.SIGNAL_MULT_DEEP_CRASH
    if pir < config.PIR_STRONG_DIP_THRESHOLD:
        return config.SIGNAL_MULT_STRONG_DIP
    return config.SIGNAL_MULT_STANDARD


def _vol_factor(atr_pct: float) -> float:
    if atr_pct < config.ATR_LOW_THRESHOLD:
        return config.VOL_FACTOR_LOW
    if atr_pct > config.ATR_HIGH_THRESHOLD:
        return config.VOL_FACTOR_HIGH
    return config.VOL_FACTOR_NORMAL


def evaluate_buy_signal(
    ticker: str,
    closes: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    current_price: float,
) -> tuple[Optional[BuySignal], str]:
    valid, reason = validate_data_quality(closes, ticker)
    if not valid:
        return None, reason

    pir, _, _ = compute_pir(closes, current_price)
    sma_50 = compute_sma(closes, config.TREND_FILTER_SMA_PERIOD)
    atr = compute_atr(highs, lows, closes, config.ATR_LOOKBACK)
    atr_pct = round(atr / current_price, 4)

    is_deep_dip = pir < config.PIR_DEEP_DIP_THRESHOLD
    standard_dip = (pir < config.PIR_STANDARD_BUY_THRESHOLD) and (current_price > sma_50)

    if not (is_deep_dip or standard_dip):
        return None, (
            f"{ticker}: no signal — pir={pir:.4f}, "
            f"price={'above' if current_price > sma_50 else 'below'} sma50={sma_50:.2f}"
        )

    return BuySignal(
        ticker=ticker,
        price=round(current_price, 4),
        pir=pir,
        sma_50=sma_50,
        atr_pct=atr_pct,
        signal_mult=_signal_mult(pir),
        vol_factor=_vol_factor(atr_pct),
        is_deep_dip=is_deep_dip,
    ), "buy"


def evaluate_sell(
    ticker: str,
    avg_entry_price: float,
    current_price: float,
    trailing_stop_active: bool,
    trailing_stop_peak: float | None,
) -> SellDecision:
    """
    Evaluate sell rules in priority order:
    1. Hard stop loss (−10%).
    2. Trailing stop trigger (if active, 10% from peak).
    3. Trailing stop activation signal (if not active, +20% unrealized).
    4. Hold.
    Note: peak ratcheting is handled by main.py via risk.update_trailing_stop_peak
    BEFORE calling this function on each run.
    """
    unrealized_pct = round((current_price - avg_entry_price) / avg_entry_price, 4)

    if unrealized_pct <= config.STOP_LOSS_PCT:
        return SellDecision(
            ticker=ticker,
            action="sell_stop_loss",
            unrealized_pct=unrealized_pct,
            reason=f"stop loss hit ({unrealized_pct:.1%} ≤ {config.STOP_LOSS_PCT:.0%})",
        )

    if trailing_stop_active and trailing_stop_peak is not None:
        drawdown = round((trailing_stop_peak - current_price) / trailing_stop_peak, 4)
        if drawdown >= config.TRAILING_STOP_TRAIL_PCT:
            return SellDecision(
                ticker=ticker,
                action="sell_trailing",
                unrealized_pct=unrealized_pct,
                reason=(
                    f"trailing stop triggered: {drawdown:.1%} from peak "
                    f"${trailing_stop_peak:.2f}"
                ),
            )

    if not trailing_stop_active and unrealized_pct >= config.TRAILING_STOP_ACTIVATION_PCT:
        return SellDecision(
            ticker=ticker,
            action="activate_trailing",
            unrealized_pct=unrealized_pct,
            reason=f"trailing stop activated at +{unrealized_pct:.1%}",
        )

    return SellDecision(
        ticker=ticker,
        action="hold",
        unrealized_pct=unrealized_pct,
        reason="hold",
    )


def compute_regime(spy_closes: pd.Series) -> RegimeInfo:
    """
    Classify current market regime from SPY daily closes.
    Requires ≥ REGIME_SMA_PERIOD + REGIME_SLOPE_PERIOD + 1 bars (≥ 221).
    """
    current = round(float(spy_closes.iloc[-1]), 4)
    sma_200 = round(float(spy_closes.iloc[-config.REGIME_SMA_PERIOD:].mean()), 4)
    spy_20d_ago = float(spy_closes.iloc[-1 - config.REGIME_SLOPE_PERIOD])
    slope_pct = round((current - spy_20d_ago) / spy_20d_ago, 4)
    above_sma = current > sma_200

    if above_sma:
        if slope_pct > config.REGIME_SLOPE_RISING_THRESHOLD:
            mult, direction = config.REGIME_MULT_BULL, "rising"
            desc = "SPY above 200SMA, slope rising → bull"
        elif slope_pct < config.REGIME_SLOPE_FALLING_THRESHOLD:
            mult, direction = config.REGIME_MULT_TOPPING, "falling"
            desc = "SPY above 200SMA, slope falling → topping"
        else:
            mult, direction = config.REGIME_MULT_LATE_CYCLE, "flat"
            desc = "SPY above 200SMA, slope flat → late-cycle"
    else:
        if slope_pct > config.REGIME_SLOPE_RISING_THRESHOLD:
            mult, direction = config.REGIME_MULT_RECOVERY, "rising"
            desc = "SPY below 200SMA, slope rising → recovery"
        else:
            mult, direction = config.REGIME_MULT_BEAR, "flat_or_falling"
            desc = "SPY below 200SMA, slope flat/falling → bear/crisis"

    log.info(
        f"REGIME: SPY=${current} | 200SMA=${sma_200} "
        f"({'above' if above_sma else 'below'}) | slope={slope_pct:+.2%}/20d "
        f"({direction}) | regime_mult={mult}"
    )
    return RegimeInfo(
        spy_price=current,
        sma_200=sma_200,
        slope_20d_pct=slope_pct,
        above_sma=above_sma,
        slope_direction=direction,
        regime_mult=mult,
        description=desc,
    )
