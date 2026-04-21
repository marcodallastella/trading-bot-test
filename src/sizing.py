# src/sizing.py
from src import config
from src.logger import get_logger
from src.models import BuySignal, SizingResult

log = get_logger(__name__)


def get_equity_tier(total_equity: float) -> dict:
    """Return the matching EQUITY_TIER dict (first where min <= equity < max)."""
    for tier in config.EQUITY_TIERS:
        if tier["min"] <= total_equity < tier["max"]:
            return tier
    return config.EQUITY_TIERS[-1]


def compute_kelly_base(total_equity: float) -> float:
    """
    BASE_KELLY_FRACTION, clamped first to the active tier's [kelly_min, kelly_max],
    then to the global hard bounds [BASE_KELLY_FRACTION_MIN, BASE_KELLY_FRACTION_MAX].
    Global bounds take precedence — they are non-negotiable. If BASE_KELLY_FRACTION
    is strictly below the global minimum, the global minimum is returned directly
    (tier floor does not raise it further). If BASE_KELLY_FRACTION is within or above
    global bounds, tier bounds apply normally, then global bounds are re-applied.
    """
    tier = get_equity_tier(total_equity)
    # If BASE is below the global floor, the global floor is the hard answer —
    # the tier minimum cannot override it upward.
    if config.BASE_KELLY_FRACTION < config.BASE_KELLY_FRACTION_MIN:
        return config.BASE_KELLY_FRACTION_MIN
    # Otherwise: tier clamp first, then global clamp (global max takes precedence over tier max)
    clamped = max(tier["kelly_min"], min(tier["kelly_max"], config.BASE_KELLY_FRACTION))
    clamped = max(config.BASE_KELLY_FRACTION_MIN, min(config.BASE_KELLY_FRACTION_MAX, clamped))
    return round(clamped, 4)


def compute_position_size(
    ticker: str,
    total_equity: float,
    signal: BuySignal,
    regime_mult: float,
    current_ticker_exposure_usd: float,
    current_total_deployed_usd: float,
    current_position_count: int,
) -> SizingResult:
    """
    Full Kelly sizing pipeline. Returns SizingResult with skip=True if the order
    should not be placed. Caps applied in order: position count → per-ticker →
    total deployment → minimum notional.
    """
    tier = get_equity_tier(total_equity)
    kelly_base = compute_kelly_base(total_equity)

    raw_pct = round(kelly_base * signal.signal_mult * signal.vol_factor * regime_mult, 4)
    raw_usd = round(total_equity * raw_pct, 4)

    capped = False
    cap_reason = ""

    def _skip(reason: str) -> SizingResult:
        return SizingResult(
            ticker=ticker,
            kelly_base=kelly_base,
            signal_mult=signal.signal_mult,
            vol_factor=signal.vol_factor,
            regime_mult=regime_mult,
            position_size_pct=raw_pct,
            buy_amount_usd=0.0,
            capped=False,
            cap_reason="",
            skip=True,
            skip_reason=reason,
        )

    # 1. Position count limit
    if current_position_count >= tier["max_positions"]:
        return _skip(
            f"max positions reached ({current_position_count}/{tier['max_positions']})"
        )

    # 2. Per-ticker exposure cap
    ticker_cap_usd = round(total_equity * tier["per_ticker_cap"], 4)
    remaining_for_ticker = round(ticker_cap_usd - current_ticker_exposure_usd, 4)
    if remaining_for_ticker <= 0:
        return _skip(
            f"{ticker} per-ticker cap {tier['per_ticker_cap']:.0%} "
            f"(${ticker_cap_usd:.2f}) already reached"
        )

    # 3. Total deployment cap
    max_deployable = round(
        total_equity * config.MAX_TOTAL_DEPLOYMENT_PCT - current_total_deployed_usd, 4
    )
    if max_deployable <= 0:
        return _skip(
            f"total deployment cap {config.MAX_TOTAL_DEPLOYMENT_PCT:.0%} reached"
        )

    # Apply caps to buy amount
    buy_usd = raw_usd
    if buy_usd > remaining_for_ticker:
        buy_usd = remaining_for_ticker
        capped = True
        cap_reason = f"per-ticker cap ${ticker_cap_usd:.2f}"
    if buy_usd > max_deployable:
        buy_usd = max_deployable
        capped = True
        cap_reason = f"total deployment cap {config.MAX_TOTAL_DEPLOYMENT_PCT:.0%}"

    buy_usd = round(buy_usd, 4)

    # 4. Minimum notional
    if buy_usd < config.MIN_ORDER_NOTIONAL:
        return _skip(
            f"buy amount ${buy_usd:.2f} below minimum ${config.MIN_ORDER_NOTIONAL:.2f}"
        )

    log.info(
        f"{ticker}: kelly={kelly_base:.4f} × sig={signal.signal_mult} × "
        f"vol={signal.vol_factor} × regime={regime_mult} = {raw_pct:.4f} → "
        f"${buy_usd:.2f}"
        + (f" [capped: {cap_reason}]" if capped else "")
    )
    return SizingResult(
        ticker=ticker,
        kelly_base=kelly_base,
        signal_mult=signal.signal_mult,
        vol_factor=signal.vol_factor,
        regime_mult=regime_mult,
        position_size_pct=raw_pct,
        buy_amount_usd=buy_usd,
        capped=capped,
        cap_reason=cap_reason,
        skip=False,
        skip_reason="",
    )
