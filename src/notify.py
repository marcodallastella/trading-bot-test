# src/notify.py
import requests

from src import config
from src.logger import get_logger
from src.models import Position, RegimeInfo, TrailingStop

log = get_logger(__name__)


def format_summary(
    run_timestamp: str,
    is_paper: bool,
    is_dry_run: bool,
    equity: float,
    starting_equity: float,
    rolling_peak_equity: float,
    rolling_peak_timestamp: str,
    drawdown_pct: float,
    halt_active: bool,
    regime: RegimeInfo,
    kelly_base: float,
    positions: list[Position],
    trailing_stops: dict[str, TrailingStop],
    actions_log: list[str],
    cumulative_realized_pnl: float,
    wash_sale_count: int,
) -> str:
    date_str = run_timestamp[:10]
    mode = "PAPER" if is_paper else "LIVE"
    dry_str = "true" if is_dry_run else "false"

    all_time_usd = equity - starting_equity
    all_time_pct = (all_time_usd / starting_equity * 100) if starting_equity > 0 else 0.0

    if halt_active:
        halt_status = "⛔ DRAWDOWN HALT ACTIVE"
    else:
        halt_status = f"OK (halt threshold {config.DRAWDOWN_HALT_THRESHOLD:.0%})"

    active_kelly = round(kelly_base * regime.regime_mult, 4)

    lines = [
        f"[{date_str}] Trading Bot Daily Summary",
        "─" * 37,
        f"Mode: {mode} | DRY_RUN={dry_str}",
        f"Equity: ${equity:.2f} ({all_time_usd:+.2f}, {all_time_pct:+.1f}% all-time)",
        f"Rolling peak: ${rolling_peak_equity:.2f} ({rolling_peak_timestamp})",
        f"Drawdown: {drawdown_pct:.1%} — {halt_status}",
        f"Regime: {regime.description} → regime_mult = {regime.regime_mult}",
        (
            f"Kelly sizing: {kelly_base:.1%} base"
            f" × {regime.regime_mult} regime"
            f" = {active_kelly:.1%} active"
        ),
        "",
        f"Positions ({len(positions)}):",
    ]

    for pos in sorted(positions, key=lambda p: p.ticker):
        trailing_note = ""
        if pos.ticker in trailing_stops:
            ts = trailing_stops[pos.ticker]
            trailing_note = f" 🎯 TRAILING (peak ${ts.peak_price:.2f})"
        lines.append(
            f"  {pos.ticker}: {pos.qty:.3f} sh @ ${pos.avg_entry_price:.2f} avg"
            f" | now ${pos.current_price:.2f}"
            f" | {pos.unrealized_pct:+.1%}{trailing_note}"
        )

    if not positions:
        lines.append("  (no open positions)")

    lines += ["", "Actions today:"]
    if actions_log:
        for action in actions_log:
            lines.append(f"  {action}")
    else:
        lines.append("  (no actions)")

    wash_line = (
        f"Wash-sale blacklist: {wash_sale_count} ticker(s)"
        if wash_sale_count > 0
        else "Wash-sale blacklist: none"
    )
    lines += [
        "",
        f"Realized P&L (cumulative): {cumulative_realized_pnl:+.2f}",
        wash_line,
    ]

    return "\n".join(lines)


def send_daily_summary(summary_text: str, webhook_url: str | None) -> None:
    """
    Always writes to stdout via logger.
    If webhook_url is set, also POSTs to the Discord/Slack endpoint.
    Never raises — a notification failure must not abort the run.
    """
    log.info(summary_text)
    if not webhook_url:
        return
    try:
        resp = requests.post(webhook_url, json={"content": summary_text}, timeout=10)
        resp.raise_for_status()
        log.info("Daily summary sent to webhook")
    except Exception as exc:
        log.error(f"Webhook send failed (non-fatal): {exc}")
