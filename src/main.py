# src/main.py
import os
import sys
from dataclasses import replace
from datetime import date, datetime

import pandas as pd

from src import config
from src.alpaca_client import AlpacaClient
from src.logger import get_logger
from src.logbook import append_decisions
from src.models import BuySignal, LogbookRow, Position, State, TrailingStop
from src.notify import format_summary, send_daily_summary
from src.risk import (
    activate_trailing_stop, add_wash_sale, clean_expired_wash_sales,
    compute_drawdown, deactivate_trailing_stop, is_kill_switch_active,
    is_wash_sale_blocked, should_activate_trailing_stop,
    update_drawdown_halt, update_rolling_peak, update_trailing_stop_peak,
)
from src.sizing import compute_kelly_base, compute_position_size, get_equity_tier
from src.state import load_state, save_state
from src.strategy import compute_regime, evaluate_buy_signal, evaluate_sell

log = get_logger(__name__)


def _spread_bps(quote: dict) -> float:
    mid = (quote["ask"] + quote["bid"]) / 2
    if mid <= 0:
        return 9999.0
    return round((quote["ask"] - quote["bid"]) / mid * 10_000, 1)


def _logbook_row(
    ts: str, ticker: str, *,
    price: float | None = None,
    sma_50: float | None = None,
    pir: float | None = None,
    atr_pct: float | None = None,
    signal: BuySignal | None = None,
    regime_mult: float | None = None,
    kelly_base: float | None = None,
    final_pct: float | None = None,
    action: str = "SKIP",
    order_notional: float | None = None,
    fill_price: float | None = None,
    reason: str = "",
) -> LogbookRow:
    def f(v: float | None) -> str:
        return "" if v is None else str(round(v, 4))
    return LogbookRow(
        timestamp=ts, ticker=ticker,
        price=f(price), sma_50=f(sma_50),
        pir=f(pir), atr_pct=f(atr_pct),
        signal_mult=f(signal.signal_mult) if signal else "",
        vol_factor=f(signal.vol_factor) if signal else "",
        regime_mult=f(regime_mult), kelly_base=f(kelly_base),
        final_pct=f(final_pct), action=action,
        order_notional=f(order_notional), fill_price=f(fill_price),
        reason=reason,
    )


def run() -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    today = ts[:10]

    # ── environment ───────────────────────────────────────────────────────────
    is_paper    = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
    is_dry_run  = os.environ.get("DRY_RUN",      "true").lower() != "false"
    live_ready  = os.environ.get("LIVE_READY",   "false").lower() == "true"

    if not is_paper and not is_dry_run and not live_ready:
        log.critical(
            "BLOCKED: live trading requires LIVE_READY=true. "
            "Complete 30-day paper burn-in first."
        )
        sys.exit(1)

    log.info(
        f"=== RUN START | mode={'PAPER' if is_paper else 'LIVE'} "
        f"| dry_run={is_dry_run} | kill={os.environ.get('KILL_SWITCH', '0')} ==="
    )

    # ── state ─────────────────────────────────────────────────────────────────
    gist_id  = os.environ["STATE_GIST_ID"]
    gh_pat   = os.environ["GH_PAT"]
    state, sell_only = load_state(gist_id, gh_pat)
    if sell_only:
        log.critical("Running in SELL-ONLY mode — no buys will be placed")

    # ── Alpaca client + clock ─────────────────────────────────────────────────
    client = AlpacaClient(is_paper=is_paper)
    clock = client.get_clock()
    if not clock["is_open"]:
        log.info(f"Market CLOSED (next open: {clock['next_open']}). Exiting.")
        return

    # ── account + equity tier ─────────────────────────────────────────────────
    account = client.get_account()
    total_equity = account["equity"]
    tier = get_equity_tier(total_equity)
    log.info(
        f"Account: cash=${account['cash']:.2f} | equity=${total_equity:.2f} "
        f"| tier=${tier['min']}–${tier['max']}"
    )

    # ── rolling peak + drawdown ───────────────────────────────────────────────
    state = update_rolling_peak(state, total_equity, today)
    state = update_drawdown_halt(state, total_equity)
    drawdown = compute_drawdown(state, total_equity)
    log.info(
        f"Drawdown: {drawdown:.1%} — "
        f"{'HALT ACTIVE' if state.drawdown_halt_active else f'OK (halt {config.DRAWDOWN_HALT_THRESHOLD:.0%})'}"
    )

    # ── regime ────────────────────────────────────────────────────────────────
    spy_bars = client.get_spy_bars(
        limit=config.REGIME_SMA_PERIOD + config.REGIME_SLOPE_PERIOD + 5
    )
    regime = compute_regime(spy_bars["close"])

    kelly_base_active = round(
        compute_kelly_base(total_equity) * regime.regime_mult, 4
    )
    log.info(
        f"REGIME: {regime.description} | regime_mult={regime.regime_mult} "
        f"| active_kelly={kelly_base_active:.4f}"
    )

    # ── positions + sell loop ─────────────────────────────────────────────────
    positions: list[Position] = client.get_positions()
    actions_log: list[str] = []
    decisions: list[LogbookRow] = []

    log.info("--- Evaluating sells ---")
    for pos in positions:
        if pos.ticker in state.trailing_stops_active:
            state = update_trailing_stop_peak(state, pos.ticker, pos.current_price)

        ts_active = pos.ticker in state.trailing_stops_active
        ts_peak   = (state.trailing_stops_active[pos.ticker].peak_price
                     if ts_active else None)
        sell_dec  = evaluate_sell(
            pos.ticker, pos.avg_entry_price, pos.current_price, ts_active, ts_peak
        )

        if sell_dec.action in ("sell_stop_loss", "sell_trailing"):
            quote = client.get_latest_quote(pos.ticker)
            spread = _spread_bps(quote)
            if spread > config.MAX_SPREAD_BPS_SELL:
                log.warning(
                    f"{pos.ticker}: SKIP sell — spread {spread:.0f}bps "
                    f"> {config.MAX_SPREAD_BPS_SELL}bps threshold"
                )
                decisions.append(_logbook_row(
                    ts, pos.ticker, price=pos.current_price,
                    action="SKIP", reason=f"wide spread {spread:.0f}bps on sell",
                ))
                continue

            if not is_dry_run:
                try:
                    client.close_position(pos.ticker)
                except Exception as exc:
                    log.error(f"{pos.ticker}: sell order failed: {exc}")
                    continue

            if sell_dec.action == "sell_stop_loss":
                state = add_wash_sale(state, pos.ticker, today)
            state = deactivate_trailing_stop(state, pos.ticker)
            state.ticker_last_buy_date.pop(pos.ticker, None)

            label = ("SELL_STOP_LOSS" if sell_dec.action == "sell_stop_loss"
                     else "SELL_TRAILING")
            actions_log.append(
                f"{label} {pos.ticker} @ ${pos.current_price:.2f} ({sell_dec.reason})"
            )
            decisions.append(_logbook_row(
                ts, pos.ticker, price=pos.current_price,
                action=label, reason=sell_dec.reason,
            ))

        elif sell_dec.action == "activate_trailing":
            state = activate_trailing_stop(
                state, pos.ticker, pos.avg_entry_price,
                pos.current_price, today,
            )
            log.info(
                f"{pos.ticker}: TRAILING STOP ACTIVATED at "
                f"+{sell_dec.unrealized_pct:.1%}"
            )
            decisions.append(_logbook_row(
                ts, pos.ticker, price=pos.current_price,
                action="HOLD", reason=sell_dec.reason,
            ))

        else:
            log.info(
                f"{pos.ticker}: {sell_dec.unrealized_pct:+.1%} — HOLD"
                + (f" (trailing active)" if ts_active else "")
            )
            decisions.append(_logbook_row(
                ts, pos.ticker, price=pos.current_price,
                action="HOLD", reason=sell_dec.reason,
            ))

    # ── buy loop ──────────────────────────────────────────────────────────────
    current_position_count = len(positions)
    current_total_deployed = sum(p.market_value for p in positions)

    log.info("--- Evaluating buys ---")
    for ticker in sorted(config.UNIVERSE):

        if is_kill_switch_active():
            decisions.append(_logbook_row(ts, ticker, action="SKIP",
                                          reason="kill switch active"))
            continue

        if state.drawdown_halt_active:
            decisions.append(_logbook_row(ts, ticker, action="SKIP",
                                          reason="drawdown halt active"))
            continue

        if sell_only:
            decisions.append(_logbook_row(ts, ticker, action="SKIP",
                                          reason="sell-only mode (state load failed)"))
            continue

        if current_position_count >= tier["max_positions"]:
            decisions.append(_logbook_row(
                ts, ticker, action="SKIP",
                reason=f"max positions {current_position_count}/{tier['max_positions']}",
            ))
            continue

        if is_wash_sale_blocked(state, ticker, today):
            log.info(f"{ticker}: SKIP — wash-sale blacklist")
            decisions.append(_logbook_row(ts, ticker, action="SKIP",
                                          reason="wash-sale blacklist"))
            continue

        last_buy_str = state.ticker_last_buy_date.get(ticker)
        if last_buy_str:
            days_since = (date.fromisoformat(today) -
                          date.fromisoformat(last_buy_str)).days
            if days_since < config.PER_TICKER_COOLDOWN_DAYS:
                remaining = config.PER_TICKER_COOLDOWN_DAYS - days_since
                log.info(f"{ticker}: SKIP — cooldown ({remaining}d remaining)")
                decisions.append(_logbook_row(
                    ts, ticker, action="SKIP",
                    reason=f"cooldown {remaining}d remaining",
                ))
                continue

        try:
            bars  = client.get_daily_bars(ticker, limit=config.LOOKBACK_DAYS + 5)
            quote = client.get_latest_quote(ticker)
        except Exception as exc:
            log.error(f"{ticker}: SKIP — API error fetching bars/quote: {exc}")
            decisions.append(_logbook_row(ts, ticker, action="SKIP",
                                          reason=f"API error: {exc}"))
            continue

        current_price = quote["last"]
        spread = _spread_bps(quote)
        if spread > config.MAX_SPREAD_BPS_BUY:
            log.info(
                f"{ticker}: SKIP — spread {spread:.0f}bps "
                f"> {config.MAX_SPREAD_BPS_BUY}bps threshold"
            )
            decisions.append(_logbook_row(
                ts, ticker, price=current_price,
                action="SKIP", reason=f"spread {spread:.0f}bps > threshold",
            ))
            continue

        signal, sig_reason = evaluate_buy_signal(
            ticker, bars["close"], bars["high"], bars["low"], current_price,
        )
        if signal is None:
            log.info(f"{ticker}: {sig_reason}")
            decisions.append(_logbook_row(
                ts, ticker, price=current_price,
                action="SKIP", reason=sig_reason,
            ))
            continue

        current_ticker_exposure = next(
            (p.market_value for p in positions if p.ticker == ticker), 0.0
        )
        sizing = compute_position_size(
            ticker=ticker, total_equity=total_equity, signal=signal,
            regime_mult=regime.regime_mult,
            current_ticker_exposure_usd=current_ticker_exposure,
            current_total_deployed_usd=current_total_deployed,
            current_position_count=current_position_count,
        )
        if sizing.skip:
            log.info(f"{ticker}: SKIP — {sizing.skip_reason}")
            decisions.append(_logbook_row(
                ts, ticker, price=current_price, signal=signal,
                regime_mult=regime.regime_mult,
                kelly_base=sizing.kelly_base,
                action="SKIP", reason=sizing.skip_reason,
            ))
            continue

        if sizing.buy_amount_usd > account["buying_power"]:
            log.warning(
                f"{ticker}: SKIP — insufficient buying power "
                f"(need ${sizing.buy_amount_usd:.2f}, "
                f"have ${account['buying_power']:.2f})"
            )
            decisions.append(_logbook_row(
                ts, ticker, price=current_price, action="SKIP",
                reason="insufficient buying power",
            ))
            continue

        log.info(
            f"{ticker}: price=${current_price:.2f} PIR={signal.pir:.4f} "
            f"ATR%={signal.atr_pct:.4f} | "
            f"sizing: {sizing.kelly_base:.4f}×{signal.signal_mult}×"
            f"{signal.vol_factor}×{regime.regime_mult} "
            f"= {sizing.position_size_pct:.4f} → ${sizing.buy_amount_usd:.2f} | "
            f"spread={spread:.0f}bps | "
            f"{'DRY_RUN (not placed)' if is_dry_run else 'BUY'}"
        )

        fill_price = None
        if not is_dry_run:
            try:
                order = client.submit_market_order(
                    ticker, sizing.buy_amount_usd, "buy"
                )
                fill_price = order.get("fill_price") if order else None
            except Exception as exc:
                log.error(f"{ticker}: buy order failed (skipping): {exc}")
                decisions.append(_logbook_row(
                    ts, ticker, price=current_price,
                    action="SKIP", reason=f"order failed: {exc}",
                ))
                continue

        state.ticker_last_buy_date[ticker] = today
        current_position_count += 1
        current_total_deployed += sizing.buy_amount_usd

        actions_log.append(
            f"BUY {ticker} ${sizing.buy_amount_usd:.2f} @ ${current_price:.2f} "
            f"(PIR={signal.pir:.3f}"
            + (" deep dip" if signal.is_deep_dip else "")
            + f", regime={regime.regime_mult})"
        )
        decisions.append(_logbook_row(
            ts, ticker,
            price=current_price, sma_50=signal.sma_50,
            pir=signal.pir, atr_pct=signal.atr_pct,
            signal=signal, regime_mult=regime.regime_mult,
            kelly_base=sizing.kelly_base,
            final_pct=sizing.position_size_pct,
            action="BUY",
            order_notional=sizing.buy_amount_usd,
            fill_price=fill_price,
            reason=f"pir={signal.pir:.4f} "
                   + ("deep dip override" if signal.is_deep_dip else "standard dip"),
        ))

    # ── end-of-run ────────────────────────────────────────────────────────────
    state = clean_expired_wash_sales(state, today)
    state = replace(state, last_run_timestamp=ts)
    save_state(state, gist_id, gh_pat)

    decisions_gist_id = os.environ.get("DECISIONS_GIST_ID", "")
    if decisions_gist_id:
        append_decisions(decisions, decisions_gist_id, gh_pat)

    summary = format_summary(
        run_timestamp=ts,
        is_paper=is_paper,
        is_dry_run=is_dry_run,
        equity=total_equity,
        starting_equity=config.BACKTEST_STARTING_EQUITY,
        rolling_peak_equity=state.rolling_peak_equity,
        rolling_peak_timestamp=state.rolling_peak_timestamp,
        drawdown_pct=drawdown,
        halt_active=state.drawdown_halt_active,
        regime=regime,
        kelly_base=compute_kelly_base(total_equity),
        positions=positions,
        trailing_stops=state.trailing_stops_active,
        actions_log=actions_log,
        cumulative_realized_pnl=state.cumulative_realized_pnl,
        wash_sale_count=len(state.wash_sale_blacklist),
    )
    webhook_url = os.environ.get("NOTIFY_WEBHOOK_URL")
    send_daily_summary(summary, webhook_url)

    log.info("=== RUN END ===")


def main() -> None:
    try:
        run()
    except SystemExit:
        raise
    except Exception as exc:
        get_logger(__name__).critical(
            f"Unhandled exception — aborting run: {exc}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
