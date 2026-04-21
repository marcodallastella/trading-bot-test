from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.models import State, Position
from src.strategy import (
    validate_data_quality,
    compute_pir,
    compute_sma,
    compute_atr,
    evaluate_buy_signal,
    evaluate_sell,
    compute_regime,
)
from src.sizing import compute_position_size
from src import risk
from backtest.data import fetch_bars
from backtest.metrics import compute_metrics, BacktestMetrics


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_pnls: list[float]
    metrics: BacktestMetrics
    kelly_recommendation: float
    decisions_log: list[dict]
    equity_curve_path: Optional[str]


def _initial_state(starting_equity: float) -> State:
    return State(
        rolling_peak_equity=starting_equity,
        rolling_peak_timestamp=date.today().isoformat(),
        last_run_timestamp="",
        ticker_last_buy_date={},
        wash_sale_blacklist={},
        trailing_stops_active={},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=False,
    )


def _get_current_price(bars_df: pd.DataFrame, as_of: date) -> Optional[float]:
    if hasattr(as_of, "date"):
        as_of = as_of.date()
    matches = bars_df[bars_df.index == as_of]
    if matches.empty:
        return None
    return float(matches["close"].iloc[-1])


def run_backtest(
    universe: list[str],
    start: date,
    end: date,
    starting_equity: float = config.BACKTEST_STARTING_EQUITY,
    api_key: str = "",
    secret_key: str = "",
    output_dir: str = "backtest_output",
    spy_ticker: str = "SPY",
) -> BacktestResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetch_start = start - timedelta(days=max(config.LOOKBACK_DAYS * 2, 120))

    all_bars: dict[str, pd.DataFrame] = {}
    spy_bars_full: pd.DataFrame | None = None

    tickers_to_fetch = list(set(universe) | {spy_ticker})
    for ticker in tickers_to_fetch:
        try:
            df = fetch_bars(ticker, fetch_start, end, api_key=api_key, secret_key=secret_key)
            if ticker == spy_ticker:
                spy_bars_full = df
            else:
                all_bars[ticker] = df
        except ValueError:
            pass

    if spy_bars_full is None or spy_bars_full.empty:
        raise ValueError(f"Could not fetch SPY bars — required for regime detection")

    spy_index = pd.to_datetime(spy_bars_full.index).normalize()
    trading_days = [d.date() for d in spy_index if start <= d.date() <= end]

    equity = starting_equity
    cash = starting_equity
    positions: dict[str, Position] = {}
    state = _initial_state(starting_equity)
    equity_series: list[tuple[date, float]] = []
    trade_pnls: list[float] = []
    decisions: list[dict] = []

    slippage_mult_buy  = 1 + config.BACKTEST_SLIPPAGE_BPS / 10_000
    slippage_mult_sell = 1 - config.BACKTEST_SLIPPAGE_BPS / 10_000

    for sim_date in trading_days:
        spy_closes_up_to = spy_bars_full[
            pd.to_datetime(spy_bars_full.index).normalize() <= pd.Timestamp(sim_date)
        ]["close"]
        if len(spy_closes_up_to) < config.REGIME_SMA_PERIOD:
            equity_series.append((sim_date, equity))
            continue
        regime_info = compute_regime(spy_closes_up_to)

        state = risk.update_rolling_peak(state, equity, sim_date.isoformat())
        state = risk.update_drawdown_halt(state, equity)

        # sell pass
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bars_df = all_bars.get(ticker)
            if bars_df is None:
                continue

            current_price = _get_current_price(bars_df, sim_date)
            if current_price is None:
                continue

            ts_info = state.trailing_stops_active.get(ticker)
            ts_active = ts_info is not None
            ts_peak = ts_info.peak_price if ts_active else None

            if ts_active:
                state = risk.update_trailing_stop_peak(state, ticker, current_price)

            sell_decision = evaluate_sell(
                ticker,
                pos.avg_entry_price,
                current_price,
                trailing_stop_active=ts_active,
                trailing_stop_peak=ts_peak,
            )

            if sell_decision.action in ("sell_stop_loss", "sell_trailing"):
                fill_price = current_price * slippage_mult_sell
                proceeds = pos.qty * fill_price
                pnl = proceeds - pos.qty * pos.avg_entry_price
                cash += proceeds
                trade_pnls.append(pnl)

                if pnl < 0:
                    state = risk.add_wash_sale(state, ticker, sim_date.isoformat())

                if ts_active:
                    state = risk.deactivate_trailing_stop(state, ticker)

                decisions.append({
                    "date": sim_date.isoformat(), "ticker": ticker,
                    "action": sell_decision.action, "price": round(fill_price, 4),
                    "pnl": round(pnl, 4),
                })
                del positions[ticker]

            elif sell_decision.action == "activate_trailing":
                state = risk.activate_trailing_stop(
                    state, ticker, pos.avg_entry_price, current_price, sim_date.isoformat()
                )

        # recalculate equity
        mark_to_market = sum(
            (p.qty * (_get_current_price(all_bars[t], sim_date) or p.avg_entry_price))
            for t, p in positions.items()
            if t in all_bars
        )
        equity = cash + mark_to_market

        # buy pass
        state = risk.clean_expired_wash_sales(state, sim_date.isoformat())

        if not state.drawdown_halt_active:
            for ticker in universe:
                bars_df = all_bars.get(ticker)
                if bars_df is None:
                    continue

                bars_up_to = bars_df[
                    pd.to_datetime(bars_df.index).normalize() <= pd.Timestamp(sim_date)
                ]
                if len(bars_up_to) < config.LOOKBACK_DAYS:
                    continue

                closes = bars_up_to["close"].iloc[-config.LOOKBACK_DAYS:]
                highs  = bars_up_to["high"].iloc[-config.LOOKBACK_DAYS:]
                lows   = bars_up_to["low"].iloc[-config.LOOKBACK_DAYS:]
                current_price = float(closes.iloc[-1])

                ok, reason = validate_data_quality(closes, ticker)
                if not ok:
                    continue

                if risk.is_wash_sale_blocked(state, ticker, sim_date.isoformat()):
                    continue

                last_buy = state.ticker_last_buy_date.get(ticker)
                if last_buy:
                    days_since = (sim_date - date.fromisoformat(last_buy)).days
                    if days_since < config.PER_TICKER_COOLDOWN_DAYS:
                        continue

                buy_signal, sig_reason = evaluate_buy_signal(ticker, closes, highs, lows, current_price)
                if buy_signal is None:
                    continue

                current_exposure = (
                    positions[ticker].qty * current_price if ticker in positions else 0.0
                )
                current_deployed = sum(
                    p.qty * (_get_current_price(all_bars[t], sim_date) or p.avg_entry_price)
                    for t, p in positions.items() if t in all_bars
                )

                sizing = compute_position_size(
                    ticker=ticker,
                    total_equity=equity,
                    signal=buy_signal,
                    regime_mult=regime_info.regime_mult,
                    current_ticker_exposure_usd=current_exposure,
                    current_total_deployed_usd=current_deployed,
                    current_position_count=len(positions),
                )

                if sizing.skip or sizing.buy_amount_usd < config.MIN_ORDER_NOTIONAL:
                    continue

                fill_price = current_price * slippage_mult_buy
                if fill_price * 0.01 > cash:
                    continue

                buy_amount = min(sizing.buy_amount_usd, cash)
                quantity = buy_amount / fill_price
                cash -= buy_amount

                if ticker in positions:
                    existing = positions[ticker]
                    new_qty = existing.qty + quantity
                    new_avg = (existing.qty * existing.avg_entry_price + quantity * fill_price) / new_qty
                    positions[ticker] = Position(ticker=ticker, qty=new_qty, avg_entry_price=new_avg, current_price=fill_price)
                else:
                    positions[ticker] = Position(ticker=ticker, qty=quantity, avg_entry_price=fill_price, current_price=fill_price)

                state = dataclasses.replace(
                    state,
                    ticker_last_buy_date={**state.ticker_last_buy_date, ticker: sim_date.isoformat()}
                )

                decisions.append({
                    "date": sim_date.isoformat(), "ticker": ticker,
                    "action": "buy", "price": round(fill_price, 4),
                    "notional": round(buy_amount, 4),
                })

        # mark equity end of day
        mark_to_market = sum(
            (p.qty * (_get_current_price(all_bars[t], sim_date) or p.avg_entry_price))
            for t, p in positions.items()
            if t in all_bars
        )
        equity = cash + mark_to_market
        equity_series.append((sim_date, equity))

    equity_curve = pd.Series(
        {d: v for d, v in equity_series},
        name="equity",
    )

    metrics = compute_metrics(equity_curve, trade_pnls)

    kelly_rec = max(config.BASE_KELLY_FRACTION_MIN,
                    min(config.BASE_KELLY_FRACTION_MAX, metrics.kelly_fraction * 0.5))

    # save equity curve PNG
    fig, ax = plt.subplots(figsize=(12, 5))
    equity_curve.plot(ax=ax, label="Strategy")

    if spy_bars_full is not None:
        spy_bt = spy_bars_full[
            pd.to_datetime(spy_bars_full.index).normalize().map(lambda d: start <= d.date() <= end)
        ]["close"]
        if len(spy_bt) > 0:
            spy_normalized = spy_bt / spy_bt.iloc[0] * starting_equity
            spy_normalized.plot(ax=ax, label="SPY B&H", linestyle="--", alpha=0.7)

    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path = str(output_path / "equity_curve.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print(f"Total return: {metrics.total_return_pct:.2f}%")
    print(f"Sharpe: {metrics.sharpe_ratio:.4f}")
    print(f"Kelly recommendation: {kelly_rec:.4f}")
    print("=" * 60 + "\n")

    return BacktestResult(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        metrics=metrics,
        kelly_recommendation=kelly_rec,
        decisions_log=decisions,
        equity_curve_path=png_path,
    )
