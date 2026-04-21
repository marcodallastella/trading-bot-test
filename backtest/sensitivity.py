from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from backtest.run_backtest import run_backtest, BacktestResult
from backtest.metrics import BacktestMetrics
from src import config


@dataclass
class SweepConfig:
    stop_loss_pcts:     list[float] = field(default_factory=lambda: [-0.08, -0.10, -0.12])
    pir_thresholds:     list[float] = field(default_factory=lambda: [0.20, 0.25, 0.30])
    lookback_days:      list[int]   = field(default_factory=lambda: [45, 60, 90])
    regime_bear_mults:  list[float] = field(default_factory=lambda: [0.20, 0.30, 0.40])


@dataclass
class SweepRow:
    stop_loss_pct:    float
    pir_threshold:    float
    lookback_days:    int
    regime_bear_mult: float
    total_return_pct: float
    sharpe_ratio:     float
    max_drawdown_pct: float
    kelly_fraction:   float
    kelly_recommendation: float
    total_trades:     int


def run_sweep(
    universe: list[str],
    start: date,
    end: date,
    sweep_config: SweepConfig | None = None,
    starting_equity: float = config.BACKTEST_STARTING_EQUITY,
    api_key: str = "",
    secret_key: str = "",
    output_dir: str = "backtest_output/sweep",
    spy_ticker: str = "SPY",
) -> list[SweepRow]:
    if sweep_config is None:
        sweep_config = SweepConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        sweep_config.stop_loss_pcts,
        sweep_config.pir_thresholds,
        sweep_config.lookback_days,
        sweep_config.regime_bear_mults,
    ))

    rows: list[SweepRow] = []

    for stop_loss, pir_thresh, lookback, bear_mult in combos:
        _orig_stop      = config.STOP_LOSS_PCT
        _orig_pir       = config.PIR_STANDARD_BUY_THRESHOLD
        _orig_lookback  = config.LOOKBACK_DAYS
        _orig_bear_mult = config.REGIME_MULT_BEAR

        config.STOP_LOSS_PCT               = stop_loss
        config.PIR_STANDARD_BUY_THRESHOLD  = pir_thresh
        config.LOOKBACK_DAYS               = lookback
        config.REGIME_MULT_BEAR            = bear_mult

        try:
            result = run_backtest(
                universe=universe,
                start=start,
                end=end,
                starting_equity=starting_equity,
                api_key=api_key,
                secret_key=secret_key,
                output_dir=str(output_path / f"sl{stop_loss}_pir{pir_thresh}_lb{lookback}_bm{bear_mult}"),
                spy_ticker=spy_ticker,
            )
            m = result.metrics
            rows.append(SweepRow(
                stop_loss_pct=stop_loss,
                pir_threshold=pir_thresh,
                lookback_days=lookback,
                regime_bear_mult=bear_mult,
                total_return_pct=m.total_return_pct,
                sharpe_ratio=m.sharpe_ratio,
                max_drawdown_pct=m.max_drawdown_pct,
                kelly_fraction=m.kelly_fraction,
                kelly_recommendation=result.kelly_recommendation,
                total_trades=m.total_trades,
            ))
        except Exception as exc:
            print(f"[sweep] combo ({stop_loss}, {pir_thresh}, {lookback}, {bear_mult}) failed: {exc}")
        finally:
            config.STOP_LOSS_PCT               = _orig_stop
            config.PIR_STANDARD_BUY_THRESHOLD  = _orig_pir
            config.LOOKBACK_DAYS               = _orig_lookback
            config.REGIME_MULT_BEAR            = _orig_bear_mult

    rows.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    csv_path = output_path / "sensitivity.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].__dataclass_fields__.keys()))
            writer.writeheader()
            writer.writerows([r.__dict__ for r in rows])

    return rows
