from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backtest.sensitivity import run_sweep, SweepConfig
from backtest.run_backtest import BacktestResult
from backtest.metrics import BacktestMetrics, compute_metrics
import pandas as pd


def _mock_backtest_result() -> BacktestResult:
    equity_curve = pd.Series([100.0, 103.0, 106.0, 110.0])
    trade_pnls = [5.0, -2.0, 8.0]
    metrics = compute_metrics(equity_curve, trade_pnls)
    return BacktestResult(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        metrics=metrics,
        kelly_recommendation=0.08,
        decisions_log=[],
        equity_curve_path=None,
    )


def test_run_sweep_returns_list_of_sweep_rows(tmp_path):
    cfg = SweepConfig(
        stop_loss_pcts=[-0.10],
        pir_thresholds=[0.30],
        lookback_days=[60],
        regime_bear_mults=[0.30],
    )
    with patch("backtest.sensitivity.run_backtest", return_value=_mock_backtest_result()):
        rows = run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=cfg,
            output_dir=str(tmp_path / "sweep"),
        )
    assert len(rows) == 1
    assert rows[0].stop_loss_pct == -0.10
    assert rows[0].pir_threshold == 0.30


def test_run_sweep_generates_all_combinations(tmp_path):
    cfg = SweepConfig(
        stop_loss_pcts=[-0.08, -0.12],
        pir_thresholds=[0.20, 0.30],
        lookback_days=[45, 90],
        regime_bear_mults=[0.20, 0.40],
    )
    with patch("backtest.sensitivity.run_backtest", return_value=_mock_backtest_result()):
        rows = run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=cfg,
            output_dir=str(tmp_path / "sweep"),
        )
    assert len(rows) == 16


def test_run_sweep_results_sorted_by_sharpe_descending(tmp_path):
    cfg = SweepConfig(
        stop_loss_pcts=[-0.08, -0.10],
        pir_thresholds=[0.25],
        lookback_days=[60],
        regime_bear_mults=[0.30],
    )
    call_count = {"n": 0}
    def mock_run(*a, **kw):
        r = _mock_backtest_result()
        sharpe = 1.5 if call_count["n"] % 2 == 0 else 0.5
        call_count["n"] += 1
        object.__setattr__(r.metrics, "sharpe_ratio", sharpe)
        return r

    with patch("backtest.sensitivity.run_backtest", side_effect=mock_run):
        rows = run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=cfg,
            output_dir=str(tmp_path / "sweep"),
        )

    sharpes = [r.sharpe_ratio for r in rows]
    assert sharpes == sorted(sharpes, reverse=True)


def test_run_sweep_writes_sensitivity_csv(tmp_path):
    cfg = SweepConfig(
        stop_loss_pcts=[-0.10],
        pir_thresholds=[0.30],
        lookback_days=[60],
        regime_bear_mults=[0.30],
    )
    with patch("backtest.sensitivity.run_backtest", return_value=_mock_backtest_result()):
        run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=cfg,
            output_dir=str(tmp_path / "sweep"),
        )
    csv_path = tmp_path / "sweep" / "sensitivity.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "stop_loss_pct" in content


def test_run_sweep_continues_on_failed_combo(tmp_path):
    cfg = SweepConfig(
        stop_loss_pcts=[-0.08, -0.10],
        pir_thresholds=[0.30],
        lookback_days=[60],
        regime_bear_mults=[0.30],
    )
    call_count = {"n": 0}
    def maybe_fail(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("simulated failure")
        return _mock_backtest_result()

    with patch("backtest.sensitivity.run_backtest", side_effect=maybe_fail):
        rows = run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=cfg,
            output_dir=str(tmp_path / "sweep"),
        )
    assert len(rows) == 1


def test_run_sweep_restores_config_after_each_combo(tmp_path):
    from src import config as cfg_module
    original_stop = cfg_module.STOP_LOSS_PCT

    sweep_cfg = SweepConfig(
        stop_loss_pcts=[-0.07],
        pir_thresholds=[0.30],
        lookback_days=[60],
        regime_bear_mults=[0.30],
    )
    with patch("backtest.sensitivity.run_backtest", return_value=_mock_backtest_result()):
        run_sweep(
            universe=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            sweep_config=sweep_cfg,
            output_dir=str(tmp_path / "sweep"),
        )

    assert cfg_module.STOP_LOSS_PCT == original_stop
