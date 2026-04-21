import math
import pytest
import pandas as pd
import numpy as np


def _equity(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_total_return_positive():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0, 120.0])
    m = compute_metrics(eq, [10.0, 10.0])
    assert m.total_return_pct == pytest.approx(20.0, rel=0.01)


def test_total_return_negative():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 90.0, 80.0])
    m = compute_metrics(eq, [-10.0, -10.0])
    assert m.total_return_pct == pytest.approx(-20.0, rel=0.01)


def test_annualized_return_one_year():
    from backtest.metrics import compute_metrics
    # 252 days, 10% total return → annualized ≈ 10%
    n = 252
    start = 100.0
    end = 110.0
    values = [start + (end - start) * i / (n - 1) for i in range(n)]
    eq = _equity(values)
    m = compute_metrics(eq, [10.0])
    assert m.annualized_return_pct == pytest.approx(10.0, rel=0.05)


def test_sharpe_positive_for_uptrend():
    from backtest.metrics import compute_metrics
    # Monotonically increasing → positive Sharpe
    eq = _equity([100.0 + i * 0.5 for i in range(100)])
    m = compute_metrics(eq, [5.0])
    assert m.sharpe_ratio > 0


def test_sharpe_zero_for_flat():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0] * 10)
    m = compute_metrics(eq, [])
    assert m.sharpe_ratio == 0.0


def test_sortino_infinite_when_no_downside():
    from backtest.metrics import compute_metrics
    # All returns positive → no downside → sortino = inf
    eq = _equity([100.0, 101.0, 102.0, 103.0])
    m = compute_metrics(eq, [3.0])
    assert math.isinf(m.sortino_ratio)


def test_max_drawdown_50_pct():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 150.0, 75.0, 100.0])
    m = compute_metrics(eq, [])
    # Peak=150, trough=75 → 50% drawdown
    assert m.max_drawdown_pct == pytest.approx(50.0, rel=0.01)


def test_max_drawdown_zero_for_monotone():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0, 120.0, 130.0])
    m = compute_metrics(eq, [])
    assert m.max_drawdown_pct == 0.0


def test_win_rate_all_winners():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0])
    m = compute_metrics(eq, [5.0, 3.0, 2.0])
    assert m.win_rate_pct == pytest.approx(100.0)


def test_win_rate_all_losers():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 90.0])
    m = compute_metrics(eq, [-5.0, -3.0])
    assert m.win_rate_pct == pytest.approx(0.0)


def test_win_rate_mixed():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 105.0])
    m = compute_metrics(eq, [10.0, -5.0])
    assert m.win_rate_pct == pytest.approx(50.0)


def test_profit_factor_no_losers():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0])
    m = compute_metrics(eq, [5.0, 3.0])
    assert math.isinf(m.profit_factor)


def test_kelly_basic():
    from backtest.metrics import compute_metrics
    # 60% win rate, avg win 10, avg loss 5 → r=2, kelly = 0.6 - 0.4/2 = 0.4
    eq = _equity([100.0, 110.0])
    pnls = [10.0, 10.0, 10.0, -5.0, -5.0]
    m = compute_metrics(eq, pnls)
    assert m.kelly_fraction == pytest.approx(0.4, rel=0.01)


def test_kelly_no_trades():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0])
    m = compute_metrics(eq, [])
    assert m.kelly_fraction == 0.0


def test_trade_counts():
    from backtest.metrics import compute_metrics
    eq = _equity([100.0, 110.0])
    m = compute_metrics(eq, [5.0, -2.0, 3.0, -1.0])
    assert m.total_trades == 4
    assert m.winning_trades == 2
    assert m.losing_trades == 2
