import math
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.run_backtest import run_backtest, _initial_state
from src.models import State


def _make_bars(n: int = 300, start_price: float = 100.0, trend: float = 0.2) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n).date
    closes = [start_price + i * trend for i in range(n)]
    highs  = [c * 1.005 for c in closes]
    lows   = [c * 0.995 for c in closes]
    opens  = [c * 1.001 for c in closes]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1_000_000] * n,
    }, index=dates)


def _make_spy_bars(n: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n).date
    closes = [400.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "open": closes, "high": [c * 1.002 for c in closes],
        "low":  [c * 0.998 for c in closes], "close": closes,
        "volume": [50_000_000] * n,
    }, index=dates)


def test_initial_state_has_correct_starting_equity():
    s = _initial_state(100.0)
    assert s.rolling_peak_equity == 100.0
    assert s.drawdown_halt_active is False
    assert s.ticker_last_buy_date == {}


def test_run_backtest_returns_backtest_result(tmp_path):
    ticker_bars = _make_bars(300)
    spy_bars    = _make_spy_bars(300)

    def mock_fetch_bars(ticker, start, end, **kwargs):
        if ticker == "SPY":
            return spy_bars
        return ticker_bars

    with patch("backtest.run_backtest.fetch_bars", side_effect=mock_fetch_bars):
        result = run_backtest(
            universe=["AAPL", "MSFT"],
            start=date(2024, 7, 1),
            end=date(2024, 12, 31),
            starting_equity=100.0,
            output_dir=str(tmp_path / "out"),
        )

    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert isinstance(result.metrics.total_return_pct, float)
    assert result.equity_curve_path is not None
    assert Path(result.equity_curve_path).exists()


def test_run_backtest_equity_curve_png_created(tmp_path):
    ticker_bars = _make_bars(300)
    spy_bars    = _make_spy_bars(300)

    with patch("backtest.run_backtest.fetch_bars",
               side_effect=lambda t, *a, **kw: spy_bars if t == "SPY" else ticker_bars):
        result = run_backtest(
            universe=["AAPL"],
            start=date(2024, 7, 1),
            end=date(2024, 12, 31),
            output_dir=str(tmp_path / "out"),
        )

    assert Path(result.equity_curve_path).suffix == ".png"
    assert Path(result.equity_curve_path).stat().st_size > 0


def test_run_backtest_equity_starts_at_starting_value(tmp_path):
    ticker_bars = _make_bars(300)
    spy_bars    = _make_spy_bars(300)

    with patch("backtest.run_backtest.fetch_bars",
               side_effect=lambda t, *a, **kw: spy_bars if t == "SPY" else ticker_bars):
        result = run_backtest(
            universe=["AAPL"],
            start=date(2024, 7, 1),
            end=date(2024, 12, 31),
            starting_equity=500.0,
            output_dir=str(tmp_path / "out"),
        )

    assert result.equity_curve.iloc[0] == pytest.approx(500.0, rel=0.01)


def test_run_backtest_kelly_recommendation_in_range(tmp_path):
    ticker_bars = _make_bars(300)
    spy_bars    = _make_spy_bars(300)

    with patch("backtest.run_backtest.fetch_bars",
               side_effect=lambda t, *a, **kw: spy_bars if t == "SPY" else ticker_bars):
        result = run_backtest(
            universe=["AAPL", "MSFT"],
            start=date(2024, 7, 1),
            end=date(2024, 12, 31),
            output_dir=str(tmp_path / "out"),
        )

    assert 0.04 <= result.kelly_recommendation <= 0.12


def test_run_backtest_missing_spy_raises_value_error(tmp_path):
    with patch("backtest.run_backtest.fetch_bars", side_effect=ValueError("no data")):
        with pytest.raises(ValueError, match="SPY"):
            run_backtest(
                universe=["AAPL"],
                start=date(2024, 7, 1),
                end=date(2024, 12, 31),
                output_dir=str(tmp_path / "out"),
            )


def test_run_backtest_decisions_log_non_empty_for_active_market(tmp_path):
    # 300 bars: first 150 bars rise from 100 to 249, then dip to 60 for the rest.
    # During simulation (2024-07-01 to 2024-12-31), bars_up_to will include the
    # rising period, creating a varied lookback window. The dip at the end
    # drives price below the PIR threshold to trigger buy signals.
    n = 300
    dates = pd.bdate_range("2024-01-02", periods=n).date
    closes = [100.0 + i for i in range(150)] + [60.0] * 150
    dip_bars = pd.DataFrame({
        "open": closes, "high": [c * 1.01 for c in closes],
        "low":  [c * 0.99 for c in closes], "close": closes,
        "volume": [1_000_000] * n,
    }, index=dates)
    spy_bars = _make_spy_bars(300)

    with patch("backtest.run_backtest.fetch_bars",
               side_effect=lambda t, *a, **kw: spy_bars if t == "SPY" else dip_bars):
        result = run_backtest(
            universe=["AAPL"],
            start=date(2024, 7, 1),
            end=date(2024, 12, 31),
            output_dir=str(tmp_path / "out"),
        )

    buy_decisions = [d for d in result.decisions_log if d["action"] == "buy"]
    assert len(buy_decisions) > 0
