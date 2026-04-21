import pandas as pd
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=n, freq="B").date
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n)],
        "high":   [102.0 + i for i in range(n)],
        "low":    [ 99.0 + i for i in range(n)],
        "close":  [101.0 + i for i in range(n)],
        "volume": [1_000_000] * n,
    }, index=pd.Index(dates, name="date"))


def test_fetch_bars_returns_dataframe_with_correct_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    with patch.object(bdata, "_fetch_yfinance", return_value=_make_ohlcv()):
        df = bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5


def test_fetch_bars_writes_parquet_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    with patch.object(bdata, "_fetch_yfinance", return_value=_make_ohlcv()):
        bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8))

    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1


def test_fetch_bars_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    mock_yf = MagicMock(return_value=_make_ohlcv())
    with patch.object(bdata, "_fetch_yfinance", mock_yf):
        bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8))
        bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8))

    assert mock_yf.call_count == 1


def test_fetch_bars_force_refresh_ignores_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    mock_yf = MagicMock(return_value=_make_ohlcv())
    with patch.object(bdata, "_fetch_yfinance", mock_yf):
        bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8))
        bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8), force_refresh=True)

    assert mock_yf.call_count == 2


def test_fetch_bars_alpaca_preferred_over_yfinance(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    alpaca_df = _make_ohlcv(10)
    mock_alpaca = MagicMock(return_value=alpaca_df)
    mock_yf = MagicMock()
    with patch.object(bdata, "_fetch_alpaca", mock_alpaca), \
         patch.object(bdata, "_fetch_yfinance", mock_yf):
        df = bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 15),
                               api_key="KEY", secret_key="SECRET")

    mock_alpaca.assert_called_once()
    mock_yf.assert_not_called()
    assert len(df) == 10


def test_fetch_bars_falls_back_to_yfinance_on_alpaca_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    with patch.object(bdata, "_fetch_alpaca", side_effect=RuntimeError("timeout")), \
         patch.object(bdata, "_fetch_yfinance", return_value=_make_ohlcv()):
        df = bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8),
                               api_key="KEY", secret_key="SECRET")
    assert len(df) == 5


def test_fetch_bars_raises_value_error_when_all_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import backtest.data as bdata
    reload(bdata)

    with patch.object(bdata, "_fetch_alpaca", side_effect=RuntimeError("err")), \
         patch.object(bdata, "_fetch_yfinance", side_effect=RuntimeError("err")):
        with pytest.raises(ValueError, match="All bar sources failed"):
            bdata.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8),
                              api_key="KEY", secret_key="SECRET")
