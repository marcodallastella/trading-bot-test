# tests/test_alpaca_client.py
import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.alpaca_client import AlpacaClient
from src.models import Position


def _make_client(mocker) -> AlpacaClient:
    """Create an AlpacaClient with SDK constructors patched out."""
    mocker.patch("src.alpaca_client.TradingClient.__init__", return_value=None)
    mocker.patch("src.alpaca_client.StockHistoricalDataClient.__init__", return_value=None)
    return AlpacaClient("key", "secret", paper=True)


# ── Clock ──────────────────────────────────────────────────────────────────────

def test_get_clock_returns_is_open_true(mocker):
    mock_clock = mocker.MagicMock()
    mock_clock.is_open = True
    mock_clock.next_open = "2026-04-21T09:30:00Z"
    mock_clock.next_close = "2026-04-21T16:00:00Z"
    mocker.patch("src.alpaca_client.TradingClient.get_clock", return_value=mock_clock)
    client = _make_client(mocker)
    result = client.get_clock()
    assert result["is_open"] is True
    assert result["next_open"] == "2026-04-21T09:30:00Z"
    assert result["next_close"] == "2026-04-21T16:00:00Z"


def test_get_clock_returns_is_open_false(mocker):
    mock_clock = mocker.MagicMock()
    mock_clock.is_open = False
    mock_clock.next_open = "2026-04-22T09:30:00Z"
    mock_clock.next_close = "2026-04-22T16:00:00Z"
    mocker.patch("src.alpaca_client.TradingClient.get_clock", return_value=mock_clock)
    client = _make_client(mocker)
    result = client.get_clock()
    assert result["is_open"] is False


# ── Account ────────────────────────────────────────────────────────────────────

def test_get_account_returns_equity_float(mocker):
    mock_account = mocker.MagicMock()
    mock_account.equity = "1234.56"
    mock_account.buying_power = "500.00"
    mock_account.cash = "300.00"
    mocker.patch("src.alpaca_client.TradingClient.get_account", return_value=mock_account)
    client = _make_client(mocker)
    result = client.get_account()
    assert isinstance(result["equity"], float)
    assert result["equity"] == 1234.56
    assert isinstance(result["buying_power"], float)
    assert isinstance(result["cash"], float)


# ── Positions ──────────────────────────────────────────────────────────────────

def test_get_positions_returns_position_objects(mocker):
    pos1 = mocker.MagicMock()
    pos1.symbol = "AAPL"
    pos1.qty = "5.0"
    pos1.avg_entry_price = "150.00"

    pos2 = mocker.MagicMock()
    pos2.symbol = "NVDA"
    pos2.qty = "2.5"
    pos2.avg_entry_price = "800.00"

    mocker.patch("src.alpaca_client.TradingClient.get_all_positions", return_value=[pos1, pos2])
    client = _make_client(mocker)
    positions = client.get_positions()
    assert len(positions) == 2
    for p in positions:
        assert isinstance(p, Position)
    assert positions[0].ticker == "AAPL"
    assert positions[1].ticker == "NVDA"


def test_get_positions_empty_returns_empty_list(mocker):
    mocker.patch("src.alpaca_client.TradingClient.get_all_positions", return_value=[])
    client = _make_client(mocker)
    positions = client.get_positions()
    assert positions == []


# ── Quote ──────────────────────────────────────────────────────────────────────

def test_get_quote_returns_ask_bid_last(mocker):
    mock_quote = mocker.MagicMock()
    mock_quote.ask_price = "151.50"
    mock_quote.bid_price = "151.40"

    mocker.patch(
        "src.alpaca_client.StockHistoricalDataClient.get_stock_latest_quote",
        return_value={"AAPL": mock_quote},
    )
    client = _make_client(mocker)
    result = client.get_quote("AAPL")
    assert "ask" in result
    assert "bid" in result
    assert "last" in result
    assert isinstance(result["ask"], float)
    assert result["ask"] == 151.50
    assert result["bid"] == 151.40


# ── Bars ───────────────────────────────────────────────────────────────────────

def test_get_bars_returns_dataframe_with_ohlcv_columns(mocker):
    bar = mocker.MagicMock()
    bar.open = 150.0
    bar.high = 155.0
    bar.low = 149.0
    bar.close = 153.0
    bar.volume = 1_000_000.0
    bar.timestamp = pd.Timestamp("2026-04-18", tz="UTC")

    mocker.patch(
        "src.alpaca_client.StockHistoricalDataClient.get_stock_bars",
        return_value={"AAPL": [bar]},
    )
    client = _make_client(mocker)
    df = client.get_bars("AAPL", lookback_days=60)
    assert isinstance(df, pd.DataFrame)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"


# ── Submit buy order ───────────────────────────────────────────────────────────

def test_submit_buy_order_calls_with_notional(mocker):
    mock_order = mocker.MagicMock()
    mock_order.id = "order-123"
    mock_order.symbol = "TSLA"
    mock_order.status = "accepted"

    mock_submit = mocker.patch(
        "src.alpaca_client.TradingClient.submit_order",
        return_value=mock_order,
    )
    client = _make_client(mocker)
    result = client.submit_buy_order("TSLA", notional=100.0)
    assert result is not None
    assert result["notional"] == 100.0
    mock_submit.assert_called_once()
    call_args = mock_submit.call_args[0][0]
    assert call_args.notional == 100.0
    assert call_args.symbol == "TSLA"


def test_submit_buy_order_returns_none_on_error(mocker):
    mocker.patch(
        "src.alpaca_client.TradingClient.submit_order",
        side_effect=RuntimeError("API error"),
    )
    client = _make_client(mocker)
    result = client.submit_buy_order("TSLA", notional=100.0)
    assert result is None


# ── Close position ─────────────────────────────────────────────────────────────

def test_close_position_calls_close_position(mocker):
    mock_order = mocker.MagicMock()
    mock_order.id = "order-456"
    mock_order.symbol = "AAPL"
    mock_order.status = "accepted"

    mock_close = mocker.patch(
        "src.alpaca_client.TradingClient.close_position",
        return_value=mock_order,
    )
    client = _make_client(mocker)
    result = client.close_position("AAPL")
    assert result is not None
    mock_close.assert_called_once_with("AAPL")


def test_close_position_returns_none_on_error(mocker):
    mocker.patch(
        "src.alpaca_client.TradingClient.close_position",
        side_effect=RuntimeError("API error"),
    )
    client = _make_client(mocker)
    result = client.close_position("AAPL")
    assert result is None
