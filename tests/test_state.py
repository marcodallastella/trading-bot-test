# tests/test_state.py
import json
import copy
import pytest
import requests as requests_lib
from unittest.mock import Mock, patch
from src.state import load_state, save_state
from src.models import State, TrailingStop, WashSaleEntry
from src import config
from tests.conftest import make_gist_response, VALID_STATE_DICT


# ── Happy-path load ────────────────────────────────────────────────────────────

def test_load_state_returns_state_object(gist_ok, mocker):
    mocker.patch("src.state.requests.get", return_value=gist_ok)
    state, sell_only = load_state("gist123", "token_abc")
    assert isinstance(state, State)
    assert sell_only is False


def test_load_state_fields_match_json(gist_ok, mocker):
    mocker.patch("src.state.requests.get", return_value=gist_ok)
    state, _ = load_state("gist123", "token_abc")
    assert state.rolling_peak_equity == 104.30
    assert state.rolling_peak_timestamp == "2026-03-15"
    assert state.ticker_last_buy_date == {"AAPL": "2026-04-15"}
    assert state.cumulative_realized_pnl == 8.42
    assert state.drawdown_halt_active is False


def test_load_state_deserialises_trailing_stop(gist_ok, mocker):
    mocker.patch("src.state.requests.get", return_value=gist_ok)
    state, _ = load_state("gist123", "token_abc")
    assert "PLTR" in state.trailing_stops_active
    ts = state.trailing_stops_active["PLTR"]
    assert isinstance(ts, TrailingStop)
    assert ts.entry_price == 22.40
    assert ts.peak_price == 28.15
    assert ts.activated == "2026-04-12"


def test_load_state_deserialises_wash_sale(gist_ok, mocker):
    mocker.patch("src.state.requests.get", return_value=gist_ok)
    state, _ = load_state("gist123", "token_abc")
    assert "MSFT" in state.wash_sale_blacklist
    ws = state.wash_sale_blacklist["MSFT"]
    assert isinstance(ws, WashSaleEntry)
    assert ws.sold_date == "2026-03-20"
    assert ws.expires == "2026-04-20"


def test_load_state_empty_dicts_are_valid(mocker):
    minimal = {
        "rolling_peak_equity": 100.0,
        "rolling_peak_timestamp": "2026-01-01",
        "last_run_timestamp": "2026-01-01T15:00:00Z",
        "ticker_last_buy_date": {},
        "wash_sale_blacklist": {},
        "trailing_stops_active": {},
        "cumulative_realized_pnl": 0.0,
        "drawdown_halt_active": False,
    }
    mocker.patch("src.state.requests.get", return_value=make_gist_response(minimal))
    state, sell_only = load_state("gist123", "token_abc")
    assert sell_only is False
    assert state.trailing_stops_active == {}
    assert state.wash_sale_blacklist == {}


# ── Retry logic ────────────────────────────────────────────────────────────────

def test_load_state_retries_once_on_network_error(valid_state_dict, mocker):
    ok = make_gist_response(valid_state_dict)
    mocker.patch(
        "src.state.requests.get",
        side_effect=[requests_lib.exceptions.ConnectionError("timeout"), ok],
    )
    mocker.patch("src.state.time.sleep")
    state, sell_only = load_state("gist123", "token_abc")
    assert sell_only is False
    assert state.rolling_peak_equity == 104.30


def test_load_state_sell_only_after_two_network_failures(mocker):
    mocker.patch(
        "src.state.requests.get",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    mocker.patch("src.state.time.sleep")
    state, sell_only = load_state("gist123", "token_abc")
    assert sell_only is True


def test_load_state_sell_only_state_has_halt_active(mocker):
    mocker.patch(
        "src.state.requests.get",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    mocker.patch("src.state.time.sleep")
    state, sell_only = load_state("gist123", "token_abc")
    assert sell_only is True
    assert state.drawdown_halt_active is True


def test_load_state_retries_exactly_once(valid_state_dict, mocker):
    ok = make_gist_response(valid_state_dict)
    mocker.patch(
        "src.state.requests.get",
        side_effect=[requests_lib.exceptions.ConnectionError("timeout"), ok],
    )
    mock_sleep = mocker.patch("src.state.time.sleep")
    load_state("gist123", "token_abc")
    mock_sleep.assert_called_once_with(config.GIST_RETRY_DELAY_SECONDS)


def test_load_state_sell_only_on_http_error(mocker):
    bad = Mock()
    bad.raise_for_status.side_effect = requests_lib.exceptions.HTTPError("403")
    mocker.patch("src.state.requests.get", return_value=bad)
    mocker.patch("src.state.time.sleep")
    _, sell_only = load_state("gist123", "token_abc")
    assert sell_only is True


# ── Parse failures ─────────────────────────────────────────────────────────────

def test_load_state_sell_only_on_malformed_json(mocker):
    bad = Mock()
    bad.raise_for_status.return_value = None
    bad.json.return_value = {
        "files": {"trading_bot_state.json": {"content": "this is not json {{{"}}
    }
    mocker.patch("src.state.requests.get", return_value=bad)
    _, sell_only = load_state("gist123", "token_abc")
    assert sell_only is True


def test_load_state_sell_only_on_missing_field(valid_state_dict, mocker):
    required_fields = [
        "rolling_peak_equity", "rolling_peak_timestamp", "last_run_timestamp",
        "ticker_last_buy_date", "wash_sale_blacklist", "trailing_stops_active",
        "cumulative_realized_pnl", "drawdown_halt_active",
    ]
    for field in required_fields:
        truncated = {k: v for k, v in valid_state_dict.items() if k != field}
        mocker.patch("src.state.requests.get", return_value=make_gist_response(truncated))
        _, sell_only = load_state("gist123", "token_abc")
        assert sell_only is True, f"Expected sell_only=True when '{field}' is missing"


def test_load_state_sell_only_on_malformed_trailing_stop(valid_state_dict, mocker):
    valid_state_dict["trailing_stops_active"] = {"PLTR": {"wrong_key": 99.9}}
    mocker.patch("src.state.requests.get", return_value=make_gist_response(valid_state_dict))
    _, sell_only = load_state("gist123", "token_abc")
    assert sell_only is True


def test_load_state_sell_only_does_not_raise(mocker):
    mocker.patch("src.state.requests.get", side_effect=RuntimeError("unexpected!"))
    mocker.patch("src.state.time.sleep")
    try:
        state, sell_only = load_state("gist123", "token_abc")
    except Exception as exc:
        pytest.fail(f"load_state raised unexpectedly: {exc}")
    assert sell_only is True


# ── save_state ─────────────────────────────────────────────────────────────────

def test_save_state_returns_true_on_success(valid_state, mocker):
    ok = Mock()
    ok.raise_for_status.return_value = None
    mock_patch = mocker.patch("src.state.requests.patch", return_value=ok)
    result = save_state(valid_state, "gist123", "token_abc")
    assert result is True
    mock_patch.assert_called_once()


def test_save_state_sends_correct_gist_filename(valid_state, mocker):
    ok = Mock()
    ok.raise_for_status.return_value = None
    mock_patch = mocker.patch("src.state.requests.patch", return_value=ok)
    save_state(valid_state, "gist123", "token_abc")
    call_kwargs = mock_patch.call_args
    payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs.kwargs["json"]
    assert config.STATE_GIST_FILENAME in payload["files"]


def test_save_state_returns_false_on_http_error(valid_state, mocker):
    bad = Mock()
    bad.raise_for_status.side_effect = requests_lib.exceptions.HTTPError("500")
    mocker.patch("src.state.requests.patch", return_value=bad)
    result = save_state(valid_state, "gist123", "token_abc")
    assert result is False


def test_save_state_returns_false_on_connection_error(valid_state, mocker):
    mocker.patch(
        "src.state.requests.patch",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    result = save_state(valid_state, "gist123", "token_abc")
    assert result is False


def test_save_state_does_not_raise(valid_state, mocker):
    mocker.patch("src.state.requests.patch", side_effect=RuntimeError("kaboom"))
    try:
        result = save_state(valid_state, "gist123", "token_abc")
    except Exception as exc:
        pytest.fail(f"save_state raised unexpectedly: {exc}")
    assert result is False


# ── Round-trip ─────────────────────────────────────────────────────────────────

def test_state_round_trip(valid_state, mocker):
    captured_content: list[str] = []

    def fake_patch(url, json, headers, timeout):
        content = json["files"][config.STATE_GIST_FILENAME]["content"]
        captured_content.append(content)
        ok = Mock()
        ok.raise_for_status.return_value = None
        return ok

    mocker.patch("src.state.requests.patch", side_effect=fake_patch)
    save_state(valid_state, "gist123", "token_abc")
    assert len(captured_content) == 1

    get_mock = Mock()
    get_mock.raise_for_status.return_value = None
    get_mock.json.return_value = {
        "files": {config.STATE_GIST_FILENAME: {"content": captured_content[0]}}
    }
    mocker.patch("src.state.requests.get", return_value=get_mock)
    restored, sell_only = load_state("gist123", "token_abc")

    assert sell_only is False
    assert restored.rolling_peak_equity == valid_state.rolling_peak_equity
    assert restored.rolling_peak_timestamp == valid_state.rolling_peak_timestamp
    assert restored.ticker_last_buy_date == valid_state.ticker_last_buy_date
    assert restored.cumulative_realized_pnl == valid_state.cumulative_realized_pnl
    assert restored.drawdown_halt_active == valid_state.drawdown_halt_active
    assert "PLTR" in restored.trailing_stops_active
    assert restored.trailing_stops_active["PLTR"].peak_price == 28.15
    assert "MSFT" in restored.wash_sale_blacklist
    assert restored.wash_sale_blacklist["MSFT"].expires == "2026-04-20"


def test_save_state_rounds_floats_to_4dp(valid_state, mocker):
    valid_state.rolling_peak_equity = 104.123456789
    captured: list[str] = []

    def fake_patch(url, json, headers, timeout):
        captured.append(json["files"][config.STATE_GIST_FILENAME]["content"])
        m = Mock()
        m.raise_for_status.return_value = None
        return m

    mocker.patch("src.state.requests.patch", side_effect=fake_patch)
    save_state(valid_state, "gist123", "token_abc")
    saved = json.loads(captured[0])
    assert saved["rolling_peak_equity"] == round(104.123456789, 4)
