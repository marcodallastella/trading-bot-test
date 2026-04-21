# tests/test_notify.py
import pytest
from unittest.mock import Mock
from src.notify import format_summary, send_daily_summary
from src.models import Position, RegimeInfo, TrailingStop
from src import config


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def regime_bull():
    return RegimeInfo(
        spy_price=520.0,
        sma_200=495.0,
        slope_20d_pct=0.018,
        above_sma=True,
        slope_direction="rising",
        regime_mult=config.REGIME_MULT_BULL,
        description="SPY above 200SMA, slope rising → bull",
    )


@pytest.fixture
def sample_position():
    return Position(
        ticker="NVDA",
        qty=0.128,
        avg_entry_price=98.30,
        current_price=102.10,
    )


@pytest.fixture
def base_summary_kwargs(regime_bull, sample_position):
    return dict(
        run_timestamp="2026-04-18T15:00:00Z",
        is_paper=True,
        is_dry_run=False,
        equity=108.40,
        starting_equity=100.0,
        rolling_peak_equity=112.60,
        rolling_peak_timestamp="2026-04-15",
        drawdown_pct=0.037,
        halt_active=False,
        regime=regime_bull,
        kelly_base=0.097,
        positions=[sample_position],
        trailing_stops={},
        actions_log=["BUY AMD $11.64 @ $142.80 (PIR=0.08, deep dip)"],
        cumulative_realized_pnl=8.42,
        wash_sale_count=0,
    )


# ── format_summary tests ──────────────────────────────────────────────────────

def test_format_summary_returns_string(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_summary_contains_date(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "2026-04-18" in result


def test_format_summary_paper_mode(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "PAPER" in result


def test_format_summary_live_mode(base_summary_kwargs):
    base_summary_kwargs["is_paper"] = False
    result = format_summary(**base_summary_kwargs)
    assert "LIVE" in result


def test_format_summary_dry_run_flag(base_summary_kwargs):
    base_summary_kwargs["is_dry_run"] = True
    result = format_summary(**base_summary_kwargs)
    assert "true" in result.lower() or "DRY_RUN=true" in result


def test_format_summary_contains_equity(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "108.40" in result


def test_format_summary_contains_all_time_pnl(base_summary_kwargs):
    # equity=108.40, starting=100 → +8.40, +8.4%
    result = format_summary(**base_summary_kwargs)
    assert "+8.40" in result or "+8.4" in result


def test_format_summary_contains_rolling_peak(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "112.60" in result
    assert "2026-04-15" in result


def test_format_summary_contains_drawdown(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "3.7%" in result


def test_format_summary_halt_active_shows_warning(base_summary_kwargs):
    base_summary_kwargs["halt_active"] = True
    result = format_summary(**base_summary_kwargs)
    assert "HALT" in result.upper()


def test_format_summary_halt_inactive_shows_ok(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "OK" in result or "ok" in result.lower()


def test_format_summary_contains_regime_description(base_summary_kwargs, regime_bull):
    result = format_summary(**base_summary_kwargs)
    assert "bull" in result.lower() or str(regime_bull.regime_mult) in result


def test_format_summary_contains_kelly_sizing(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "9.7%" in result or "0.097" in result


def test_format_summary_contains_position(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "NVDA" in result
    assert "102.10" in result or "102.1" in result


def test_format_summary_marks_trailing_stop(base_summary_kwargs, sample_position):
    base_summary_kwargs["trailing_stops"] = {
        "NVDA": TrailingStop(entry_price=98.30, peak_price=108.0, activated="2026-04-10")
    }
    result = format_summary(**base_summary_kwargs)
    assert "TRAILING" in result.upper() or "108.0" in result


def test_format_summary_contains_action(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "AMD" in result
    assert "11.64" in result


def test_format_summary_no_positions_shows_placeholder(base_summary_kwargs):
    base_summary_kwargs["positions"] = []
    result = format_summary(**base_summary_kwargs)
    assert "no open positions" in result.lower() or "0)" in result


def test_format_summary_wash_sale_count(base_summary_kwargs):
    base_summary_kwargs["wash_sale_count"] = 2
    result = format_summary(**base_summary_kwargs)
    assert "2" in result and ("wash" in result.lower() or "blacklist" in result.lower())


def test_format_summary_cumulative_pnl(base_summary_kwargs):
    result = format_summary(**base_summary_kwargs)
    assert "8.42" in result


import requests as requests_lib

# ── send_daily_summary tests ──────────────────────────────────────────────────

def test_send_posts_to_webhook(mocker):
    ok = Mock()
    ok.raise_for_status.return_value = None
    mock_post = mocker.patch("src.notify.requests.post", return_value=ok)
    send_daily_summary("hello summary", webhook_url="https://hooks.example.com/abc")
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["json"]["content"] == "hello summary"


def test_send_posts_to_correct_url(mocker):
    ok = Mock()
    ok.raise_for_status.return_value = None
    mock_post = mocker.patch("src.notify.requests.post", return_value=ok)
    send_daily_summary("msg", webhook_url="https://hooks.example.com/xyz")
    assert mock_post.call_args.args[0] == "https://hooks.example.com/xyz"


def test_send_no_webhook_does_not_call_post(mocker):
    mock_post = mocker.patch("src.notify.requests.post")
    send_daily_summary("msg", webhook_url=None)
    mock_post.assert_not_called()


def test_send_empty_webhook_does_not_call_post(mocker):
    mock_post = mocker.patch("src.notify.requests.post")
    send_daily_summary("msg", webhook_url="")
    mock_post.assert_not_called()


def test_send_does_not_raise_on_http_error(mocker):
    bad = Mock()
    bad.raise_for_status.side_effect = requests_lib.exceptions.HTTPError("500")
    mocker.patch("src.notify.requests.post", return_value=bad)
    try:
        send_daily_summary("msg", webhook_url="https://hooks.example.com/abc")
    except Exception as exc:
        pytest.fail(f"send_daily_summary raised unexpectedly: {exc}")


def test_send_does_not_raise_on_connection_error(mocker):
    mocker.patch(
        "src.notify.requests.post",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    try:
        send_daily_summary("msg", webhook_url="https://hooks.example.com/abc")
    except Exception as exc:
        pytest.fail(f"send_daily_summary raised unexpectedly: {exc}")
