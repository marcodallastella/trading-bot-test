# tests/test_main.py
import sys
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.models import State, Position, RegimeInfo, TrailingStop
from src import config


# ── shared setup ──────────────────────────────────────────────────────────────

def _make_state(**overrides) -> State:
    base = dict(
        rolling_peak_equity=100.0,
        rolling_peak_timestamp="2026-04-01",
        last_run_timestamp="2026-04-17T15:00:00Z",
        ticker_last_buy_date={},
        wash_sale_blacklist={},
        trailing_stops_active={},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=False,
    )
    base.update(overrides)
    return State(**base)


def _make_regime(mult: float = config.REGIME_MULT_BULL) -> RegimeInfo:
    return RegimeInfo(
        spy_price=520.0, sma_200=495.0, slope_20d_pct=0.018,
        above_sma=True, slope_direction="rising",
        regime_mult=mult,
        description="SPY above 200SMA, slope rising → bull",
    )


# 220+ SPY bars for compute_regime
_SPY_BARS = pd.DataFrame({"close": [400.0 + i * 0.5 for i in range(225)]})

# 60 daily bars for evaluate_buy_signal — spike-then-settle (no buy signal at current=160)
_TICKER_BARS = pd.DataFrame({
    "close": [200.0] * 10 + [100.0] * 50,
    "high":  [201.0] * 10 + [101.0] * 50,
    "low":   [199.0] * 10 + [ 99.0] * 50,
})

# Quote with tight spread (2bps — well under 20bps threshold)
_TIGHT_QUOTE = {"ask": 160.10, "bid": 160.00, "last": 160.05}

# Quote with wide spread (30bps — exceeds 20bps buy threshold)
_WIDE_QUOTE  = {"ask": 162.40, "bid": 160.00, "last": 161.20}


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER",     "true")
    monkeypatch.setenv("DRY_RUN",          "true")
    monkeypatch.setenv("LIVE_READY",       "false")
    monkeypatch.setenv("KILL_SWITCH",      "0")
    monkeypatch.setenv("STATE_GIST_ID",    "gist_state")
    monkeypatch.setenv("DECISIONS_GIST_ID","gist_decisions")
    monkeypatch.setenv("GH_PAT",           "gh_token")
    monkeypatch.setenv("ALPACA_API_KEY",   "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY","test_secret")
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)


@pytest.fixture
def mock_client(mocker):
    client = Mock()
    client.get_clock.return_value     = {"is_open": True,  "next_open": "09:30", "next_close": "16:00"}
    client.get_account.return_value   = {"equity": 100.0,  "cash": 50.0, "buying_power": 50.0, "status": "ACTIVE"}
    client.get_positions.return_value = []
    client.get_spy_bars.return_value  = _SPY_BARS
    client.get_daily_bars.return_value = _TICKER_BARS
    client.get_latest_quote.return_value = _TIGHT_QUOTE
    mocker.patch("src.main.AlpacaClient", return_value=client)
    return client


@pytest.fixture
def mock_state(mocker):
    state = _make_state()
    mocker.patch("src.main.load_state", return_value=(state, False))
    mocker.patch("src.main.save_state",  return_value=True)
    return state


@pytest.fixture
def mock_logbook(mocker):
    return mocker.patch("src.main.append_decisions", return_value=True)


@pytest.fixture
def mock_notify(mocker):
    mocker.patch("src.main.send_daily_summary")
    return mocker.patch("src.main.format_summary", return_value="summary text")


# ── three-gate safety checks ──────────────────────────────────────────────────

def test_live_refused_without_live_ready(mock_env, mock_client, mock_state,
                                         mock_logbook, mock_notify, monkeypatch):
    """ALPACA_PAPER=false + DRY_RUN=false + LIVE_READY=false → sys.exit(1)."""
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("DRY_RUN",      "false")
    monkeypatch.setenv("LIVE_READY",   "false")
    from src.main import run
    with pytest.raises(SystemExit) as exc_info:
        run()
    assert exc_info.value.code == 1


def test_paper_dry_run_proceeds(mock_env, mock_client, mock_state,
                                mock_logbook, mock_notify):
    """Default paper + dry_run → no SystemExit."""
    from src.main import run
    run()   # must not raise


def test_live_all_gates_open_proceeds(mock_env, mock_client, mock_state,
                                      mock_logbook, mock_notify, monkeypatch):
    """All three gates open → proceeds (live mode allowed after burn-in)."""
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("DRY_RUN",      "false")
    monkeypatch.setenv("LIVE_READY",   "true")
    from src.main import run
    run()   # must not raise


# ── market-closed early exit ──────────────────────────────────────────────────

def test_market_closed_exits_without_trading(mock_env, mock_state,
                                              mock_logbook, mock_notify, mocker):
    """Market closed → run() returns early; no positions fetched."""
    client = Mock()
    client.get_clock.return_value = {"is_open": False, "next_open": "09:30", "next_close": "16:00"}
    mocker.patch("src.main.AlpacaClient", return_value=client)
    from src.main import run
    run()
    client.get_account.assert_not_called()
    client.get_positions.assert_not_called()


def test_market_closed_does_not_save_state(mock_env, mock_state,
                                            mock_logbook, mock_notify, mocker):
    client = Mock()
    client.get_clock.return_value = {"is_open": False, "next_open": "09:30", "next_close": "16:00"}
    mocker.patch("src.main.AlpacaClient", return_value=client)
    mock_save = mocker.patch("src.main.save_state")
    from src.main import run
    run()
    mock_save.assert_not_called()


# ── sell evaluation loop ──────────────────────────────────────────────────────

def test_stop_loss_triggers_sell(mock_env, mock_client, mock_state,
                                  mock_logbook, mock_notify, mocker):
    """Position at -10% → close_position called, wash-sale added."""
    pos = Position(ticker="AAPL", qty=0.1, avg_entry_price=100.0, current_price=89.0)
    mock_client.get_positions.return_value = [pos]
    mock_client.get_latest_quote.return_value = {"ask": 89.05, "bid": 89.00, "last": 89.0}
    monkeypatch_dry_run = mocker.patch.dict("os.environ", {"DRY_RUN": "false"})
    from src.main import run
    run()
    mock_client.close_position.assert_called_once_with("AAPL")


def test_stop_loss_dry_run_does_not_call_close(mock_env, mock_client, mock_state,
                                                mock_logbook, mock_notify):
    """DRY_RUN=true: stop loss logged but close_position not called."""
    pos = Position(ticker="AAPL", qty=0.1, avg_entry_price=100.0, current_price=89.0)
    mock_client.get_positions.return_value = [pos]
    mock_client.get_latest_quote.return_value = {"ask": 89.05, "bid": 89.00, "last": 89.0}
    from src.main import run
    run()
    mock_client.close_position.assert_not_called()


def test_wide_spread_blocks_sell(mock_env, mock_client, mock_state,
                                  mock_logbook, mock_notify):
    """Spread > 30bps on a stop-loss position → sell skipped."""
    pos = Position(ticker="AAPL", qty=0.1, avg_entry_price=100.0, current_price=89.0)
    mock_client.get_positions.return_value = [pos]
    # 40bps spread: (91-89)/90 * 10000 ≈ 222bps — well above 30bps
    mock_client.get_latest_quote.return_value = {"ask": 91.0, "bid": 89.0, "last": 90.0}
    from src.main import run
    run()
    mock_client.close_position.assert_not_called()


def test_trailing_stop_activated_on_20pct_gain(mock_env, mock_client, mock_state,
                                                mock_logbook, mock_notify):
    """Position at +20% with no trailing stop → state gets trailing stop added."""
    pos = Position(ticker="PLTR", qty=1.0, avg_entry_price=100.0, current_price=120.0)
    mock_client.get_positions.return_value = [pos]
    mock_client.get_latest_quote.return_value = {"ask": 120.05, "bid": 120.00, "last": 120.0}

    import src.main
    saved_states: list[State] = []

    original_save = src.main.save_state
    def capture_save(state, gist_id, token):
        saved_states.append(state)
        return True

    src.main.save_state = capture_save
    try:
        from src.main import run
        run()
    finally:
        src.main.save_state = original_save

    assert len(saved_states) == 1
    assert "PLTR" in saved_states[0].trailing_stops_active


def test_position_hold_does_not_call_close(mock_env, mock_client, mock_state,
                                            mock_logbook, mock_notify):
    """Position at +5% (no stop, no trailing) → hold, close_position not called."""
    pos = Position(ticker="AAPL", qty=0.1, avg_entry_price=100.0, current_price=105.0)
    mock_client.get_positions.return_value = [pos]
    mock_client.get_latest_quote.return_value = {"ask": 105.05, "bid": 105.00, "last": 105.0}
    from src.main import run
    run()
    mock_client.close_position.assert_not_called()


# ── buy evaluation loop ───────────────────────────────────────────────────────

def test_kill_switch_blocks_all_buys(mock_env, mock_client, mock_state,
                                      mock_logbook, mock_notify, monkeypatch):
    """KILL_SWITCH=1 → no buy orders placed for any ticker."""
    monkeypatch.setenv("KILL_SWITCH", "1")
    from src.main import run
    run()
    mock_client.submit_market_order.assert_not_called()


def test_drawdown_halt_blocks_buys(mock_env, mock_client, mock_logbook,
                                    mock_notify, mocker):
    """drawdown_halt_active=True in state → no buy orders placed."""
    halted_state = _make_state(drawdown_halt_active=True)
    mocker.patch("src.main.load_state", return_value=(halted_state, False))
    mocker.patch("src.main.save_state", return_value=True)
    from src.main import run
    run()
    mock_client.submit_market_order.assert_not_called()


def test_sell_only_mode_blocks_buys(mock_env, mock_client, mock_logbook,
                                     mock_notify, mocker):
    """sell_only=True → no buy orders placed."""
    state = _make_state()
    mocker.patch("src.main.load_state", return_value=(state, True))   # sell_only=True
    mocker.patch("src.main.save_state", return_value=True)
    from src.main import run
    run()
    mock_client.submit_market_order.assert_not_called()


def test_wide_buy_spread_skips_ticker(mock_env, mock_client, mock_state,
                                       mock_logbook, mock_notify):
    """Spread > 20bps on all quotes → no buy submitted."""
    mock_client.get_latest_quote.return_value = _WIDE_QUOTE   # ~30bps
    from src.main import run
    run()
    mock_client.submit_market_order.assert_not_called()


def test_cooldown_blocks_rebuy(mock_env, mock_client, mock_logbook,
                                mock_notify, mocker):
    """Ticker bought 1 day ago (< 3-day cooldown) → skipped."""
    state = _make_state(ticker_last_buy_date={"AAPL": "2026-04-19"})
    mocker.patch("src.main.load_state", return_value=(state, False))
    mocker.patch("src.main.save_state",  return_value=True)
    # Patch datetime so today is deterministic
    import src.main as m
    m_dt = Mock()
    m_dt.utcnow.return_value.strftime.return_value = "2026-04-20T15:00:00Z"
    mocker.patch.object(m, "datetime", m_dt)
    from src.main import run
    run()
    # AAPL bought yesterday — should be skipped (1 < 3 days)
    for call in mock_client.submit_market_order.call_args_list:
        assert call.args[0] != "AAPL", "AAPL should be on cooldown"


def test_buy_placed_in_live_mode(mock_env, mock_client, mock_logbook,
                                  mock_notify, mocker, monkeypatch):
    """DRY_RUN=false → submit_market_order called when signal fires."""
    monkeypatch.setenv("DRY_RUN", "false")
    mocker.patch("src.main.load_state", return_value=(_make_state(), False))
    mocker.patch("src.main.save_state", return_value=True)
    # Override bars with a clear buy signal: spike-then-settle at current=115
    bars = pd.DataFrame({
        "close": [200.0] * 10 + [100.0] * 50,
        "high":  [201.0] * 10 + [101.0] * 50,
        "low":   [199.0] * 10 + [ 99.0] * 50,
    })
    mock_client.get_daily_bars.return_value = bars
    mock_client.get_latest_quote.return_value = {"ask": 115.05, "bid": 115.00, "last": 115.0}
    mock_client.submit_market_order.return_value = {"fill_price": 115.05}
    from src.main import run
    run()
    assert mock_client.submit_market_order.called


def test_buy_not_placed_in_dry_run(mock_env, mock_client, mock_state,
                                    mock_logbook, mock_notify):
    """DRY_RUN=true (default in mock_env) → submit_market_order never called."""
    bars = pd.DataFrame({
        "close": [200.0] * 10 + [100.0] * 50,
        "high":  [201.0] * 10 + [101.0] * 50,
        "low":   [199.0] * 10 + [ 99.0] * 50,
    })
    mock_client.get_daily_bars.return_value = bars
    mock_client.get_latest_quote.return_value = {"ask": 115.05, "bid": 115.00, "last": 115.0}
    from src.main import run
    run()
    mock_client.submit_market_order.assert_not_called()


# ── end-of-run: state save, logbook, notify, error handler ───────────────────

def test_state_saved_at_end_of_run(mock_env, mock_client, mock_state,
                                    mock_logbook, mock_notify, mocker):
    mock_save = mocker.patch("src.main.save_state", return_value=True)
    from src.main import run
    run()
    mock_save.assert_called_once()


def test_decisions_appended_at_end_of_run(mock_env, mock_client, mock_state,
                                           mock_logbook, mock_notify):
    from src.main import run
    run()
    mock_logbook.assert_called_once()
    rows = mock_logbook.call_args.args[0]
    # At minimum one row per UNIVERSE ticker (all skipped — no signals on default bars)
    assert len(rows) >= len(config.UNIVERSE)


def test_notify_called_at_end_of_run(mock_env, mock_client, mock_state,
                                      mock_logbook, mock_notify):
    from src.main import run
    run()
    mock_notify.assert_called_once()  # format_summary


def test_no_decisions_gist_id_skips_logbook(mock_env, mock_client, mock_state,
                                              mock_logbook, mock_notify, monkeypatch):
    """If DECISIONS_GIST_ID is unset, logbook append is skipped gracefully."""
    monkeypatch.delenv("DECISIONS_GIST_ID", raising=False)
    from src.main import run
    run()
    mock_logbook.assert_not_called()


def test_main_catches_unhandled_exception_and_exits_1(mock_env, mocker):
    """Top-level exception in run() → main() catches it and sys.exit(1)."""
    mocker.patch("src.main.run", side_effect=RuntimeError("catastrophic failure"))
    from src.main import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
