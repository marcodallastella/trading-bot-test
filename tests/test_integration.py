"""
Manual-only integration tests.  Skipped automatically unless ALPACA_API_KEY is set.

Run with:
    ALPACA_API_KEY=... ALPACA_SECRET_KEY=... GIST_ID=... GIST_TOKEN=... \
    ALPACA_PAPER=true DRY_RUN=true LIVE_READY=false \
    pytest tests/test_integration.py -v -s

These tests hit real external APIs.  Do NOT run in CI.
"""
import os
import pytest


# Skip entire module if integration env vars are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("ALPACA_API_KEY"),
    reason="Integration env vars not set — skipping",
)


def test_alpaca_client_get_clock():
    """AlpacaClient.get_clock() returns a dict with 'is_open' key."""
    from src.alpaca_client import AlpacaClient
    client = AlpacaClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )
    clock = client.get_clock()
    assert "is_open" in clock


def test_alpaca_client_get_account():
    """AlpacaClient.get_account() returns equity > 0 for a funded paper account."""
    from src.alpaca_client import AlpacaClient
    client = AlpacaClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )
    account = client.get_account()
    assert float(account["equity"]) > 0


def test_state_load_save_roundtrip():
    """load_state → modify → save_state → load_state produces identical state."""
    from src.state import load_state, save_state
    gist_id = os.environ["GIST_ID"]
    token   = os.environ["GIST_TOKEN"]

    state, sell_only = load_state(gist_id, token)
    assert not sell_only

    original_pnl = state.cumulative_realized_pnl
    from dataclasses import replace
    modified = replace(state, cumulative_realized_pnl=original_pnl + 0.01)

    ok = save_state(modified, gist_id, token)
    assert ok

    reloaded, _ = load_state(gist_id, token)
    assert abs(reloaded.cumulative_realized_pnl - (original_pnl + 0.01)) < 1e-6

    # Restore original value
    save_state(state, gist_id, token)


def test_full_dry_run():
    """Run the full bot in DRY_RUN mode — must exit 0 with no orders submitted."""
    import subprocess, sys
    env = {**os.environ, "DRY_RUN": "true", "ALPACA_PAPER": "true", "LIVE_READY": "false"}
    result = subprocess.run(
        [sys.executable, "-m", "src.main"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Bot exited {result.returncode}:\n{result.stderr}"
    assert "Submitting order" not in result.stdout or "DRY_RUN" in result.stdout
