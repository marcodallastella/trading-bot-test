# src/state.py
import json
import time

import requests

from src import config
from src.logger import get_logger
from src.models import State, TrailingStop, WashSaleEntry

log = get_logger(__name__)

_REQUIRED_FIELDS = frozenset({
    "rolling_peak_equity",
    "rolling_peak_timestamp",
    "last_run_timestamp",
    "ticker_last_buy_date",
    "wash_sale_blacklist",
    "trailing_stops_active",
    "cumulative_realized_pnl",
    "drawdown_halt_active",
})


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _fetch_raw(gist_id: str, token: str) -> str | None:
    """GET the Gist and return the state file content string. None on any error."""
    url = f"{config.GIST_API_BASE_URL}/{gist_id}"
    try:
        resp = requests.get(url, headers=_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()["files"][config.STATE_GIST_FILENAME]["content"]
    except Exception as exc:
        log.error(f"Gist fetch error: {exc}")
        return None


def _parse(content: str) -> State | None:
    """Deserialise JSON string → State. None on any error."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.critical(f"State JSON malformed: {exc}")
        return None

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        log.critical(f"State missing required fields: {missing}")
        return None

    try:
        return State(
            rolling_peak_equity=float(data["rolling_peak_equity"]),
            rolling_peak_timestamp=str(data["rolling_peak_timestamp"]),
            last_run_timestamp=str(data["last_run_timestamp"]),
            ticker_last_buy_date=dict(data["ticker_last_buy_date"]),
            wash_sale_blacklist={
                t: WashSaleEntry(**v)
                for t, v in data["wash_sale_blacklist"].items()
            },
            trailing_stops_active={
                t: TrailingStop(**v)
                for t, v in data["trailing_stops_active"].items()
            },
            cumulative_realized_pnl=float(data["cumulative_realized_pnl"]),
            drawdown_halt_active=bool(data["drawdown_halt_active"]),
        )
    except Exception as exc:
        log.critical(f"State deserialisation error: {exc}")
        return None


def _sell_only_state() -> State:
    """
    Placeholder returned when loading fails.
    drawdown_halt_active=True ensures no buys are placed even if
    caller forgets to check sell_only flag.
    """
    return State(
        rolling_peak_equity=0.0,
        rolling_peak_timestamp="1970-01-01",
        last_run_timestamp="1970-01-01T00:00:00Z",
        ticker_last_buy_date={},
        wash_sale_blacklist={},
        trailing_stops_active={},
        cumulative_realized_pnl=0.0,
        drawdown_halt_active=True,
    )


def load_state(gist_id: str, token: str) -> tuple[State, bool]:
    """
    Returns (state, sell_only_mode).
    sell_only_mode=True: state is unusable; bot must not place buys this run.
    Never raises.
    """
    try:
        raw = _fetch_raw(gist_id, token)
        if raw is None:
            log.warning(f"Gist fetch failed — retrying in {config.GIST_RETRY_DELAY_SECONDS}s")
            time.sleep(config.GIST_RETRY_DELAY_SECONDS)
            raw = _fetch_raw(gist_id, token)

        if raw is None:
            log.critical("SELL-ONLY MODE: state fetch failed after retry")
            return _sell_only_state(), True

        state = _parse(raw)
        if state is None:
            log.critical("SELL-ONLY MODE: state parse failed")
            return _sell_only_state(), True

        log.info(
            f"State loaded: peak=${state.rolling_peak_equity} ({state.rolling_peak_timestamp})"
            f" | cum_pnl=${state.cumulative_realized_pnl:.2f}"
            f" | trailing_stops={len(state.trailing_stops_active)}"
            f" | wash_sale={len(state.wash_sale_blacklist)}"
        )
        return state, False

    except Exception as exc:
        log.critical(f"SELL-ONLY MODE: unexpected error in load_state: {exc}")
        return _sell_only_state(), True


def save_state(state: State, gist_id: str, token: str) -> bool:
    """Serialise State → JSON and PATCH the Gist. Returns True on success. Never raises."""
    data = {
        "rolling_peak_equity": round(state.rolling_peak_equity, 4),
        "rolling_peak_timestamp": state.rolling_peak_timestamp,
        "last_run_timestamp": state.last_run_timestamp,
        "ticker_last_buy_date": state.ticker_last_buy_date,
        "wash_sale_blacklist": {
            t: {"sold_date": ws.sold_date, "expires": ws.expires}
            for t, ws in state.wash_sale_blacklist.items()
        },
        "trailing_stops_active": {
            t: {
                "entry_price": round(ts.entry_price, 4),
                "peak_price": round(ts.peak_price, 4),
                "activated": ts.activated,
            }
            for t, ts in state.trailing_stops_active.items()
        },
        "cumulative_realized_pnl": round(state.cumulative_realized_pnl, 4),
        "drawdown_halt_active": state.drawdown_halt_active,
    }
    url = f"{config.GIST_API_BASE_URL}/{gist_id}"
    payload = {
        "files": {
            config.STATE_GIST_FILENAME: {"content": json.dumps(data, indent=2)}
        }
    }
    try:
        resp = requests.patch(url, json=payload, headers=_headers(token), timeout=10)
        resp.raise_for_status()
        log.info("State saved to Gist")
        return True
    except Exception as exc:
        log.error(f"State save failed: {exc}")
        return False
