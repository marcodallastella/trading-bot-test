# tests/conftest.py
import json
import copy
import pytest
from unittest.mock import Mock
from src.models import State, TrailingStop, WashSaleEntry

VALID_STATE_DICT: dict = {
    "rolling_peak_equity": 104.30,
    "rolling_peak_timestamp": "2026-03-15",
    "last_run_timestamp": "2026-04-18T15:00:00Z",
    "ticker_last_buy_date": {"AAPL": "2026-04-15"},
    "wash_sale_blacklist": {
        "MSFT": {"sold_date": "2026-03-20", "expires": "2026-04-20"}
    },
    "trailing_stops_active": {
        "PLTR": {"entry_price": 22.40, "peak_price": 28.15, "activated": "2026-04-12"}
    },
    "cumulative_realized_pnl": 8.42,
    "drawdown_halt_active": False,
}


@pytest.fixture
def valid_state_dict() -> dict:
    return copy.deepcopy(VALID_STATE_DICT)


@pytest.fixture
def valid_state() -> State:
    return State(
        rolling_peak_equity=104.30,
        rolling_peak_timestamp="2026-03-15",
        last_run_timestamp="2026-04-18T15:00:00Z",
        ticker_last_buy_date={"AAPL": "2026-04-15"},
        wash_sale_blacklist={
            "MSFT": WashSaleEntry(sold_date="2026-03-20", expires="2026-04-20")
        },
        trailing_stops_active={
            "PLTR": TrailingStop(entry_price=22.40, peak_price=28.15, activated="2026-04-12")
        },
        cumulative_realized_pnl=8.42,
        drawdown_halt_active=False,
    )


def make_gist_response(content_dict: dict) -> Mock:
    mock = Mock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "files": {
            "trading_bot_state.json": {
                "content": json.dumps(content_dict)
            }
        }
    }
    return mock


@pytest.fixture
def gist_ok(valid_state_dict) -> Mock:
    return make_gist_response(valid_state_dict)
