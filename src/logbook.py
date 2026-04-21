# src/logbook.py
import requests

from src import config
from src.logger import get_logger
from src.models import LogbookRow

log = get_logger(__name__)

CSV_HEADER = (
    "timestamp,ticker,price,sma_50,pir,atr_pct,"
    "signal_mult,vol_factor,regime_mult,kelly_base,"
    "final_pct,action,order_notional,fill_price,reason"
)


def _row_to_csv(row: LogbookRow) -> str:
    reason = f'"{row.reason}"' if "," in row.reason else row.reason
    return ",".join([
        row.timestamp, row.ticker, row.price, row.sma_50,
        row.pir, row.atr_pct, row.signal_mult, row.vol_factor,
        row.regime_mult, row.kelly_base, row.final_pct,
        row.action, row.order_notional, row.fill_price, reason,
    ])


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def append_decisions(rows: list[LogbookRow], gist_id: str, token: str) -> bool:
    """
    Fetch current decisions.csv from Gist, append new rows, PATCH back.
    Returns True on successful PATCH. Never raises.
    GET failure is non-fatal — treated as an empty file.
    """
    url = f"{config.GIST_API_BASE_URL}/{gist_id}"
    hdrs = _headers(token)

    existing = ""
    try:
        resp = requests.get(url, headers=hdrs, timeout=10)
        resp.raise_for_status()
        files = resp.json().get("files", {})
        existing = files.get(config.DECISIONS_GIST_FILENAME, {}).get("content", "")
    except Exception as exc:
        log.warning(f"Could not fetch decisions Gist (will start fresh): {exc}")

    if not existing.strip():
        content = CSV_HEADER + "\n"
    else:
        content = existing if existing.endswith("\n") else existing + "\n"

    for row in rows:
        content += _row_to_csv(row) + "\n"

    payload = {
        "files": {config.DECISIONS_GIST_FILENAME: {"content": content}}
    }
    try:
        resp = requests.patch(url, json=payload, headers=hdrs, timeout=10)
        resp.raise_for_status()
        log.info(f"Appended {len(rows)} row(s) to decisions logbook")
        return True
    except Exception as exc:
        log.error(f"Failed to save decisions logbook: {exc}")
        return False
