# tests/test_logbook.py
import json
import pytest
from unittest.mock import Mock
import requests as requests_lib
from src.logbook import append_decisions, CSV_HEADER
from src.models import LogbookRow
from src import config


# ── helpers ───────────────────────────────────────────────────────────────────

def make_row(ticker: str = "AAPL", action: str = "BUY") -> LogbookRow:
    return LogbookRow(
        timestamp="2026-04-18T15:00:00Z",
        ticker=ticker,
        price="142.80",
        sma_50="139.20",
        pir="0.08",
        atr_pct="0.035",
        signal_mult="1.5",
        vol_factor="1.0",
        regime_mult="1.0",
        kelly_base="0.097",
        final_pct="0.1455",
        action=action,
        order_notional="11.64",
        fill_price="142.85",
        reason="deep dip override",
    )


def gist_with_content(content: str) -> Mock:
    m = Mock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "files": {
            config.DECISIONS_GIST_FILENAME: {"content": content}
        }
    }
    return m


def empty_gist() -> Mock:
    """Gist exists but the decisions file is absent."""
    m = Mock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"files": {}}
    return m


def patch_mock() -> Mock:
    m = Mock()
    m.raise_for_status.return_value = None
    return m


# ── CSV header ────────────────────────────────────────────────────────────────

def test_csv_header_contains_required_columns():
    cols = CSV_HEADER.split(",")
    for col in ["timestamp", "ticker", "action", "reason", "pir", "kelly_base"]:
        assert col in cols, f"Expected column '{col}' in CSV_HEADER"


# ── append_decisions ──────────────────────────────────────────────────────────

def test_append_creates_header_on_first_run(mocker):
    """When Gist has no existing file → header is written before the row."""
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    append_decisions([make_row()], "gist123", "token_abc")
    sent_content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    assert sent_content.startswith(CSV_HEADER)


def test_append_row_appears_after_header_on_first_run(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    append_decisions([make_row(ticker="AMD")], "gist123", "token_abc")
    content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    lines = content.strip().split("\n")
    assert lines[0] == CSV_HEADER
    assert "AMD" in lines[1]


def test_append_preserves_existing_rows(mocker):
    existing = CSV_HEADER + "\n2026-04-17T15:00:00Z,AAPL,,,,,,,,,,,,,old row\n"
    mocker.patch("src.logbook.requests.get", return_value=gist_with_content(existing))
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    append_decisions([make_row(ticker="NVDA")], "gist123", "token_abc")
    content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    assert "old row" in content
    assert "NVDA" in content


def test_append_does_not_duplicate_header(mocker):
    """When existing content already has header, don't write it again."""
    existing = CSV_HEADER + "\n2026-04-17T15:00:00Z,AAPL,,,,,,,,,,,,,row1\n"
    mocker.patch("src.logbook.requests.get", return_value=gist_with_content(existing))
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    append_decisions([make_row()], "gist123", "token_abc")
    content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    assert content.count(CSV_HEADER) == 1


def test_append_multiple_rows(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    rows = [make_row("AAPL", "BUY"), make_row("TSLA", "SKIP"), make_row("NVDA", "HOLD")]
    append_decisions(rows, "gist123", "token_abc")
    content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    assert "AAPL" in content
    assert "TSLA" in content
    assert "NVDA" in content


def test_append_returns_true_on_success(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    result = append_decisions([make_row()], "gist123", "token_abc")
    assert result is True


def test_append_returns_false_on_patch_failure(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    bad_patch = Mock()
    bad_patch.raise_for_status.side_effect = requests_lib.exceptions.HTTPError("500")
    mocker.patch("src.logbook.requests.patch", return_value=bad_patch)
    result = append_decisions([make_row()], "gist123", "token_abc")
    assert result is False


def test_append_does_not_raise_on_patch_failure(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mocker.patch(
        "src.logbook.requests.patch",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    try:
        result = append_decisions([make_row()], "gist123", "token_abc")
    except Exception as exc:
        pytest.fail(f"append_decisions raised unexpectedly: {exc}")
    assert result is False


def test_append_proceeds_if_get_fails(mocker):
    """GET failure → treat as empty file; still attempt PATCH."""
    mocker.patch(
        "src.logbook.requests.get",
        side_effect=requests_lib.exceptions.ConnectionError("timeout"),
    )
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    result = append_decisions([make_row()], "gist123", "token_abc")
    mock_patch.assert_called_once()
    assert result is True


def test_append_row_contains_all_fields(mocker):
    mocker.patch("src.logbook.requests.get", return_value=empty_gist())
    mock_patch = mocker.patch("src.logbook.requests.patch", return_value=patch_mock())
    row = make_row("AMD", "BUY")
    append_decisions([row], "gist123", "token_abc")
    content = mock_patch.call_args.kwargs["json"]["files"][config.DECISIONS_GIST_FILENAME]["content"]
    data_line = [l for l in content.split("\n") if "AMD" in l][0]
    assert "2026-04-18T15:00:00Z" in data_line
    assert "0.097" in data_line   # kelly_base
    assert "deep dip override" in data_line
