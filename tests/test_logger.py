import logging
import sys
import pytest
from src.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)


def test_logger_has_exactly_one_handler_after_first_call():
    # Use a unique name to avoid contamination from other tests
    logger = get_logger("test.one_handler")
    assert len(logger.handlers) == 1


def test_get_logger_idempotent_no_duplicate_handlers():
    logger1 = get_logger("test.idempotent")
    logger2 = get_logger("test.idempotent")
    assert logger1 is logger2
    assert len(logger2.handlers) == 1


def test_handler_writes_to_stdout():
    logger = get_logger("test.stdout_check")
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout


def test_logged_info_message_appears_in_stdout(capsys):
    logger = get_logger("test.capsys")
    logger.info("hello from test")
    captured = capsys.readouterr()
    assert "hello from test" in captured.out


def test_captured_output_contains_logger_name(capsys):
    logger = get_logger("test.named_logger")
    logger.info("checking name presence")
    captured = capsys.readouterr()
    assert "test.named_logger" in captured.out
