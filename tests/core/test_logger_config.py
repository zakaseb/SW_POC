import pytest
import os
import logging
from unittest.mock import patch, MagicMock
import importlib  # For reloading the module

# Module to test
from core import logger_config  # Initial import to allow reload

# Store original handlers of the root logger to restore them after tests
original_rag_app_handlers = None
original_rag_app_level = None
rag_app_logger_name = "rag_app"


@pytest.fixture(autouse=True)
def manage_rag_app_logger_state(request):
    """
    Fixture to ensure the 'rag_app' logger is reset before and after each test.
    This is crucial because logging configuration is global.
    """
    global original_rag_app_handlers, original_rag_app_level

    logger_to_manage = logging.getLogger(rag_app_logger_name)

    # Save current state if not already saved (only first time)
    if original_rag_app_handlers is None:
        original_rag_app_handlers = logger_to_manage.handlers[:]
    if original_rag_app_level is None:
        original_rag_app_level = logger_to_manage.level

    # Reset before each test: remove handlers added by previous tests and reset level
    logger_to_manage.handlers = []
    logger_to_manage.setLevel(logging.NOTSET)  # Reset level to default
    # Also need to ensure the logger_config module itself is reloaded if it caches logger instances or states
    importlib.reload(logger_config)

    yield  # Run the test

    # Restore after each test
    logger_to_manage.handlers = original_rag_app_handlers
    logger_to_manage.setLevel(original_rag_app_level)
    # Clear any specific loggers created under rag_app to avoid interference
    # This is a bit more involved, might need to iterate logging.Logger.manager.loggerDict
    # For now, resetting handlers and level of the main app logger is a good start.


@pytest.fixture
def clean_environ():
    """Fixture to provide a clean environment for os.getenv tests."""
    original_environ = os.environ.copy()
    os.environ.clear()  # Clear all for the test
    yield
    os.environ.clear()  # Clear again
    os.environ.update(original_environ)  # Restore original


# --- Tests for setup_logging ---


def test_setup_logging_default_level(clean_environ):
    # Ensure LOG_LEVEL is not set
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]

    importlib.reload(
        logger_config
    )  # Reload to re-evaluate os.getenv at module level if any
    logger_config.setup_logging()

    app_logger = logging.getLogger(rag_app_logger_name)
    assert app_logger.level == logging.INFO
    assert len(app_logger.handlers) == 1
    assert isinstance(app_logger.handlers[0], logging.StreamHandler)
    assert app_logger.handlers[0].formatter is not None
    # Check formatter string if it's simple enough or type
    assert isinstance(app_logger.handlers[0].formatter, logging.Formatter)
    # Example check of format string, might be brittle if format changes often
    assert (
        app_logger.handlers[0].formatter._fmt
        == "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def test_setup_logging_env_var_debug_level(clean_environ):
    os.environ["LOG_LEVEL"] = "DEBUG"
    importlib.reload(logger_config)
    logger_config.setup_logging()

    app_logger = logging.getLogger(rag_app_logger_name)
    assert app_logger.level == logging.DEBUG
    assert len(app_logger.handlers) == 1  # Ensure it doesn't add duplicate handlers


def test_setup_logging_env_var_invalid_level(clean_environ):
    os.environ["LOG_LEVEL"] = "INVALID_LEVEL"
    importlib.reload(logger_config)

    # Patch logger.info within setup_logging to check for the fallback message
    with patch.object(logging.getLogger(rag_app_logger_name), "info") as mock_log_info:
        logger_config.setup_logging()

    app_logger = logging.getLogger(rag_app_logger_name)
    assert app_logger.level == logging.INFO  # Should default to INFO

    # Check if the info message about initialization includes the level it set to
    # This depends on the logger.info call inside setup_logging
    mock_log_info.assert_any_call("Logging initialized with level INFO")


def test_setup_logging_idempotency(clean_environ):
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]
    importlib.reload(logger_config)

    logger_config.setup_logging()
    app_logger = logging.getLogger(rag_app_logger_name)
    initial_handler_count = len(app_logger.handlers)
    assert initial_handler_count == 1  # Should have one handler after first call

    logger_config.setup_logging()  # Call again
    assert (
        len(app_logger.handlers) == initial_handler_count
    )  # Handler count should not increase


# --- Tests for get_logger ---


def test_get_logger_creates_child_logger(clean_environ):
    # First setup the main app logger
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]
    importlib.reload(logger_config)
    logger_config.setup_logging()

    module_logger = logger_config.get_logger("my_test_module")
    assert module_logger.name == f"{rag_app_logger_name}.my_test_module"

    # Child logger should inherit level from parent if not set explicitly
    # The parent "rag_app" logger is set to INFO by default by setup_logging
    assert module_logger.getEffectiveLevel() == logging.INFO

    # Check it has handlers (should propagate to parent's handlers)
    # This is a bit tricky as child loggers don't have handlers themselves by default,
    # they pass messages to parent handlers if propagate=True (default).
    # We can check propagation.
    assert module_logger.propagate is True
    # And that the parent "rag_app" logger (which it will propagate to) has a handler.
    assert len(logging.getLogger(rag_app_logger_name).handlers) > 0


def test_get_logger_hierarchy_level_propagation(clean_environ):
    os.environ["LOG_LEVEL"] = "DEBUG"  # Set parent to DEBUG
    importlib.reload(logger_config)
    logger_config.setup_logging()

    parent_app_logger = logging.getLogger(rag_app_logger_name)
    assert parent_app_logger.level == logging.DEBUG

    child_logger = logger_config.get_logger("module.submodule")
    assert child_logger.name == f"{rag_app_logger_name}.module.submodule"
    assert child_logger.getEffectiveLevel() == logging.DEBUG  # Inherits from "rag_app"

    # Ensure it doesn't have its own handlers unless explicitly added
    assert len(child_logger.handlers) == 0
    assert child_logger.propagate is True
    assert len(parent_app_logger.handlers) == 1  # Parent has the handler
