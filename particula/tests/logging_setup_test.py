"""Tests for the logging setup module."""

import logging
import logging.handlers

logger = logging.getLogger("particula")


def test_setup_configures_logger():
    """Test that the logger name is set to 'particula'."""
    # Arrange
    expected_logger_name = "particula"

    # Assert
    assert logger.name == expected_logger_name

    logger.info("Logger Test: This is a test message.")
    logger.debug("Logger Test: This is a debug message.")
    logger.error("Logger Test: This is an error message.")
    logger.warning("Logger Test: This is a warning message.")
    logger.critical("Logger Test: This is a critical message.")
    logger.exception("Logger Test: This is an exception message.")
    assert True  # if no exception is raised, all logging levels are working
