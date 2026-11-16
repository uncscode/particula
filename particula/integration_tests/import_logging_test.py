"""Import must preserve application-owned logging and read-only installs."""

import subprocess
import sys


def test_import_does_not_configure_logging_or_open_log_files():
    """A fresh import neither configures root handlers nor opens file handlers."""
    probe = """
import logging
import logging.config
root = logging.getLogger()
handler = logging.StreamHandler()
root.addHandler(handler)
root.setLevel(logging.WARNING)
other = logging.getLogger("application")
other.setLevel(logging.INFO)
def forbidden(*args, **kwargs):
    raise AssertionError("import attempted logging configuration or file output")
logging.config.dictConfig = forbidden
logging.FileHandler.__init__ = forbidden
import particula
assert root.handlers == [handler]
assert root.level == logging.WARNING
assert other.level == logging.INFO and not other.disabled
assert not any(isinstance(h, logging.FileHandler) for h in particula.logger.handlers)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
