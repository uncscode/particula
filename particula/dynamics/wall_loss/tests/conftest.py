"""Pytest configuration for wall loss tests.

This module prevents ``pytest`` from collecting the legacy
``wall_loss_strategies_test.py`` file under this package.

The tests have been moved to
``particula/dynamics/tests/wall_loss_strategies_test.py``
to avoid rare import edge cases when running ``pytest particula``
via ADW tooling.
"""

# Ignore the legacy test module in this package; see module docstring.
collect_ignore_glob = ["wall_loss_strategies_test.py"]
