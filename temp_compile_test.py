"""Compile check for condensation strategies module.

This module verifies that ``particula/dynamics/condensation/
condensation_strategies.py`` still compiles cleanly after recent updates.
"""

import py_compile


def test_compile_condensation():
    """Compile the condensation strategies module to ensure
    syntactic validity.
    """
    py_compile.compile(
        "particula/dynamics/condensation/condensation_strategies.py",
        doraise=True,
    )
