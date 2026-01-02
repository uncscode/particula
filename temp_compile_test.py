"""Compile check for condensation strategies module.

Ensures ``particula/dynamics/condensation/condensation_strategies.py``
compiles after staggered batching and nan-safety updates without running
runtime-heavy tests.
"""

import py_compile


def test_compile_condensation():
    """Compile the condensation strategies module to catch syntax errors.

    Raises:
        py_compile.PyCompileError: If the module fails to compile.
    """
    py_compile.compile(
        "particula/dynamics/condensation/condensation_strategies.py",
        doraise=True,
    )
