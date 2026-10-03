"""Runtime regressions for the CPU dilution example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/Examples/cpu_dilution.py"
)


def _load_example():
    """Load the standalone example without package-importing docs."""
    spec = importlib.util.spec_from_file_location(
        "cpu_dilution_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_dilution_example_executes_exact_public_api_decay() -> None:
    """Example decays every concentration domain exactly."""
    result = _load_example().run_example()
    factor = np.exp(-result.coefficient * result.time_step)

    assert np.isfinite(factor)
    assert 0.0 < factor < 1.0
    for initial, final in (
        (result.particle_initial, result.particle_final),
        (result.partitioning_initial, result.partitioning_final),
        (result.gas_only_initial, result.gas_only_final),
    ):
        assert initial.shape == final.shape
        assert np.all(initial > 0.0)
        assert np.all(final > 0.0)
        assert np.all(final < initial)
        npt.assert_allclose(final, initial * factor, rtol=1e-12, atol=0.0)


def test_cpu_dilution_example_results_are_isolated_snapshots() -> None:
    """Fresh calls and all initial/final snapshots own independent arrays."""
    example = _load_example()
    first = example.run_example()
    second = example.run_example()

    for initial, final, other in (
        (first.particle_initial, first.particle_final, second.particle_initial),
        (
            first.partitioning_initial,
            first.partitioning_final,
            second.partitioning_initial,
        ),
        (first.gas_only_initial, first.gas_only_final, second.gas_only_initial),
    ):
        assert not np.shares_memory(initial, final)
        assert not np.shares_memory(initial, other)
        snapshot = other.copy()
        initial[...] = -1.0
        npt.assert_array_equal(other, snapshot)


def test_cpu_dilution_example_result_rejects_metadata_reassignment() -> None:
    """Example result keeps its execution metadata immutable."""
    result = _load_example().run_example()
    with pytest.raises(AttributeError, match="ExampleResult is immutable"):
        result.coefficient = 1.0


def test_cpu_dilution_example_main_reports_all_domains(capsys) -> None:
    """Example command reports metadata and before/after snapshots."""
    _load_example().main()
    output = capsys.readouterr().out.lower()
    for label in (
        "coefficient",
        "duration",
        "decay factor",
        "particle before",
        "particle after",
        "partitioning before",
        "partitioning after",
        "gas-only before",
        "gas-only after",
    ):
        assert label in output
