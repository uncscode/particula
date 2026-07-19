"""Regression tests for the independent condensation parity walkthrough."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu import WARP_AVAILABLE
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "Examples"
    / "gpu_condensation_parity_walkthrough.py"
)


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the CPU-safe published walkthrough module."""
    monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
    sys.modules.pop("gpu_condensation_parity_walkthrough", None)
    return importlib.import_module("gpu_condensation_parity_walkthrough")


def test_fixture_and_builders_are_fp64_readonly_and_non_aliasing(
    example_module: types.ModuleType,
) -> None:
    """Templates remain immutable and each detached builder owns its state."""
    fixture = example_module.build_fixture()
    oracle = example_module.build_oracle_input(fixture)
    source = example_module.build_warp_source(fixture)
    assert fixture.names == ("Water", "Organic")
    assert fixture.masses.shape == (2, 2, 2)
    assert fixture.gas_concentration.shape == (2, 2)
    assert fixture.thermodynamic_parameters.shape == (2, 4)
    assert fixture.thermodynamic_modes.dtype == np.int32
    for values in (
        fixture.masses,
        fixture.gas_concentration,
        fixture.latent_heat,
    ):
        assert values.dtype == np.float64
        assert not values.flags.writeable
    assert fixture.partitioning.dtype == np.bool_
    assert not fixture.partitioning.flags.writeable
    assert not fixture.thermodynamic_modes.flags.writeable
    assert not np.shares_memory(fixture.masses, oracle.particles.masses)
    assert not np.shares_memory(fixture.masses, source.particles.masses)
    assert not np.shares_memory(
        oracle.particles.masses, source.particles.masses
    )
    assert oracle.gas.name is not source.gas.name


def test_oracle_has_four_substep_uptake_evaporation_coupling_and_energy(
    example_module: types.ModuleType,
) -> None:
    """The independent result retains raw proposal and applied-total evidence."""
    fixture = example_module.build_fixture()
    result = example_module.run_oracle(
        example_module.build_oracle_input(fixture)
    )
    assert np.any(result.total_mass_transfer > 0.0)
    assert np.any(result.total_mass_transfer < 0.0)
    assert np.any(result.gas_concentration != fixture.gas_concentration)
    assert np.any(result.energy_transfer != 0.0)
    npt.assert_allclose(
        result.energy_transfer,
        np.sum(result.total_mass_transfer, axis=1)
        * fixture.latent_heat[None, :],
        rtol=1e-12,
        atol=1e-30,
    )
    assert not np.array_equal(result.raw_proposal, result.total_mass_transfer)
    assert len(result.substeps) == 4
    assert all(substep.time_step == 0.08 / 4.0 for substep in result.substeps)
    for earlier, later in zip(
        result.substeps, result.substeps[1:], strict=False
    ):
        gas_after = earlier.gas_concentration_before - np.sum(
            earlier.finalized_transfer * fixture.concentration[:, :, None],
            axis=1,
        )
        gas_after = np.maximum(gas_after, 0.0)
        npt.assert_allclose(later.gas_concentration_before, gas_after)
    assert not np.array_equal(
        result.substeps[0].raw_proposal,
        result.substeps[1].raw_proposal,
    )


def test_oracle_does_not_mutate_supplied_source(
    example_module: types.ModuleType,
) -> None:
    """Oracle execution preserves caller-owned source particles and gas."""
    source = example_module.build_oracle_input()
    initial_masses = source.particles.masses.copy()
    initial_gas_concentration = source.gas.concentration.copy()

    example_module.run_oracle(source)

    npt.assert_array_equal(source.particles.masses, initial_masses)
    npt.assert_array_equal(source.gas.concentration, initial_gas_concentration)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", np.array([np.nan, 298.15]), "finite"),
        ("pressure", np.array([np.inf, 98000.0]), "finite"),
        ("density", np.array([0.0, 1100.0]), "positive"),
        ("gas_concentration", np.full((2, 2), -1.0), "nonnegative"),
    ],
)
def test_fixture_validation_rejects_nonfinite_and_invalid_physical_values(
    example_module: types.ModuleType,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    """Injected fixtures enforce direct-step numeric and physical domains."""
    fixture = replace(example_module.build_fixture(), **{field: value})
    with pytest.raises(ValueError, match=message):
        example_module.build_oracle_input(fixture)


def test_fixture_validation_rejects_unsupported_thermodynamic_mode(
    example_module: types.ModuleType,
) -> None:
    """The constant-mode oracle rejects thermodynamics it cannot model."""
    modes = np.array([1, 0], dtype=np.int32)
    fixture = replace(example_module.build_fixture(), thermodynamic_modes=modes)

    with pytest.raises(ValueError, match="constant mode"):
        example_module.build_oracle_input(fixture)


def test_disabled_or_unavailable_warp_completes_oracle_without_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """No-kernel paths run the oracle but defer all direct runtime activity."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", False)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module, "_load_gpu_runtime", lambda: pytest.fail("loaded")
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        lambda *a, **k: pytest.fail("converted"),
    )
    result = example_module.run_example()
    assert "oracle completed; no kernel ran" in result.output
    assert result.particle_data is None
    assert result.oracle.total_mass_transfer.shape == (2, 2, 2)
    assert [item.category for item in result.acceptance] == [
        "physics",
        "conservation",
        "energy",
    ]
    assert all(item.status == "unavailable" for item in result.acceptance)
    assert all(
        "no-Warp observations" in item.diagnostic for item in result.acceptance
    )


def test_force_disabled_warp_defers_runtime_after_oracle(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Forced CPU-only execution never loads or converts a Warp source."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.setattr(
        example_module, "_load_gpu_runtime", lambda: pytest.fail("loaded")
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_gas_data",
        lambda *args, **kwargs: pytest.fail("converted"),
    )
    result = example_module.run_example()
    assert "oracle completed; no kernel ran" in result.output
    assert all(item.status == "unavailable" for item in result.acceptance)


class _FakeArray:
    """Metadata-bearing minimal Warp array fake."""

    def __init__(self, values: Any, dtype: object, device: str) -> None:
        self.values = np.asarray(values)
        self.dtype = dtype
        self.device = device
        self.shape = self.values.shape

    def numpy(self) -> np.ndarray:
        """Return fake device values."""
        return self.values


class _FakeWP:
    """Minimal allocation and synchronization runtime fake."""

    float64 = object()
    int32 = object()
    events: list[str] = []

    @classmethod
    def array(cls, values: Any, dtype: object, device: str) -> _FakeArray:
        """Construct a fake array."""
        return _FakeArray(values, dtype, device)

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], dtype: object, device: str
    ) -> _FakeArray:
        """Allocate fake zero storage."""
        return _FakeArray(np.zeros(shape), dtype, device)

    @classmethod
    def synchronize_device(cls, device: str) -> None:
        """Record a required pre-readback synchronization."""
        cls.events.append(f"sync:{device}")


class _FakeScratch:
    """Retain caller-owned scratch identities."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeThermodynamics:
    """Retain thermodynamic sidecar identities."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _fake_gas_with_vapor_pressure(value: Any, **kwargs: Any) -> Any:
    """Attach GPU-only vapor-pressure state to a fake converted gas object."""
    value.vapor_pressure = _FakeArray(
        kwargs["vapor_pressure"], _FakeWP.float64, "cpu"
    )
    return value


def test_fake_enabled_route_has_explicit_sidecars_and_synchronized_readback(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Enabled route uses direct arrays, caller-owned sidecars, and syncs."""
    calls: list[dict[str, Any]] = []
    _FakeWP.events = []

    def step(particles: Any, gas: Any, **kwargs: Any) -> tuple[Any, Any]:
        calls.append(kwargs)
        kwargs["scratch_buffers"].work_mass_transfer.values.fill(2.0)
        kwargs["scratch_buffers"].total_mass_transfer.values.fill(3.0)
        kwargs["energy_transfer"].values.fill(4.0)
        return particles, kwargs["scratch_buffers"].total_mass_transfer

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP, step, _FakeScratch, _FakeThermodynamics),
    )
    monkeypatch.setattr(
        example_module, "to_warp_particle_data", lambda value, **k: value
    )
    monkeypatch.setattr(
        example_module, "to_warp_gas_data", _fake_gas_with_vapor_pressure
    )
    monkeypatch.setattr(
        example_module, "from_warp_particle_data", lambda value: value
    )
    monkeypatch.setattr(
        example_module, "from_warp_gas_data", lambda value, **k: value
    )
    result = example_module.run_example()
    assert len(calls) == 1
    call = calls[0]
    assert result.scratch_buffers is call["scratch_buffers"]
    assert call["temperature"].shape == call["pressure"].shape == (2,)
    assert call["temperature"].dtype is _FakeWP.float64
    assert call["thermodynamics"].modes.dtype is _FakeWP.int32
    assert call["scratch_buffers"].total_mass_transfer.shape == (2, 2, 2)
    assert call["energy_transfer"].shape == (2, 2)
    assert len(_FakeWP.events) == 6
    npt.assert_allclose(result.raw_proposal, 2.0)
    npt.assert_allclose(result.total_mass_transfer, 3.0)
    npt.assert_allclose(result.energy_transfer, 4.0)
    npt.assert_allclose(result.vapor_pressure, 0.0)
    assert [item.category for item in result.acceptance] == [
        "physics",
        "conservation",
        "energy",
    ]
    assert [item.status for item in result.acceptance] == [
        "failed",
        "passed",
        "failed",
    ]
    for prefix in (
        "physics: failed",
        "conservation: passed",
        "energy: failed",
    ):
        assert any(line.startswith(prefix) for line in result.output)


def test_oracle_completes_before_runtime_and_ignores_warp_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Detached Warp-source mutation cannot alter the completed oracle result."""
    events: list[str] = []
    original_oracle = example_module.run_oracle
    original_source = example_module.build_warp_source
    baseline_oracle = original_oracle(example_module.build_oracle_input())
    untouched_source = original_source()

    def record_oracle(*args: Any, **kwargs: Any) -> Any:
        events.append("oracle")
        return original_oracle(*args, **kwargs)

    def mutate_source(*args: Any, **kwargs: Any) -> Any:
        source = original_source(*args, **kwargs)
        source.particles.masses.fill(99.0)
        events.append("source")
        return source

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.setattr(example_module, "run_oracle", record_oracle)
    monkeypatch.setattr(example_module, "build_warp_source", mutate_source)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (
            _FakeWP,
            lambda particles, gas, **kwargs: (
                particles,
                kwargs["scratch_buffers"].total_mass_transfer,
            ),
            _FakeScratch,
            _FakeThermodynamics,
        ),
    )
    monkeypatch.setattr(
        example_module, "to_warp_particle_data", lambda value, **kwargs: value
    )
    monkeypatch.setattr(
        example_module, "to_warp_gas_data", _fake_gas_with_vapor_pressure
    )
    monkeypatch.setattr(
        example_module, "from_warp_particle_data", lambda value: value
    )
    monkeypatch.setattr(
        example_module, "from_warp_gas_data", lambda value, **kwargs: value
    )
    result = example_module.run_example()
    assert events == ["oracle", "source"]
    npt.assert_allclose(result.oracle.masses, baseline_oracle.masses)
    npt.assert_allclose(
        untouched_source.particles.masses,
        example_module.build_fixture().masses,
    )


@pytest.mark.parametrize(
    "stage",
    [
        "loader",
        "particle conversion",
        "gas conversion",
        "allocation",
        "kernel",
    ],
)
def test_enabled_failures_propagate_after_completed_oracle_without_restore(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
    stage: str,
) -> None:
    """Enabled errors preserve type/message and never report fallback success."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    message = f"{stage} failure"
    events: list[str] = []
    original_oracle = example_module.run_oracle

    def record_oracle(*args: Any, **kwargs: Any) -> Any:
        events.append("oracle")
        return original_oracle(*args, **kwargs)

    monkeypatch.setattr(example_module, "run_oracle", record_oracle)
    if stage == "loader":
        monkeypatch.setattr(
            example_module,
            "_load_gpu_runtime",
            lambda: (_ for _ in ()).throw(RuntimeError(message)),
        )
    else:
        monkeypatch.setattr(
            example_module,
            "_load_gpu_runtime",
            lambda: (
                _FakeWP,
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError(message)
                ),
                _FakeScratch,
                _FakeThermodynamics,
            ),
        )
        if stage == "particle conversion":
            monkeypatch.setattr(
                example_module,
                "to_warp_particle_data",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError(message)
                ),
            )
        elif stage == "gas conversion":
            monkeypatch.setattr(
                example_module,
                "to_warp_particle_data",
                lambda value, **kwargs: value,
            )
            monkeypatch.setattr(
                example_module,
                "to_warp_gas_data",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError(message)
                ),
            )
        elif stage == "allocation":

            class FailingAllocationWP(_FakeWP):
                """Fail only when the walkthrough allocates scratch state."""

                @classmethod
                def zeros(
                    cls,
                    shape: tuple[int, ...],
                    dtype: object,
                    device: str,
                ) -> _FakeArray:
                    raise RuntimeError(message)

            monkeypatch.setattr(
                example_module,
                "_load_gpu_runtime",
                lambda: (
                    FailingAllocationWP,
                    lambda *args, **kwargs: pytest.fail("invoked"),
                    _FakeScratch,
                    _FakeThermodynamics,
                ),
            )
            monkeypatch.setattr(
                example_module,
                "to_warp_particle_data",
                lambda value, **kwargs: value,
            )
            monkeypatch.setattr(
                example_module,
                "to_warp_gas_data",
                lambda value, **kwargs: value,
            )
        else:
            monkeypatch.setattr(
                example_module,
                "to_warp_particle_data",
                lambda value, **kwargs: value,
            )
            monkeypatch.setattr(
                example_module,
                "to_warp_gas_data",
                lambda value, **kwargs: value,
            )
        monkeypatch.setattr(
            example_module,
            "from_warp_particle_data",
            lambda *args, **kwargs: pytest.fail("restored"),
        )
    with pytest.raises(RuntimeError, match=message):
        example_module.run_example()
    assert events == ["oracle"]


def test_acceptance_categories_are_independently_evaluated(
    example_module: types.ModuleType,
) -> None:
    """Each criterion can fail without hiding the two unrelated results."""
    fixture = example_module.build_fixture()
    oracle = example_module.run_oracle(
        example_module.build_oracle_input(fixture)
    )
    vapor_pressure = np.broadcast_to(
        fixture.thermodynamic_parameters[:, 0], (2, 2)
    ).copy()
    reference = replace(
        oracle,
        masses=fixture.masses.copy(),
        gas_concentration=fixture.gas_concentration.copy(),
        total_mass_transfer=np.zeros_like(oracle.total_mass_transfer),
        energy_transfer=np.zeros_like(oracle.energy_transfer),
    )

    def statuses(
        checked_fixture: Any,
        checked_vapor_pressure: np.ndarray = vapor_pressure,
        checked_energy: np.ndarray = reference.energy_transfer,
    ) -> list[str]:
        return [
            item.status
            for item in example_module.evaluate_acceptance(
                checked_fixture,
                reference,
                fixture.masses,
                fixture.gas_concentration,
                reference.total_mass_transfer,
                checked_energy,
                checked_vapor_pressure,
            )
        ]

    bad_vapor_pressure = vapor_pressure.copy()
    bad_vapor_pressure[0, 0] += 1.0
    assert statuses(fixture, bad_vapor_pressure) == [
        "failed",
        "passed",
        "passed",
    ]

    bad_energy = reference.energy_transfer.copy()
    bad_energy[0, 0] += 1.0
    assert statuses(fixture, checked_energy=bad_energy) == [
        "passed",
        "passed",
        "failed",
    ]

    conservation_fixture = replace(fixture, masses=fixture.masses.copy() * 2.0)
    assert statuses(conservation_fixture) == ["passed", "failed", "passed"]


def test_acceptance_reports_all_categories_after_multiple_failures(
    example_module: types.ModuleType,
) -> None:
    """Multiple mismatches do not short-circuit later acceptance reporting."""
    fixture = example_module.build_fixture()
    oracle = example_module.run_oracle(
        example_module.build_oracle_input(fixture)
    )
    vapor_pressure = np.broadcast_to(
        fixture.thermodynamic_parameters[:, 0], (2, 2)
    ).copy()
    reference = replace(
        oracle,
        masses=fixture.masses.copy(),
        gas_concentration=fixture.gas_concentration.copy(),
        total_mass_transfer=np.zeros_like(oracle.total_mass_transfer),
        energy_transfer=np.zeros_like(oracle.energy_transfer),
    )
    vapor_pressure[0, 0] += 1.0
    energy = reference.energy_transfer.copy()
    energy[0, 0] += 1.0
    conservation_fixture = replace(fixture, masses=fixture.masses.copy() * 2.0)
    results = example_module.evaluate_acceptance(
        conservation_fixture,
        reference,
        fixture.masses,
        fixture.gas_concentration,
        reference.total_mass_transfer,
        energy,
        vapor_pressure,
    )
    assert [item.category for item in results] == [
        "physics",
        "conservation",
        "energy",
    ]
    assert [item.status for item in results] == ["failed", "failed", "failed"]
    assert all(example_module._format_acceptance(item) for item in results)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_warp_cpu_matches_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Warp CPU satisfies separate physics, conservation, and energy checks."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    result = example_module.run_example(device="cpu")
    assert result.particle_data is not None and result.gas_data is not None
    assert (
        result.total_mass_transfer is not None
        and result.raw_proposal is not None
        and result.scratch_buffers is not None
    )
    assert result.vapor_pressure is not None
    assert [item.status for item in result.acceptance] == ["passed"] * 3
    npt.assert_allclose(
        result.particle_data.masses,
        result.oracle.masses,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.gas_data.concentration,
        result.oracle.gas_concentration,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.total_mass_transfer,
        result.oracle.total_mass_transfer,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.raw_proposal, result.oracle.raw_proposal, rtol=1e-10, atol=1e-30
    )
    npt.assert_allclose(
        result.raw_proposal,
        result.scratch_buffers.work_mass_transfer.numpy(),
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.energy_transfer,
        result.oracle.energy_transfer,
        rtol=1e-10,
        atol=1e-30,
    )
    fixture = example_module.build_fixture()
    npt.assert_array_equal(
        result.vapor_pressure,
        np.broadcast_to(fixture.thermodynamic_parameters[:, 0], (2, 2)),
    )
    inventory = np.sum(
        (result.particle_data.masses - fixture.masses)
        * fixture.concentration[:, :, None],
        axis=1,
    )
    npt.assert_allclose(
        inventory,
        fixture.gas_concentration - result.gas_data.concentration,
        # Four sequential fp64 gas updates accumulate host readback rounding.
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.energy_transfer,
        np.sum(result.total_mass_transfer, axis=1)
        * fixture.latent_heat[None, :],
        rtol=1e-12,
        atol=1e-18,
    )


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_matches_independent_oracle_when_available(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """CUDA is optional additive evidence using the shared skip policy."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    if not WARP_AVAILABLE:
        pytest.skip(CUDA_SKIP_REASON)
    wp = importlib.import_module("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    result = example_module.run_example(device="cuda")
    assert result.particle_data is not None and result.gas_data is not None
    assert result.total_mass_transfer is not None
    assert (
        result.raw_proposal is not None
        and result.energy_transfer is not None
        and result.scratch_buffers is not None
    )
    assert result.vapor_pressure is not None
    assert [item.status for item in result.acceptance] == ["passed"] * 3
    npt.assert_allclose(
        result.particle_data.masses,
        result.oracle.masses,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.gas_data.concentration,
        result.oracle.gas_concentration,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.total_mass_transfer,
        result.oracle.total_mass_transfer,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.raw_proposal,
        result.oracle.raw_proposal,
        rtol=1e-10,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.energy_transfer,
        result.oracle.energy_transfer,
        rtol=1e-10,
        atol=1e-30,
    )
    fixture = example_module.build_fixture()
    npt.assert_array_equal(
        result.vapor_pressure,
        np.broadcast_to(fixture.thermodynamic_parameters[:, 0], (2, 2)),
    )
    particle_inventory_change = np.sum(
        (result.particle_data.masses - fixture.masses)
        * fixture.concentration[:, :, None],
        axis=1,
    )
    drift = (
        particle_inventory_change
        + result.gas_data.concentration
        - fixture.gas_concentration
    )
    conservation_scale = np.maximum(
        np.abs(fixture.gas_concentration),
        np.abs(result.gas_data.concentration),
    )
    npt.assert_allclose(
        drift + conservation_scale,
        conservation_scale,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        result.energy_transfer,
        np.sum(result.total_mass_transfer, axis=1)
        * fixture.latent_heat[None, :],
        rtol=1e-12,
        atol=1e-18,
    )


def test_main_force_no_warp_prints_oracle_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Direct script execution preserves the CPU-only walkthrough route."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "oracle completed; no kernel ran" in output
    for category in ("physics", "conservation", "energy"):
        assert f"{category}: unavailable" in output
    assert "parity: passed" not in output


def test_main_returns_nonzero_for_failed_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed acceptance result produces a nonzero process status."""
    failed = example_module.AcceptanceResult("physics", "failed", "mismatch")
    monkeypatch.setattr(
        example_module,
        "run_example",
        lambda: types.SimpleNamespace(
            output=["physics: failed"], acceptance=(failed,)
        ),
    )

    assert example_module.main() == 1
    assert capsys.readouterr().out == "physics: failed\n"
