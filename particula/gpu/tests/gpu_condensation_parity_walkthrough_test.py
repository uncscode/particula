"""Regression tests for the independent condensation parity walkthrough."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
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
    assert result.output[-1] == "oracle completed; no kernel ran"
    assert result.particle_data is None
    assert result.oracle.total_mass_transfer.shape == (2, 2, 2)


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
    assert result.output[-1] == "oracle completed; no kernel ran"


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
        example_module, "to_warp_gas_data", lambda value, **k: value
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
    assert call["temperature"].shape == call["pressure"].shape == (2,)
    assert call["temperature"].dtype is _FakeWP.float64
    assert call["thermodynamics"].modes.dtype is _FakeWP.int32
    assert call["scratch_buffers"].total_mass_transfer.shape == (2, 2, 2)
    assert call["energy_transfer"].shape == (2, 2)
    assert len(_FakeWP.events) == 5
    npt.assert_allclose(result.raw_proposal, 2.0)
    npt.assert_allclose(result.total_mass_transfer, 3.0)
    npt.assert_allclose(result.energy_transfer, 4.0)


def test_oracle_completes_before_runtime_and_ignores_warp_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Detached Warp-source mutation cannot alter the completed oracle result."""
    events: list[str] = []
    original_oracle = example_module.run_oracle
    original_source = example_module.build_warp_source

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
        lambda: (_ for _ in ()).throw(RuntimeError("stop after oracle")),
    )
    with pytest.raises(RuntimeError, match="stop after oracle"):
        example_module.run_example()
    assert events == ["oracle"]


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


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_warp_cpu_matches_independent_oracle(
    example_module: types.ModuleType,
) -> None:
    """Warp CPU matches masses, gas, P2 total, raw proposal, and energy."""
    result = example_module.run_example(device="cpu")
    assert result.particle_data is not None and result.gas_data is not None
    assert (
        result.total_mass_transfer is not None
        and result.raw_proposal is not None
    )
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
        result.energy_transfer,
        result.oracle.energy_transfer,
        rtol=1e-10,
        atol=1e-30,
    )
    fixture = example_module.build_fixture()
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


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_matches_independent_oracle_when_available(
    example_module: types.ModuleType,
) -> None:
    """CUDA is optional additive evidence using the shared skip policy."""
    if not WARP_AVAILABLE:
        pytest.skip(CUDA_SKIP_REASON)
    wp = importlib.import_module("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    result = example_module.run_example(device="cuda")
    assert result.particle_data is not None
    npt.assert_allclose(
        result.particle_data.masses,
        result.oracle.masses,
        rtol=1e-10,
        atol=1e-30,
    )


def test_main_force_no_warp_prints_oracle_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Direct script execution preserves the CPU-only walkthrough route."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert "oracle completed; no kernel ran" in capsys.readouterr().out
