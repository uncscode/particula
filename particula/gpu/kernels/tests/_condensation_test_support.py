"""End-to-end tests for GPU condensation kernels."""

# mypy: ignore-errors
# ruff: noqa: F821, S101

from __future__ import annotations

import dataclasses
import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import numpy.testing as npt
import pytest

from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
    get_mass_transfer_rate_latent_heat,
)
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (
    get_molecule_mean_free_path,
)
from particula.gas.properties.pressure_function import (
    get_partial_pressure,
)
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.particles.particle_data import ParticleData
from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility,
)
from particula.particles.properties.diffusion_coefficient import (
    get_diffusion_coefficient,
)
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number,
)
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction,
)
from particula.util import constants

if TYPE_CHECKING:
    import warp as wp

    from particula.gas.environment_data import EnvironmentData
    from particula.gas.gas_data import GasData
    from particula.gpu.conversion import (
        from_warp_gas_data,
        from_warp_particle_data,
        to_warp_environment_data,
        to_warp_gas_data,
        to_warp_particle_data,
    )
    from particula.gpu.kernels.condensation import (
        CondensationActivitySurfaceConfig,
        ThermodynamicsConfig,
        _condensation_step_gpu,
        _validate_mass_transfer_buffer,
        _validate_species_array,
        condensation_step_gpu,
        particle_radius_from_volume_wp,
        validate_condensation_activity_surface_config,
        validate_condensation_scratch_buffers,
    )
    from particula.gpu.tests.cuda_availability import (
        cuda_available,
    )
    from particula.particles.particle_data import ParticleData

pytestmark = pytest.mark.warp

# Import-safe selector values are overwritten with the production Warp values
# by ``_load_warp_runtime`` before a selected test executes.
ACTIVITY_MODE_IDEAL = 0
ACTIVITY_MODE_KAPPA = 1
SURFACE_TENSION_MODE_STATIC = 0
SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED = 1
THERMODYNAMICS_MODE_CONSTANT = 0
THERMODYNAMICS_MODE_BUCK = 1


@lru_cache(maxsize=1)
def _load_warp_runtime() -> SimpleNamespace:
    """Import Warp-only dependencies when a selected test needs them."""
    runtime = SimpleNamespace()
    runtime.wp = pytest.importorskip("warp")
    runtime.condensation_module = importlib.import_module(
        "particula.gpu.kernels.condensation"
    )
    conversion = importlib.import_module("particula.gpu.conversion")
    thermodynamics = importlib.import_module(
        "particula.gpu.kernels.thermodynamics"
    )
    cuda = importlib.import_module("particula.gpu.tests.cuda_availability")
    runtime.__dict__.update(
        condensation_module=runtime.condensation_module,
        from_warp_gas_data=conversion.from_warp_gas_data,
        from_warp_particle_data=conversion.from_warp_particle_data,
        to_warp_environment_data=conversion.to_warp_environment_data,
        to_warp_gas_data=conversion.to_warp_gas_data,
        to_warp_particle_data=conversion.to_warp_particle_data,
        ACTIVITY_MODE_IDEAL=runtime.condensation_module.ACTIVITY_MODE_IDEAL,
        ACTIVITY_MODE_KAPPA=runtime.condensation_module.ACTIVITY_MODE_KAPPA,
        SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED=(
            runtime.condensation_module.SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED
        ),
        SURFACE_TENSION_MODE_STATIC=(
            runtime.condensation_module.SURFACE_TENSION_MODE_STATIC
        ),
        CondensationActivitySurfaceConfig=(
            runtime.condensation_module.CondensationActivitySurfaceConfig
        ),
        CondensationScratchBuffers=(
            runtime.condensation_module.CondensationScratchBuffers
        ),
        _validate_mass_transfer_buffer=(
            runtime.condensation_module._validate_mass_transfer_buffer
        ),
        _finalize_inventory_limited_mass_transfer=(
            runtime.condensation_module._finalize_inventory_limited_mass_transfer
        ),
        _validate_species_array=runtime.condensation_module._validate_species_array,
        validate_condensation_activity_surface_config=(
            runtime.condensation_module.validate_condensation_activity_surface_config
        ),
        validate_condensation_scratch_buffers=(
            runtime.condensation_module.validate_condensation_scratch_buffers
        ),
        _condensation_step_gpu=runtime.condensation_module.condensation_step_gpu,
        particle_radius_from_volume_wp=(
            runtime.condensation_module.particle_radius_from_volume_wp
        ),
        THERMODYNAMICS_MODE_BUCK=thermodynamics.THERMODYNAMICS_MODE_BUCK,
        THERMODYNAMICS_MODE_CONSTANT=thermodynamics.THERMODYNAMICS_MODE_CONSTANT,
        ThermodynamicsConfig=thermodynamics.ThermodynamicsConfig,
        cuda_available=cuda.cuda_available,
        CUDA_SKIP_REASON=cuda.CUDA_SKIP_REASON,
    )
    globals().update(runtime.__dict__)
    return runtime


_ENVIRONMENT_ARRAY_RTOL = 1.0e-7


def _make_thermodynamics_config(gpu_gas: Any) -> ThermodynamicsConfig:
    """Build a constant-model sidecar with nonzero vapor pressures in Pa."""
    n_species = gpu_gas.molar_mass.shape[0]
    return ThermodynamicsConfig(
        modes=wp.zeros(
            n_species, dtype=wp.int32, device=gpu_gas.molar_mass.device
        ),
        parameters=wp.array(
            np.column_stack(
                (
                    np.full(n_species, 800.0, dtype=np.float64),
                    np.zeros((n_species, 3), dtype=np.float64),
                )
            ),
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
        molar_mass_reference=wp.array(
            gpu_gas.molar_mass.numpy(),
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
    )


def _make_mixed_thermodynamics_config(gpu_gas: Any) -> ThermodynamicsConfig:
    """Build a two-mode constant/Buck sidecar for refresh integration tests."""
    n_species = gpu_gas.molar_mass.shape[0]
    modes = np.full(
        n_species, int(THERMODYNAMICS_MODE_CONSTANT), dtype=np.int32
    )
    if n_species > 1:
        modes[1] = int(THERMODYNAMICS_MODE_BUCK)
    parameters = np.zeros((n_species, 4), dtype=np.float64)
    parameters[:, 0] = np.linspace(725.0, 900.0, n_species)
    return ThermodynamicsConfig(
        modes=wp.array(modes, dtype=wp.int32, device=gpu_gas.molar_mass.device),
        parameters=wp.array(
            parameters,
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
        molar_mass_reference=wp.array(
            gpu_gas.molar_mass.numpy(),
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
    )


def _make_activity_surface_config(
    gpu_gas: Any,
    activity_mode: int,
    surface_mode: int,
) -> CondensationActivitySurfaceConfig:
    """Build a deterministic two-or-more-species activity sidecar."""
    n_species = gpu_gas.molar_mass.shape[0]
    return CondensationActivitySurfaceConfig(
        activity_mode=activity_mode,
        surface_tension_mode=surface_mode,
        water_species_index=0,
        kappas=wp.array(
            np.linspace(0.0, 0.5, n_species, dtype=np.float64),
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
        molar_mass_reference=wp.array(
            gpu_gas.molar_mass.numpy(),
            dtype=wp.float64,
            device=gpu_gas.molar_mass.device,
        ),
    )


def _make_condensation_scratch_buffers(
    shape: tuple[int, int, int],
    device: str,
    *,
    fields: tuple[str, ...] = (
        "work_mass_transfer",
        "total_mass_transfer",
        "dynamic_viscosity",
        "mean_free_path",
        "positive_mass_transfer_demand",
        "negative_mass_transfer_release",
        "positive_mass_transfer_scale",
    ),
) -> Any:
    """Build selected caller-owned fp64 scratch fields for a fixed shape."""
    n_boxes, _, n_species = shape
    values: dict[str, Any] = {
        "work_mass_transfer": None,
        "total_mass_transfer": None,
        "dynamic_viscosity": None,
        "mean_free_path": None,
        "positive_mass_transfer_demand": None,
        "negative_mass_transfer_release": None,
        "positive_mass_transfer_scale": None,
    }
    for name in fields:
        field_shape = (
            shape
            if name in {"work_mass_transfer", "total_mass_transfer"}
            else (n_boxes, n_species)
            if "mass_transfer" in name
            else (n_boxes,)
        )
        values[name] = wp.full(
            field_shape, wp.float64(-1.0), dtype=wp.float64, device=device
        )
    return CondensationScratchBuffers(**values)


def _inventory_reference(
    masses: np.ndarray,
    concentration: np.ndarray,
    gas_concentration: np.ndarray,
    proposal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the independent fixed-order P2 inventory finalization oracle."""
    candidate = np.maximum(proposal, -masses)
    demand = np.zeros(gas_concentration.shape, dtype=np.float64)
    release = np.zeros(gas_concentration.shape, dtype=np.float64)
    n_boxes, n_particles, n_species = candidate.shape
    for box_idx in range(n_boxes):
        for species_idx in range(n_species):
            for particle_idx in range(n_particles):
                transfer = candidate[box_idx, particle_idx, species_idx]
                weighted = transfer * concentration[box_idx, particle_idx]
                if transfer > 0.0:
                    demand[box_idx, species_idx] += weighted
                elif transfer < 0.0:
                    release[box_idx, species_idx] -= weighted
    available = gas_concentration + release
    scale = np.ones(gas_concentration.shape, dtype=np.float64)
    limited = demand > available
    scale[limited] = np.clip(available[limited] / demand[limited], 0.0, 1.0)
    finalized = candidate.copy()
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            for species_idx in range(n_species):
                if finalized[box_idx, particle_idx, species_idx] > 0.0:
                    finalized[box_idx, particle_idx, species_idx] *= scale[
                        box_idx, species_idx
                    ]
    return candidate, demand, release, scale, finalized


@pytest.mark.parametrize(
    ("masses", "concentration", "gas_concentration", "proposal"),
    (
        (
            np.array([[[2.0, 3.0], [4.0, 5.0]]], dtype=np.float64),
            np.array([[2.0, 1.0]], dtype=np.float64),
            np.array([[20.0, 20.0]], dtype=np.float64),
            np.array([[[1.0, -1.0], [2.0, 1.0]]], dtype=np.float64),
        ),
        (
            np.array([[[2.0], [4.0]]], dtype=np.float64),
            np.array([[1.0, 3.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            np.array([[[2.0], [2.0]]], dtype=np.float64),
        ),
        (
            np.array([[[1.0], [4.0]]], dtype=np.float64),
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
            np.array([[[-3.0], [2.0]]], dtype=np.float64),
        ),
        (
            np.array(
                [
                    [[2.0, 3.0], [5.0, 7.0]],
                    [[11.0, 13.0], [17.0, 19.0]],
                ],
                dtype=np.float64,
            ),
            np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float64),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array(
                [
                    # The second particle is inactive and the first species in
                    # its proposal is explicitly pre-gated to zero.
                    [[4.0, -4.0], [0.0, 0.0]],
                    [[-20.0, 3.0], [6.0, -30.0]],
                ],
                dtype=np.float64,
            ),
        ),
    ),
)
def test_finalize_inventory_limited_transfer_matches_numpy_oracle(
    warp_cpu_device: str,
    masses: np.ndarray,
    concentration: np.ndarray,
    gas_concentration: np.ndarray,
    proposal: np.ndarray,
) -> None:
    """The direct P2 helper matches fixed-order fp64 inventory accounting."""
    n_boxes, n_particles, n_species = masses.shape
    particles = _make_particle_data(n_boxes, n_particles, n_species)
    particles = dataclasses.replace(
        particles,
        masses=masses.copy(),
        concentration=concentration.copy(),
    )
    gas = _make_gas_data(n_boxes, n_species)
    gas = dataclasses.replace(gas, concentration=gas_concentration.copy())
    expected = _inventory_reference(
        masses, concentration, gas_concentration, proposal
    )
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros_like(gas_concentration),
    )
    proposal_buffer = wp.array(
        proposal, dtype=wp.float64, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        masses.shape,
        warp_cpu_device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    supplied_demand = scratch.positive_mass_transfer_demand
    supplied_release = scratch.negative_mass_transfer_release
    supplied_scale = scratch.positive_mass_transfer_scale
    initial_gas = gpu_gas.concentration.numpy().copy()
    finalized = _finalize_inventory_limited_mass_transfer(
        gpu_particles,
        gpu_gas,
        proposal_buffer,
        scratch,
    )
    _, demand, release, scale, expected_finalized = expected
    npt.assert_allclose(
        finalized.numpy(), expected_finalized, rtol=1.0e-12, atol=1.0e-14
    )
    npt.assert_allclose(
        gpu_particles.masses.numpy(),
        masses + expected_finalized,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    npt.assert_allclose(
        scratch.positive_mass_transfer_demand.numpy(), demand, rtol=1.0e-12
    )
    npt.assert_allclose(
        scratch.negative_mass_transfer_release.numpy(), release, rtol=1.0e-12
    )
    npt.assert_allclose(
        scratch.positive_mass_transfer_scale.numpy(), scale, rtol=1.0e-12
    )
    assert scratch.positive_mass_transfer_demand is not None
    assert scratch.negative_mass_transfer_release is not None
    assert scratch.positive_mass_transfer_scale is not None
    assert scratch.positive_mass_transfer_demand is supplied_demand
    assert scratch.negative_mass_transfer_release is supplied_release
    assert scratch.positive_mass_transfer_scale is supplied_scale
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    assert np.all(np.isfinite(finalized.numpy()))
    assert np.all(gpu_particles.masses.numpy() >= 0.0)
    assert np.all((scale >= 0.0) & (scale <= 1.0))
    assert np.all(initial_gas + release - demand * scale >= -1.0e-14)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    (
        ("shape", "mass_transfer shape"),
        ("dtype", "mass_transfer must use dtype"),
        ("scratch_shape", "positive_mass_transfer_demand"),
        ("scratch_dtype", "negative_mass_transfer_release"),
        ("scratch_object", "positive_mass_transfer_scale"),
    ),
)
def test_finalize_inventory_limited_transfer_preflight_is_atomic(
    warp_cpu_device: str, invalid_kind: str, message: str
) -> None:
    """Malformed P2 direct-helper inputs leave every caller buffer unchanged."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros((1, 2), dtype=np.float64),
    )
    proposal = wp.full(
        (1, 2, 2), wp.float64(1.0), dtype=wp.float64, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        (1, 2, 2),
        warp_cpu_device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    if invalid_kind == "shape":
        proposal = wp.zeros((1, 1, 2), dtype=wp.float64, device=warp_cpu_device)
    elif invalid_kind == "dtype":
        proposal = wp.zeros((1, 2, 2), dtype=wp.float32, device=warp_cpu_device)
    elif invalid_kind == "scratch_shape":
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_demand=wp.zeros(
                (1, 3), dtype=wp.float64, device=warp_cpu_device
            ),
        )
    elif invalid_kind == "scratch_dtype":
        scratch = dataclasses.replace(
            scratch,
            negative_mass_transfer_release=wp.zeros(
                (1, 2), dtype=wp.float32, device=warp_cpu_device
            ),
        )
    else:
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_scale=np.zeros((1, 2), dtype=np.float64),
        )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_proposal = proposal.numpy().copy()
    initial_scratch = {
        name: getattr(scratch, name).numpy().copy()
        if hasattr(getattr(scratch, name), "numpy")
        else getattr(scratch, name).copy()
        for name in (
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        )
    }
    with pytest.raises(ValueError, match=message):
        _finalize_inventory_limited_mass_transfer(
            gpu_particles, gpu_gas, proposal, scratch
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(proposal.numpy(), initial_proposal)
    for name, values in initial_scratch.items():
        actual = getattr(scratch, name)
        actual = actual.numpy() if hasattr(actual, "numpy") else actual
        npt.assert_array_equal(actual, values)


@pytest.mark.parametrize(
    ("invalid_kind", "invalid_value", "message"),
    (
        ("masses", -1.0, "particles.masses.*finite"),
        ("concentration", np.nan, "particles.concentration.*finite"),
        ("gas", -1.0, "gas.concentration.*finite"),
        ("proposal", np.inf, "gated_mass_transfer must be finite"),
    ),
)
def test_finalize_inventory_rejects_invalid_physical_inputs_atomically(
    warp_cpu_device: str,
    invalid_kind: str,
    invalid_value: float,
    message: str,
) -> None:
    """Invalid direct-P2 physical inputs fail before caller-state mutation."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros((1, 1), dtype=np.float64),
    )
    proposal = wp.full(
        (1, 1, 1), wp.float64(1.0), dtype=wp.float64, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        (1, 1, 1),
        warp_cpu_device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    if invalid_kind == "masses":
        wp.copy(
            gpu_particles.masses,
            wp.array(
                [[[invalid_value]]], dtype=wp.float64, device=warp_cpu_device
            ),
        )
    elif invalid_kind == "concentration":
        wp.copy(
            gpu_particles.concentration,
            wp.array(
                [[invalid_value]], dtype=wp.float64, device=warp_cpu_device
            ),
        )
    elif invalid_kind == "gas":
        wp.copy(
            gpu_gas.concentration,
            wp.array(
                [[invalid_value]], dtype=wp.float64, device=warp_cpu_device
            ),
        )
    else:
        proposal = wp.array(
            [[[invalid_value]]], dtype=wp.float64, device=warp_cpu_device
        )
    initial = (
        gpu_particles.masses.numpy().copy(),
        gpu_gas.concentration.numpy().copy(),
        proposal.numpy().copy(),
        *(
            getattr(scratch, name).numpy().copy()
            for name in (
                "positive_mass_transfer_demand",
                "negative_mass_transfer_release",
                "positive_mass_transfer_scale",
            )
        ),
    )
    with pytest.raises(ValueError, match=message):
        _finalize_inventory_limited_mass_transfer(
            gpu_particles, gpu_gas, proposal, scratch
        )
    actual = (
        gpu_particles.masses.numpy(),
        gpu_gas.concentration.numpy(),
        proposal.numpy(),
        *(
            getattr(scratch, name).numpy()
            for name in (
                "positive_mass_transfer_demand",
                "negative_mass_transfer_release",
                "positive_mass_transfer_scale",
            )
        ),
    )
    for actual_values, initial_values in zip(actual, initial, strict=True):
        npt.assert_array_equal(actual_values, initial_values)


@pytest.mark.parametrize("alias_kind", ("peer", "gas", "energy"))
def test_finalize_inventory_rejects_p2_sidecar_aliases_atomically(
    warp_cpu_device: str, alias_kind: str
) -> None:
    """P2 reduction sidecars cannot alias peers or protected input/output."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros((1, 1), dtype=np.float64),
    )
    proposal = wp.full(
        (1, 1, 1), wp.float64(1.0), dtype=wp.float64, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        (1, 1, 1),
        warp_cpu_device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    energy_transfer = None
    if alias_kind == "peer":
        scratch = dataclasses.replace(
            scratch,
            negative_mass_transfer_release=scratch.positive_mass_transfer_demand,
        )
    elif alias_kind == "gas":
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_demand=gpu_gas.concentration,
        )
    else:
        energy_transfer = wp.full(
            (1, 1), wp.float64(-2.0), dtype=wp.float64, device=warp_cpu_device
        )
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_demand=energy_transfer,
        )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_proposal = proposal.numpy().copy()
    initial_energy = (
        None if energy_transfer is None else energy_transfer.numpy().copy()
    )
    with pytest.raises(ValueError, match="must not overlap"):
        _finalize_inventory_limited_mass_transfer(
            gpu_particles,
            gpu_gas,
            proposal,
            scratch,
            energy_transfer=energy_transfer,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(proposal.numpy(), initial_proposal)
    if energy_transfer is not None:
        assert initial_energy is not None
        npt.assert_array_equal(energy_transfer.numpy(), initial_energy)


@pytest.mark.cuda
def test_finalize_inventory_limited_transfer_rejects_cuda_p2_sidecar(
    warp_cpu_device: str,
    cuda_device: str,
) -> None:
    """A P2 sidecar on another device fails before mutating caller state."""
    particles = _make_particle_data(1, 2, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros((1, 1), dtype=np.float64),
    )
    proposal = wp.full(
        (1, 2, 1), wp.float64(1.0), dtype=wp.float64, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        (1, 2, 1),
        warp_cpu_device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    scratch = dataclasses.replace(
        scratch,
        positive_mass_transfer_demand=wp.zeros(
            (1, 1), dtype=wp.float64, device=cuda_device
        ),
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_proposal = proposal.numpy().copy()
    initial_sidecars = {
        name: getattr(scratch, name).numpy().copy()
        for name in (
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        )
    }

    with pytest.raises(ValueError, match="positive_mass_transfer_demand"):
        _finalize_inventory_limited_mass_transfer(
            gpu_particles, gpu_gas, proposal, scratch
        )

    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(proposal.numpy(), initial_proposal)
    for name, values in initial_sidecars.items():
        actual = getattr(scratch, name)
        assert actual is not None
        npt.assert_array_equal(actual.numpy(), values)


def test_finalize_inventory_limited_transfer_is_exactly_repeatable(
    warp_cpu_device: str,
) -> None:
    """Fresh identical P2 inputs produce bitwise-identical finalized state."""
    masses = np.array([[[2.0], [4.0]]], dtype=np.float64)
    concentration = np.array([[1.0, 3.0]], dtype=np.float64)
    gas_concentration = np.array([[2.0]], dtype=np.float64)
    proposal = np.array([[[2.0], [-6.0]]], dtype=np.float64)

    def _run_once() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        particles = dataclasses.replace(
            _make_particle_data(1, 2, 1),
            masses=masses.copy(),
            concentration=concentration.copy(),
        )
        gas = dataclasses.replace(
            _make_gas_data(1, 1), concentration=gas_concentration.copy()
        )
        gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
        gpu_gas = to_warp_gas_data(
            gas,
            device=warp_cpu_device,
            vapor_pressure=np.zeros_like(gas_concentration),
        )
        scratch = _make_condensation_scratch_buffers(
            masses.shape,
            warp_cpu_device,
            fields=(
                "positive_mass_transfer_demand",
                "negative_mass_transfer_release",
                "positive_mass_transfer_scale",
            ),
        )
        finalized = _finalize_inventory_limited_mass_transfer(
            gpu_particles,
            gpu_gas,
            wp.array(proposal, dtype=wp.float64, device=warp_cpu_device),
            scratch,
        )
        assert scratch.positive_mass_transfer_demand is not None
        assert scratch.negative_mass_transfer_release is not None
        assert scratch.positive_mass_transfer_scale is not None
        return (
            finalized.numpy(),
            gpu_particles.masses.numpy(),
            scratch.positive_mass_transfer_demand.numpy(),
            scratch.negative_mass_transfer_release.numpy(),
            scratch.positive_mass_transfer_scale.numpy(),
        )

    first = _run_once()
    second = _run_once()
    for first_values, second_values in zip(first, second, strict=True):
        npt.assert_array_equal(first_values, second_values)


def test_finalize_inventory_limited_transfer_allocates_omitted_sidecars(
    warp_cpu_device: str,
) -> None:
    """The direct P2 helper uses private fp64 fallback reduction storage."""
    masses = np.array([[[1.0], [2.0]]], dtype=np.float64)
    concentration = np.array([[1.0, 2.0]], dtype=np.float64)
    gas_concentration = np.array([[1.0]], dtype=np.float64)
    proposal = np.array([[[2.0], [-3.0]]], dtype=np.float64)
    particles = dataclasses.replace(
        _make_particle_data(1, 2, 1),
        masses=masses.copy(),
        concentration=concentration.copy(),
    )
    gas = dataclasses.replace(
        _make_gas_data(1, 1), concentration=gas_concentration.copy()
    )
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=np.zeros_like(gas_concentration),
    )

    finalized = _finalize_inventory_limited_mass_transfer(
        gpu_particles,
        gpu_gas,
        wp.array(proposal, dtype=wp.float64, device=warp_cpu_device),
    )

    _, _, _, _, expected = _inventory_reference(
        masses, concentration, gas_concentration, proposal
    )
    npt.assert_allclose(finalized.numpy(), expected, rtol=1.0e-12)
    npt.assert_allclose(
        gpu_particles.masses.numpy(), masses + expected, rtol=1.0e-12
    )


@pytest.mark.parametrize(
    ("state_name", "invalid_value"),
    (
        ("masses", np.nan),
        ("masses", np.inf),
        ("masses", -1.0),
        ("concentration", np.nan),
        ("concentration", np.inf),
        ("concentration", -1.0),
        ("gas", np.nan),
        ("gas", np.inf),
        ("gas", -1.0),
    ),
)
def test_condensation_step_gpu_invalid_primary_p2_state_is_atomic(
    device: str,
    state_name: str,
    invalid_value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid P2 physical state fails before public outputs are changed."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    latent_heat = wp.array([2.2e6], dtype=wp.float64, device=device)
    energy_transfer = wp.full(
        (1, 1), wp.float64(13.0), dtype=wp.float64, device=device
    )
    if state_name == "masses":
        wp.copy(
            gpu_particles.masses,
            wp.array([[[invalid_value]]], dtype=wp.float64, device=device),
        )
    elif state_name == "concentration":
        wp.copy(
            gpu_particles.concentration,
            wp.array([[invalid_value]], dtype=wp.float64, device=device),
        )
    else:
        wp.copy(
            gpu_gas.concentration,
            wp.array([[invalid_value]], dtype=wp.float64, device=device),
        )
    state_arrays = (
        gpu_particles.masses,
        gpu_particles.concentration,
        gpu_gas.concentration,
        gpu_gas.vapor_pressure,
        scratch.work_mass_transfer,
        scratch.total_mass_transfer,
        scratch.dynamic_viscosity,
        scratch.mean_free_path,
        scratch.positive_mass_transfer_demand,
        scratch.negative_mass_transfer_release,
        scratch.positive_mass_transfer_scale,
        energy_transfer,
    )
    assert all(array is not None for array in state_arrays)
    snapshots = tuple(array.numpy().copy() for array in state_arrays)
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
            latent_heat=latent_heat,
            energy_transfer=energy_transfer,
        )
    for array, snapshot in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), snapshot)
    assert (
        condensation_module._clear_mass_transfer_kernel not in launched_kernels
    )
    assert (
        condensation_module._clear_energy_transfer_kernel
        not in launched_kernels
    )


def test_condensation_step_gpu_empty_particle_axis_preserves_gas_and_outputs(
    device: str,
) -> None:
    """An empty particle axis completes four P2 cycles without gas loss."""
    particles = _make_particle_data(2, 0, 2)
    gas = _make_gas_data(2, 2)
    initial_gas = gas.concentration.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(2, 2)
    )
    total = wp.full(
        particles.masses.shape,
        wp.float64(9.0),
        dtype=wp.float64,
        device=device,
    )
    _, returned = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        mass_transfer=total,
    )

    assert returned is total
    assert returned.shape == (2, 0, 2)
    npt.assert_array_equal(gpu_particles.masses.numpy(), particles.masses)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(returned.numpy(), np.zeros_like(particles.masses))


def test_condensation_step_gpu_couples_four_substeps_to_numpy_oracle(
    warp_cpu_device: str,
) -> None:
    """Four P2-finalized substeps agree with the coupled NumPy oracle."""
    particles = _make_particle_data(2, 2, 2)
    gas = _make_gas_data(2, 2)
    vapor_pressure = np.full((2, 2), 800.0, dtype=np.float64)
    surface_tension = np.full(2, 0.072, dtype=np.float64)
    mass_accommodation = np.ones(2, dtype=np.float64)
    diffusion = np.full(2, 2.0e-5, dtype=np.float64)
    expected_mass, expected_total, _, expected_gas = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        298.15,
        101325.0,
        0.1,
        return_gas=True,
    )

    final_particles, returned, final_gas = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        298.15,
        101325.0,
        0.1,
        warp_cpu_device,
        return_gas_state=True,
    )

    npt.assert_allclose(
        final_particles.masses, expected_mass, rtol=1e-12, atol=1e-25
    )
    npt.assert_allclose(
        returned.numpy(), expected_total, rtol=1e-12, atol=1e-25
    )
    npt.assert_allclose(
        final_gas.concentration, expected_gas, rtol=1e-12, atol=1e-22
    )
    npt.assert_allclose(
        gas.concentration - final_gas.concentration,
        np.sum(returned.numpy() * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        atol=1e-22,
    )
    assert np.all(np.isfinite(final_gas.concentration))
    assert np.all(final_gas.concentration >= 0.0)


def test_condensation_step_gpu_rejects_p2_vapor_pressure_alias_atomically(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 sidecars cannot overwrite vapor pressure before refresh launches."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    scratch = dataclasses.replace(
        scratch,
        positive_mass_transfer_demand=gpu_gas.vapor_pressure,
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_vapor_pressure = gpu_gas.vapor_pressure.numpy().copy()
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    with pytest.raises(ValueError, match="must not overlap"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(
        gpu_gas.vapor_pressure.numpy(), initial_vapor_pressure
    )
    assert (
        condensation_module._clear_mass_transfer_kernel not in launched_kernels
    )


def test_condensation_step_gpu_rejects_legacy_output_alias_atomically(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy mass-transfer output cannot clear particle-owned mass storage."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    with pytest.raises(ValueError, match="mass_transfer must not overlap"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            mass_transfer=gpu_particles.masses,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    assert (
        condensation_module._clear_mass_transfer_kernel not in launched_kernels
    )


@pytest.mark.parametrize(
    "alias_kind",
    (
        "work_total",
        "work_particles",
        "total_particles",
        "dynamic_temperature",
        "mean_pressure",
        "demand_parameters",
        "release_parameters",
        "scale_parameters",
    ),
)
def test_condensation_scratch_ownership_aliases_are_atomic(
    device: str,
    alias_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every supplied scratch role rejects aliased condensation state.

    This check remains atomic.
    """
    particles = _make_particle_data(4, 1, 4)
    gas = _make_gas_data(4, 4)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(4, 4)
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    scratch = _make_condensation_scratch_buffers((4, 1, 4), device)
    temperature = wp.full(4, 298.15, dtype=wp.float64, device=device)
    pressure = wp.full(4, 101325.0, dtype=wp.float64, device=device)
    if alias_kind == "work_total":
        scratch = dataclasses.replace(
            scratch, total_mass_transfer=scratch.work_mass_transfer
        )
    elif alias_kind == "work_particles":
        scratch = dataclasses.replace(
            scratch, work_mass_transfer=gpu_particles.masses
        )
    elif alias_kind == "total_particles":
        scratch = dataclasses.replace(
            scratch, total_mass_transfer=gpu_particles.masses
        )
    elif alias_kind == "dynamic_temperature":
        scratch = dataclasses.replace(scratch, dynamic_viscosity=temperature)
    elif alias_kind == "mean_pressure":
        scratch = dataclasses.replace(scratch, mean_free_path=pressure)
    elif alias_kind == "demand_parameters":
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_demand=thermodynamics.parameters,
        )
    elif alias_kind == "release_parameters":
        scratch = dataclasses.replace(
            scratch,
            negative_mass_transfer_release=thermodynamics.parameters,
        )
    else:
        scratch = dataclasses.replace(
            scratch,
            positive_mass_transfer_scale=thermodynamics.parameters,
        )
    state_arrays = (
        gpu_particles.masses,
        gpu_particles.concentration,
        gpu_gas.concentration,
        gpu_gas.vapor_pressure,
        thermodynamics.parameters,
        temperature,
        pressure,
        *(
            getattr(scratch, name)
            for name in (
                "work_mass_transfer",
                "total_mass_transfer",
                "dynamic_viscosity",
                "mean_free_path",
                "positive_mass_transfer_demand",
                "negative_mass_transfer_release",
                "positive_mass_transfer_scale",
            )
        ),
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _track_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _track_launch)
    with pytest.raises(ValueError, match="must not overlap"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            thermodynamics=thermodynamics,
            scratch_buffers=scratch,
        )
    for array, snapshot in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), snapshot)
    assert (
        condensation_module._clear_mass_transfer_kernel not in launched_kernels
    )


def test_condensation_step_gpu_stale_nonfinite_work_buffer_is_overwritten(
    device: str,
) -> None:
    """Fresh P1 output replaces stale non-finite caller work storage."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    assert scratch.work_mass_transfer is not None
    assert scratch.total_mass_transfer is not None
    wp.copy(
        scratch.work_mass_transfer,
        wp.array([[[np.nan]]], dtype=wp.float64, device=device),
    )

    _, returned = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        scratch_buffers=scratch,
    )

    assert returned is scratch.total_mass_transfer
    assert np.all(np.isfinite(scratch.work_mass_transfer.numpy()))
    assert np.all(np.isfinite(returned.numpy()))


def test_condensation_step_gpu_nonfinite_fresh_proposal_is_p2_atomic(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite fresh P1 proposal cannot mutate P2-owned state."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.concentration,
        scratch.positive_mass_transfer_demand,
        scratch.negative_mass_transfer_release,
        scratch.positive_mass_transfer_scale,
    )
    assert all(array is not None for array in state_arrays)
    snapshots = tuple(array.numpy().copy() for array in state_arrays)
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _inject_nonfinite_proposal(
        kernel: Any, *args: Any, **kwargs: Any
    ) -> Any:
        launched_kernels.append(kernel)
        result = original_launch(kernel, *args, **kwargs)
        if kernel is condensation_module._gate_mass_transfer_kernel:
            work = kwargs["inputs"][2]
            wp.copy(
                work,
                wp.array([[[np.inf]]], dtype=wp.float64, device=device),
            )
        return result

    monkeypatch.setattr(
        condensation_module.wp, "launch", _inject_nonfinite_proposal
    )
    with pytest.raises(ValueError, match="gated_mass_transfer must be finite"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )

    for array, snapshot in zip(state_arrays, snapshots, strict=True):
        assert array is not None
        npt.assert_array_equal(array.numpy(), snapshot)
    assert scratch.work_mass_transfer is not None
    assert scratch.total_mass_transfer is not None
    assert np.isinf(scratch.work_mass_transfer.numpy()).all()
    npt.assert_array_equal(scratch.total_mass_transfer.numpy(), 0.0)
    assert (
        condensation_module._bound_evaporation_candidate_kernel
        not in launched_kernels
    )
    assert (
        condensation_module._couple_finalized_transfer_to_gas_kernel
        not in launched_kernels
    )


def test_condensation_energy_retains_committed_substeps_after_proposal_failure(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Energy retains the first committed substep.

    This applies when the next proposal fails.
    """
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    latent_heat = wp.array([2.2e6], dtype=wp.float64, device=device)
    energy_transfer = wp.full((1, 1), 13.0, dtype=wp.float64, device=device)
    original_launch = condensation_module.wp.launch
    gate_launches = 0

    def _fail_second_proposal(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal gate_launches
        result = original_launch(kernel, *args, **kwargs)
        if kernel is condensation_module._gate_mass_transfer_kernel:
            gate_launches += 1
            if gate_launches == 2:
                wp.copy(
                    kwargs["inputs"][2],
                    wp.array([[[np.inf]]], dtype=wp.float64, device=device),
                )
        return result

    monkeypatch.setattr(condensation_module.wp, "launch", _fail_second_proposal)
    with pytest.raises(ValueError, match="gated_mass_transfer must be finite"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
            latent_heat=latent_heat,
            energy_transfer=energy_transfer,
        )
    assert gate_launches == 2
    assert scratch.total_mass_transfer is not None
    assert scratch.work_mass_transfer is not None
    npt.assert_allclose(
        energy_transfer.numpy(),
        np.sum(scratch.total_mass_transfer.numpy(), axis=1)
        * latent_heat.numpy()[None, :],
        rtol=1e-12,
        atol=1e-24,
    )
    assert np.any(energy_transfer.numpy() != 0.0)
    assert np.isinf(scratch.work_mass_transfer.numpy()).all()


def test_condensation_rejects_nonfinite_extreme_gas_delta_before_coupling(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite finalized evaporation overflow cannot make gas non-finite."""
    particles = dataclasses.replace(
        _make_particle_data(1, 1, 1),
        masses=np.array([[[1.0e100]]]),
        concentration=np.array([[1.0e308]]),
    )
    gas = dataclasses.replace(
        _make_gas_data(1, 1), concentration=np.array([[1.0]])
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    initial_gas = gpu_gas.concentration.numpy().copy()
    original_launch = condensation_module.wp.launch

    def _inject_extreme_evaporation(
        kernel: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = original_launch(kernel, *args, **kwargs)
        if kernel is condensation_module._gate_mass_transfer_kernel:
            wp.copy(
                kwargs["inputs"][2],
                wp.array([[[-1.0e100]]], dtype=wp.float64, device=device),
            )
        return result

    monkeypatch.setattr(
        condensation_module.wp, "launch", _inject_extreme_evaporation
    )
    with pytest.raises(ValueError, match="gas coupling must be finite"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
        )
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    assert np.all(np.isfinite(gpu_gas.concentration.numpy()))


def test_condensation_public_insufficient_inventory_scales_uptake_and_conserves(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public P2 coupling limits uptake while conserving weighted inventory."""
    particles = dataclasses.replace(
        _make_particle_data(1, 1, 1),
        masses=np.array([[[1.0e-18]]]),
        concentration=np.array([[1.0e9]]),
    )
    gas = dataclasses.replace(
        _make_gas_data(1, 1), concentration=np.array([[1.0e-12]])
    )
    initial_weighted_total = (
        particles.masses * particles.concentration[:, :, None]
    ).sum(axis=1) + gas.concentration
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers((1, 1, 1), device)
    original_launch = condensation_module.wp.launch

    def _inject_insufficient_uptake(
        kernel: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = original_launch(kernel, *args, **kwargs)
        if kernel is condensation_module._gate_mass_transfer_kernel:
            wp.copy(
                kwargs["inputs"][2],
                wp.array([[[1.0e-18]]], dtype=wp.float64, device=device),
            )
        return result

    monkeypatch.setattr(
        condensation_module.wp, "launch", _inject_insufficient_uptake
    )
    _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        scratch_buffers=scratch,
    )
    assert scratch.positive_mass_transfer_scale is not None
    assert np.all(scratch.positive_mass_transfer_scale.numpy() < 1.0)
    final_gas = gpu_gas.concentration.numpy()
    final_weighted_total = (
        gpu_particles.masses.numpy() * particles.concentration[:, :, None]
    ).sum(axis=1) + final_gas
    assert np.all(final_gas >= 0.0)
    npt.assert_allclose(
        final_weighted_total, initial_weighted_total, rtol=1e-12
    )


def condensation_step_gpu(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Call the public step with the standard valid test sidecar by default."""
    if "thermodynamics" not in kwargs:
        kwargs["thermodynamics"] = _make_thermodynamics_config(args[1])
    return _condensation_step_gpu(*args, **kwargs)


def _evaluate_vapor_pressure_config(
    thermodynamics: ThermodynamicsConfig,
    temperature: float | np.ndarray,
) -> np.ndarray:
    """Evaluate a test thermodynamic sidecar using the production equations."""
    temperatures = np.atleast_1d(np.asarray(temperature, dtype=np.float64))
    modes = np.asarray(thermodynamics.modes.numpy(), dtype=np.int32)
    parameters = np.asarray(thermodynamics.parameters.numpy(), dtype=np.float64)
    vapor_pressure = np.empty((temperatures.size, modes.size), dtype=np.float64)
    for species_idx, mode in enumerate(modes):
        if mode == int(THERMODYNAMICS_MODE_CONSTANT):
            vapor_pressure[:, species_idx] = parameters[species_idx, 0]
            continue
        if mode != int(THERMODYNAMICS_MODE_BUCK):
            raise ValueError("unsupported test thermodynamics mode")
        temperature_celsius = temperatures - 273.15
        ice = temperature_celsius < 0.0
        vapor_pressure[ice, species_idx] = (
            6.1115
            * np.exp(
                (23.036 - temperature_celsius[ice] / 333.7)
                * temperature_celsius[ice]
                / (279.82 + temperature_celsius[ice])
            )
            * 100.0
        )
        vapor_pressure[~ice, species_idx] = (
            6.1121
            * np.exp(
                (18.678 - temperature_celsius[~ice] / 234.5)
                * temperature_celsius[~ice]
                / (257.14 + temperature_celsius[~ice])
            )
            * 100.0
        )
    return vapor_pressure


@pytest.fixture
def warp_cpu_device() -> str:
    """Provide the required Warp CPU parity backend lazily."""
    _load_warp_runtime()
    return "cpu"


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime() -> None:
    """Load Warp only while executing a selected support-backed test."""
    _load_warp_runtime()


@pytest.fixture
def cuda_device() -> str:
    """Provide CUDA only for separately selected, availability-guarded tests."""
    runtime = _load_warp_runtime()
    if not runtime.cuda_available(runtime.wp):
        pytest.skip(runtime.CUDA_SKIP_REASON)
    return "cuda"


@pytest.fixture
def device(warp_cpu_device: str) -> str:
    """Retain the legacy fixture name while keeping CPU as the baseline."""
    return warp_cpu_device


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> ParticleData:
    """Create deterministic particle data for GPU tests."""
    base_masses = np.linspace(1.0e-18, 3.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.1 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.ones((n_boxes, n_particles), dtype=np.float64)
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create deterministic gas data for GPU tests."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        concentration[box_idx, :] = 1.0e-6 * (1.0 + 0.2 * box_idx)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_vapor_pressure(n_boxes: int, n_species: int) -> np.ndarray:
    """Create deterministic vapor pressure array."""
    vapor_pressure = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        vapor_pressure[box_idx, :] = 800.0 + 50.0 * box_idx
    return vapor_pressure


def _make_environment_data(
    n_boxes: int,
    n_species: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> EnvironmentData:
    """Create deterministic environment data for contract tests."""
    return EnvironmentData(
        temperature=np.full((n_boxes,), temperature, dtype=np.float64),
        pressure=np.full((n_boxes,), pressure, dtype=np.float64),
        saturation_ratio=np.ones((n_boxes, n_species), dtype=np.float64),
    )


@dataclass(frozen=True)
class CondensationStiffnessCase:
    """Define a deterministic fixed-shape condensation stress case.

    Attributes:
        name: Recorded case name used by the stiffness sweep helpers.
        n_boxes: Number of spatial boxes in the deterministic case.
        n_particles: Number of particles per box.
        n_species: Number of condensable species.
        time_step: Baseline timestep used by legacy single-step checks.
        temperature: Baseline scalar temperature in K.
        pressure: Baseline scalar pressure in Pa.
        particle_mass_scale: Multiplier applied to the seed particle masses.
        gas_concentration_scale: Multiplier applied to the seed gas field.
        vapor_pressure_scale: Multiplier applied to the seed vapor pressure.
        box_temperature_step: Per-box temperature increment for multi-box cases.
        box_pressure_step: Per-box pressure increment for multi-box cases.
        zero_mass_particles: Particle indices forced to zero initial mass.
        zero_concentration_particles: Particle indices forced inactive.
    """

    name: str
    n_boxes: int
    n_particles: int
    n_species: int
    time_step: float
    temperature: float = 298.15
    pressure: float = 101325.0
    particle_mass_scale: float = 1.0
    gas_concentration_scale: float = 1.0
    vapor_pressure_scale: float = 1.0
    box_temperature_step: float = 0.0
    box_pressure_step: float = 0.0
    zero_mass_particles: tuple[tuple[int, int], ...] = ()
    zero_concentration_particles: tuple[tuple[int, int], ...] = ()

    def build_particle_data(self) -> ParticleData:
        """Build deterministic particle data for the configured case."""
        particles = _make_particle_data(
            n_boxes=self.n_boxes,
            n_particles=self.n_particles,
            n_species=self.n_species,
        )
        particles.masses *= self.particle_mass_scale
        for box_idx, particle_idx in self.zero_mass_particles:
            particles.masses[box_idx, particle_idx, :] = 0.0
        for box_idx, particle_idx in self.zero_concentration_particles:
            particles.concentration[box_idx, particle_idx] = 0.0
        return particles

    def build_gas_data(self) -> GasData:
        """Build deterministic gas data for the configured case."""
        gas = _make_gas_data(self.n_boxes, self.n_species)
        gas.concentration *= self.gas_concentration_scale
        return gas

    def build_vapor_pressure(self) -> np.ndarray:
        """Build deterministic vapor pressure data for the configured case."""
        vapor_pressure = _make_vapor_pressure(self.n_boxes, self.n_species)
        return vapor_pressure * self.vapor_pressure_scale

    def build_environment_data(self) -> EnvironmentData:
        """Build deterministic environment data for the configured case."""
        environment = _make_environment_data(
            self.n_boxes,
            self.n_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if self.n_boxes > 1:
            environment.temperature = (
                self.temperature
                + self.box_temperature_step
                * np.arange(
                    self.n_boxes,
                    dtype=np.float64,
                )
            )
            environment.pressure = (
                self.pressure
                + self.box_pressure_step
                * np.arange(
                    self.n_boxes,
                    dtype=np.float64,
                )
            )
        return environment

    def temperature_array(self) -> np.ndarray:
        """Return the per-box temperature inputs for the case."""
        return self.build_environment_data().temperature

    def pressure_array(self) -> np.ndarray:
        """Return the per-box pressure inputs for the case."""
        return self.build_environment_data().pressure


@dataclass(frozen=True)
class CondensationStiffnessClassification:
    """Store the particle-only stability classification for one trial.

    Attributes:
        label: Stability label derived from the recorded classification rule.
        mass_nonnegative: Whether all final particle masses stay non-negative.
        values_finite: Whether particle, gas, and vapor-pressure values are
            finite.
        metadata_valid: Whether shape and dtype checks passed for the case.
        zero_mass_change_stable: Whether zero-mass entries remain unchanged.
        max_fractional_mass_change: Largest positive-mass fractional change.
        threshold: Stability threshold applied to the trial.
        particle_only_update: Whether the contract only updates particle masses.
    """

    label: str
    mass_nonnegative: bool
    values_finite: bool
    metadata_valid: bool
    zero_mass_change_stable: bool
    max_fractional_mass_change: float
    threshold: float
    particle_only_update: bool = True


def _make_condensation_stiffness_cases() -> tuple[
    CondensationStiffnessCase, ...
]:
    """Return the compact reusable stiffness catalog."""
    return (
        CondensationStiffnessCase(
            name="nanometer",
            n_boxes=1,
            n_particles=4,
            n_species=2,
            time_step=0.05,
            particle_mass_scale=0.15,
            gas_concentration_scale=4.0,
            vapor_pressure_scale=0.6,
            zero_mass_particles=((0, 0),),
        ),
        CondensationStiffnessCase(
            name="accumulation_mode",
            n_boxes=1,
            n_particles=3,
            n_species=2,
            time_step=0.4,
            particle_mass_scale=15.0,
            gas_concentration_scale=1.5,
            vapor_pressure_scale=0.85,
        ),
        CondensationStiffnessCase(
            name="droplet_like",
            n_boxes=2,
            n_particles=2,
            n_species=2,
            time_step=4.0,
            particle_mass_scale=8.0e4,
            gas_concentration_scale=0.4,
            vapor_pressure_scale=1.25,
            box_temperature_step=4.0,
            box_pressure_step=-750.0,
        ),
    )


_RECORDED_TIMESTEP_GRID_BY_CASE: dict[str, tuple[float, ...]] = {
    "nanometer": (0.00005, 0.05, 50.0),
    "accumulation_mode": (0.004, 0.4, 40.0),
    "droplet_like": (0.04, 4.0, 400.0),
}


_RECORDED_STIFFNESS_THRESHOLD = 1.0


@dataclass(frozen=True)
class CondensationIntegrationCandidate:
    """Describe one deterministic fixed-shape integration candidate."""

    name: str
    family: str
    fixed_substeps: int
    reference_rtol: float
    baseline_relative_error_bound: float
    graph_capture_compatible: bool
    autodiff_implication: str


@dataclass
class CondensationCandidateScratch:
    """Own fixed-shape reusable scratch arrays for one candidate run."""

    mass_transfer: np.ndarray
    work: np.ndarray
    accumulator: np.ndarray | None = None


@dataclass(frozen=True)
class CondensationCandidateResult:
    """Capture deterministic candidate output and buffer identity evidence."""

    candidate_name: str
    final_masses: np.ndarray
    mass_transfer: np.ndarray
    work: np.ndarray
    accumulator: np.ndarray | None


_CONDENSATION_INTEGRATION_CANDIDATES: tuple[
    CondensationIntegrationCandidate, ...
] = (
    CondensationIntegrationCandidate(
        name="fixed_count_substeps_4",
        family="fixed_count_explicit",
        fixed_substeps=4,
        reference_rtol=5.0e-2,
        baseline_relative_error_bound=5.0e-2,
        graph_capture_compatible=True,
        autodiff_implication=(
            "Fixed loop count; no data-dependent branching beyond clamps."
        ),
    ),
    CondensationIntegrationCandidate(
        name="asymptotic_relaxation",
        family="asymptotic_first_order",
        fixed_substeps=1,
        reference_rtol=3.5e-1,
        baseline_relative_error_bound=3.5e-1,
        graph_capture_compatible=True,
        autodiff_implication=(
            "Uses exp-based bounded relaxation, so autodiff stays plausible "
            "but clamp boundaries remain non-smooth."
        ),
    ),
)


@dataclass(frozen=True)
class CondensationStiffnessTrialRecord:
    """Capture one recorded-grid timestep trial for a stiffness case.

    Attributes:
        case_name: Name of the deterministic stiffness case.
        time_step: Executed timestep for the trial.
        configured_time_step: Matching timestep from the recorded grid.
        timestep_index: Recorded-grid position for the trial.
        environment_input_mode: Whether scalar or Warp-array inputs were used.
        classification: Particle-only stability classification for the result.
        gas_unchanged: Whether executed Warp gas concentration stayed unchanged.
        reuses_caller_mass_transfer_buffer: Whether the caller buffer was
            reused.
        mass_transfer_has_nonzero_values: Whether the buffer was populated.
        mass_transfer_changed_from_previous_trial: Whether reuse overwrote
            values.
        final_masses: Final particle masses copied back from Warp.
        initial_masses: Initial particle masses used for classification.
        mass_transfer_values: CPU copy of the reused mass-transfer buffer.
    """

    case_name: str
    time_step: float
    configured_time_step: float
    timestep_index: int
    environment_input_mode: str
    classification: CondensationStiffnessClassification
    gas_unchanged: bool
    initial_gas_concentration: np.ndarray
    final_gas_concentration: np.ndarray
    initial_vapor_pressure: np.ndarray
    final_vapor_pressure: np.ndarray
    expected_vapor_pressure: np.ndarray
    reuses_caller_mass_transfer_buffer: bool
    reuses_work_mass_transfer_buffer: bool
    reuses_total_mass_transfer_buffer: bool
    reuses_dynamic_viscosity_buffer: bool
    reuses_mean_free_path_buffer: bool
    reuses_all_scratch_buffers: bool
    returned_total_is_scratch_total: bool
    mass_transfer_has_nonzero_values: bool
    mass_transfer_changed_from_previous_trial: bool
    values_finite: bool
    zero_mass_stable: bool
    vapor_pressure_refreshed: bool
    final_masses: np.ndarray
    initial_masses: np.ndarray
    mass_transfer_values: np.ndarray
    reference_final_masses: np.ndarray
    repeated_final_masses: np.ndarray
    repeated_mass_transfer_values: np.ndarray
    repeated_final_vapor_pressure: np.ndarray


def _validate_stiffness_case_metadata(
    case: CondensationStiffnessCase,
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
) -> None:
    """Validate fixed-shape and dtype expectations for a stiffness case."""
    expected_particle_shape = (case.n_boxes, case.n_particles, case.n_species)
    expected_concentration_shape = (case.n_boxes, case.n_particles)
    expected_gas_shape = (case.n_boxes, case.n_species)
    if particles.masses.shape != expected_particle_shape:
        raise ValueError(
            "particle masses shape does not match declared case metadata"
        )
    if particles.concentration.shape != expected_concentration_shape:
        raise ValueError(
            "particle concentration shape does not match declared case metadata"
        )
    if gas.concentration.shape != expected_gas_shape:
        raise ValueError(
            "gas concentration shape does not match declared case metadata"
        )
    if vapor_pressure.shape != expected_gas_shape:
        raise ValueError(
            "vapor pressure shape does not match declared case metadata"
        )

    dtype_expectations = (
        ("particle masses", particles.masses),
        ("particle concentration", particles.concentration),
        ("particle density", particles.density),
        ("particle volume", particles.volume),
        ("gas concentration", gas.concentration),
        ("gas molar mass", gas.molar_mass),
        ("vapor pressure", vapor_pressure),
    )
    for name, values in dtype_expectations:
        if values.dtype != np.float64:
            raise TypeError(f"{name} must use np.float64")


def _particle_mass_is_nonnegative(masses: np.ndarray) -> bool:
    """Return whether all particle masses are non-negative."""
    return bool(np.all(masses >= 0.0))


def _particle_values_are_finite(*arrays: np.ndarray) -> bool:
    """Return whether all provided arrays contain only finite values."""
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _fractional_mass_change_per_bin(
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
) -> np.ndarray:
    """Return per-bin fractional mass change for positive-mass entries only."""
    if initial_masses.shape != final_masses.shape:
        raise ValueError("initial and final masses must have matching shape")
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    change = np.abs(final - initial)
    fractional_change = np.zeros_like(change)
    positive_mask = initial > 0.0
    fractional_change[positive_mask] = (
        change[positive_mask] / initial[positive_mask]
    )
    return fractional_change


def _zero_mass_entries_remain_stable(
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
) -> bool:
    """Return whether zero-mass entries remain unchanged."""
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    if initial.shape != final.shape:
        raise ValueError("initial and final masses must have matching shape")
    zero_mask = initial == 0.0
    return bool(np.all(final[zero_mask] == 0.0))


def _classify_particle_only_condensation_stiffness(
    case: CondensationStiffnessCase,
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
    gas: GasData,
    vapor_pressure: np.ndarray,
    *,
    max_fractional_change: float,
) -> CondensationStiffnessClassification:
    """Classify particle-only condensation behavior as stable or unstable."""
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    particles = case.build_particle_data()
    particles.masses = final
    _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
    fractional_change = _fractional_mass_change_per_bin(
        initial,
        final,
    )
    zero_mass_change_stable = _zero_mass_entries_remain_stable(
        initial,
        final,
    )
    mass_nonnegative = _particle_mass_is_nonnegative(final)
    values_finite = _particle_values_are_finite(
        initial,
        final,
        gas.concentration,
        vapor_pressure,
    )
    max_change = (
        float(np.max(fractional_change)) if fractional_change.size else 0.0
    )
    label = "stable"
    if (
        max_change > max_fractional_change
        or not mass_nonnegative
        or not values_finite
        or not zero_mass_change_stable
    ):
        label = "unstable"
    return CondensationStiffnessClassification(
        label=label,
        mass_nonnegative=mass_nonnegative,
        values_finite=values_finite,
        metadata_valid=True,
        zero_mass_change_stable=zero_mass_change_stable,
        max_fractional_mass_change=max_change,
        threshold=max_fractional_change,
    )


def _cpu_mass_transfer(  # noqa: C901
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    surface_tension: np.ndarray,
    mass_accommodation: np.ndarray,
    diffusion_coefficient_vapor: np.ndarray,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
    time_step: float,
    thermodynamics: ThermodynamicsConfig | None = None,
    activity_mode: int | None = None,
    surface_tension_mode: int | None = None,
    water_species_index: int = 0,
    kappas: np.ndarray | None = None,
    latent_heat: np.ndarray | None = None,
) -> np.ndarray:
    """Compute CPU mass transfer matching refreshed GPU kernel physics."""
    n_boxes, n_particles, n_species = particles.masses.shape
    mass_transfer = np.zeros_like(particles.masses)
    temperature_array = np.full((n_boxes,), temperature, dtype=np.float64)
    if isinstance(temperature, np.ndarray):
        temperature_array = np.asarray(temperature, dtype=np.float64)
    pressure_array = np.full((n_boxes,), pressure, dtype=np.float64)
    if isinstance(pressure, np.ndarray):
        pressure_array = np.asarray(pressure, dtype=np.float64)
    if thermodynamics is not None:
        vapor_pressure = _evaluate_vapor_pressure_config(
            thermodynamics,
            temperature_array,
        )

    for box_idx in range(n_boxes):
        box_temperature = float(temperature_array[box_idx])
        box_pressure = float(pressure_array[box_idx])
        dynamic_viscosity = get_dynamic_viscosity(
            box_temperature,
            reference_viscosity=constants.REF_VISCOSITY_AIR_STP,
            reference_temperature=constants.REF_TEMPERATURE_STP,
        )
        mean_free_path = get_molecule_mean_free_path(
            molar_mass=constants.MOLECULAR_WEIGHT_AIR,
            temperature=box_temperature,
            pressure=box_pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        for particle_idx in range(n_particles):
            if particles.concentration[box_idx, particle_idx] == 0.0:
                continue
            total_volume = np.sum(
                particles.masses[box_idx, particle_idx, :] / particles.density
            )
            if total_volume <= 0.0:
                continue
            total_mass = np.sum(particles.masses[box_idx, particle_idx, :])
            radius = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
            effective_density = (
                total_mass / total_volume if total_volume > 0.0 else 0.0
            )
            if effective_density <= 0.0:
                effective_density = particles.density[0]

            knudsen_number = get_knudsen_number(mean_free_path, radius)
            slip_correction = get_cunningham_slip_correction(knudsen_number)
            mobility = get_aerodynamic_mobility(
                particle_radius=radius,
                slip_correction_factor=slip_correction,
                dynamic_viscosity=dynamic_viscosity,
            )
            diffusion_particle = get_diffusion_coefficient(
                temperature=box_temperature,
                aerodynamic_mobility=mobility,
                boltzmann_constant=constants.BOLTZMANN_CONSTANT,
            )

            for species_idx in range(n_species):
                transition = get_vapor_transition_correction(
                    knudsen_number=knudsen_number,
                    mass_accommodation=mass_accommodation[species_idx],
                )
                diffusion_value = diffusion_coefficient_vapor[species_idx]
                if diffusion_value <= 0.0:
                    diffusion_value = diffusion_particle
                mass_transport = get_first_order_mass_transport_k(
                    particle_radius=radius,
                    vapor_transition=transition,
                    diffusion_coefficient=diffusion_value,
                )
                tension = surface_tension[species_idx]
                if surface_tension_mode == int(
                    SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED
                ):
                    volumes = particles.masses[box_idx, particle_idx, :] / (
                        particles.density
                    )
                    total_particle_volume = np.sum(volumes)
                    tension = (
                        np.dot(surface_tension, volumes) / total_particle_volume
                        if total_particle_volume > 0.0
                        else np.mean(surface_tension)
                    )
                kelvin_radius = get_kelvin_radius(
                    effective_surface_tension=tension,
                    effective_density=effective_density,
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=box_temperature,
                )
                kelvin_term = get_kelvin_term(radius, kelvin_radius)
                partial_pressure_gas = get_partial_pressure(
                    concentration=gas.concentration[box_idx, species_idx],
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=box_temperature,
                )
                activity_factor = 1.0
                if (
                    species_idx == water_species_index
                    and activity_mode is not None
                ):
                    if activity_mode == int(ACTIVITY_MODE_IDEAL):
                        moles = (
                            particles.masses[box_idx, particle_idx, :]
                            / gas.molar_mass
                        )
                        activity_factor = (
                            moles[water_species_index] / np.sum(moles)
                            if np.sum(moles) > 0.0
                            else 0.0
                        )
                    else:
                        assert kappas is not None
                        volumes = (
                            particles.masses[box_idx, particle_idx, :]
                            / particles.density
                        )
                        water_volume = volumes[water_species_index]
                        solute_mask = (
                            np.arange(n_species) != water_species_index
                        )
                        solute_volume = np.sum(volumes[solute_mask])
                        if water_volume == 0.0:
                            activity_factor = 0.0
                        elif solute_volume > 0.0:
                            activity_factor = 1.0 / (
                                1.0
                                + np.sum(
                                    kappas[solute_mask] * volumes[solute_mask]
                                )
                                / water_volume
                            )
                surface_vapor_pressure = (
                    activity_factor
                    * vapor_pressure[box_idx, species_idx]
                    * kelvin_term
                )
                pressure_delta = partial_pressure_gas - surface_vapor_pressure
                if latent_heat is None or latent_heat[species_idx] == 0.0:
                    mass_rate = get_mass_transfer_rate(
                        pressure_delta=pressure_delta,
                        first_order_mass_transport=mass_transport,
                        temperature=box_temperature,
                        molar_mass=gas.molar_mass[species_idx],
                    )
                else:
                    mass_rate = get_mass_transfer_rate_latent_heat(
                        pressure_delta=pressure_delta,
                        first_order_mass_transport=mass_transport,
                        temperature=box_temperature,
                        molar_mass=gas.molar_mass[species_idx],
                        latent_heat=latent_heat[species_idx],
                        thermal_conductivity=get_thermal_conductivity(
                            box_temperature
                        ),
                        vapor_pressure_surface=surface_vapor_pressure,
                        diffusion_coefficient=diffusion_value,
                    )
                mass_transfer[box_idx, particle_idx, species_idx] = (
                    mass_rate * time_step
                )
    return mass_transfer


def _cpu_four_substep_oracle(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Return four-substep final masses, applied total, and final proposal.

    The production entry point refreshes properties and recomputes proposals
    from the predecessor mass at each fixed interval. This independent oracle
    mirrors that contract while retaining the final raw proposal separately
    from the accumulated, clamped transfer returned to callers.
    """
    particles = kwargs["particles"] if "particles" in kwargs else args[0]
    return_gas = kwargs.pop("return_gas", False)
    partitioning = kwargs.pop("partitioning", None)
    time_step = kwargs["time_step"] if "time_step" in kwargs else args[8]
    gas = kwargs["gas"] if "gas" in kwargs else args[1]
    working_particles = dataclasses.replace(
        particles,
        masses=np.asarray(particles.masses, dtype=np.float64).copy(),
    )
    working_gas = dataclasses.replace(
        gas,
        concentration=np.asarray(gas.concentration, dtype=np.float64).copy(),
    )
    call_args = list(args)
    if call_args:
        call_args[0] = working_particles
        call_args[1] = working_gas
    total = np.zeros_like(working_particles.masses)
    proposal = np.zeros_like(working_particles.masses)
    for _ in range(4):
        call_kwargs = dict(kwargs)
        if len(call_args) > 8:
            call_args[8] = time_step / 4.0
        else:
            call_kwargs["time_step"] = time_step / 4.0
        if "gas" in call_kwargs:
            call_kwargs["gas"] = working_gas
        proposal = _cpu_mass_transfer(*call_args, **call_kwargs)
        if partitioning is not None:
            proposal[:, :, ~np.asarray(partitioning, dtype=bool)] = 0.0
        _, _, _, _, applied = _inventory_reference(
            working_particles.masses,
            working_particles.concentration,
            working_gas.concentration,
            proposal,
        )
        working_particles.masses += applied
        working_gas.concentration -= np.sum(
            applied * working_particles.concentration[:, :, None], axis=1
        )
        total += applied
    result = (working_particles.masses.copy(), total, proposal)
    if return_gas:
        return (*result, working_gas.concentration.copy())
    return result


def _get_condensation_stiffness_case(name: str) -> CondensationStiffnessCase:
    """Return one named deterministic stiffness case."""
    return next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == name
    )


def _get_integration_candidate(name: str) -> CondensationIntegrationCandidate:
    """Return one named deterministic integration candidate."""
    return next(
        candidate
        for candidate in _CONDENSATION_INTEGRATION_CANDIDATES
        if candidate.name == name
    )


def _make_candidate_scratch(
    case: CondensationStiffnessCase,
    candidate: CondensationIntegrationCandidate,
) -> CondensationCandidateScratch:
    """Allocate fixed-shape reusable scratch buffers for one candidate."""
    shape = (case.n_boxes, case.n_particles, case.n_species)
    accumulator = None
    if candidate.family == "fixed_count_explicit":
        accumulator = np.zeros(shape, dtype=np.float64)
    return CondensationCandidateScratch(
        mass_transfer=np.zeros(shape, dtype=np.float64),
        work=np.zeros(shape, dtype=np.float64),
        accumulator=accumulator,
    )


def _environment_inputs_for_case(
    case: CondensationStiffnessCase,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return scalar or per-box CPU environment inputs for one case."""
    if case.n_boxes == 1:
        return case.temperature, case.pressure
    return case.temperature_array(), case.pressure_array()


def _apply_particle_only_mass_transfer(
    particle_masses: np.ndarray,
    mass_transfer: np.ndarray,
) -> np.ndarray:
    """Apply a particle-only mass-transfer update with a non-negative clamp."""
    bounded_transfer = np.maximum(mass_transfer, -particle_masses)
    return particle_masses + bounded_transfer


# Parity tolerances cover independent fp64 equation evaluation.  Ownership
# invariants are deliberately tighter because they compare one kernel's buffers.
P4_PARITY_RTOL = 2.0e-10
P4_PARITY_ATOL = 1.0e-30
P4_INVARIANT_RTOL = 1.0e-12
P4_INVARIANT_ATOL = 1.0e-30


@dataclass(frozen=True)
class P4CondensationCase:
    """Immutable NumPy-only descriptor for a direct-kernel parity fixture."""

    name: str
    masses: np.ndarray
    gas_concentration: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    clamp_index: tuple[int, int, int]

    def build_particle_data(self) -> ParticleData:
        """Build detached particle data for one execution."""
        n_boxes, n_particles, n_species = self.masses.shape
        return ParticleData(
            masses=self.masses.copy(),
            concentration=np.ones((n_boxes, n_particles), dtype=np.float64),
            charge=np.zeros((n_boxes, n_particles), dtype=np.float64),
            density=np.array([1000.0, 1450.0], dtype=np.float64),
            volume=np.full(n_boxes, 1.0e-6, dtype=np.float64),
        )

    def build_gas_data(self) -> GasData:
        """Build detached gas data for one execution."""
        return GasData(
            name=["water", "solute"],
            molar_mass=np.array([0.018, 0.098], dtype=np.float64),
            concentration=self.gas_concentration.copy(),
            partitioning=np.array([True, True]),
        )


@dataclass(frozen=True)
class P4ThermodynamicsMetadata:
    """NumPy-owned sidecar metadata used by the independent reference."""

    modes: np.ndarray
    parameters: np.ndarray
    water_species_index: int
    kappas: np.ndarray
    molar_mass_reference: np.ndarray
    surface_tension: np.ndarray


P4_THERMODYNAMICS = P4ThermodynamicsMetadata(
    modes=np.array([1, 0], dtype=np.int32),
    parameters=np.array([[0.0, 0.0, 0.0, 0.0], [1400.0, 0.0, 0.0, 0.0]]),
    water_species_index=0,
    kappas=np.array([0.0, 0.55], dtype=np.float64),
    molar_mass_reference=np.array([0.018, 0.098], dtype=np.float64),
    surface_tension=np.array([0.035, 0.094], dtype=np.float64),
)


P4_CASES: tuple[P4CondensationCase, ...] = (
    P4CondensationCase(
        name="one_box",
        masses=np.array([[[1.0e-18, 0.0], [1.0e-18, 9.0e-18]]]),
        gas_concentration=np.array([[1.0e-12, 2.0e-12]], dtype=np.float64),
        temperature=np.array([298.15], dtype=np.float64),
        pressure=np.array([101325.0], dtype=np.float64),
        clamp_index=(0, 0, 0),
    ),
    P4CondensationCase(
        name="multi_box",
        masses=np.array(
            [
                [[1.0e-18, 0.0], [2.0e-18, 8.0e-18]],
                [[3.0e-18, 0.0], [7.0e-18, 3.0e-18]],
            ],
            dtype=np.float64,
        ),
        gas_concentration=np.array(
            [[1.0e-12, 2.0e-12], [2.0e-12, 3.0e-12]], dtype=np.float64
        ),
        temperature=np.array([268.15, 278.15], dtype=np.float64),
        pressure=np.array([101325.0, 99000.0], dtype=np.float64),
        clamp_index=(0, 0, 0),
    ),
)


def _p4_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Evaluate constant and Buck vapor pressure from NumPy-only metadata."""
    temperatures = np.asarray(temperature, dtype=np.float64)
    output = np.empty((temperatures.size, 2), dtype=np.float64)
    celsius = temperatures - 273.15
    ice = celsius < 0.0
    output[ice, 0] = 611.15 * np.exp(
        (23.036 - celsius[ice] / 333.7) * celsius[ice] / (279.82 + celsius[ice])
    )
    output[~ice, 0] = 611.21 * np.exp(
        (18.678 - celsius[~ice] / 234.5)
        * celsius[~ice]
        / (257.14 + celsius[~ice])
    )
    output[:, 1] = P4_THERMODYNAMICS.parameters[1, 0]
    return output


def _p4_reference(
    case: P4CondensationCase, activity_mode: int, surface_mode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent fp64 final mass, transfer, vapor, and gas state."""
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = _p4_vapor_pressure(case.temperature)
    final_masses, total_transfer, _, final_gas = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension=P4_THERMODYNAMICS.surface_tension,
        mass_accommodation=np.ones(2, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(2, 2.0e-5, dtype=np.float64),
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=1000.0,
        activity_mode=activity_mode,
        surface_tension_mode=surface_mode,
        water_species_index=P4_THERMODYNAMICS.water_species_index,
        kappas=P4_THERMODYNAMICS.kappas,
        return_gas=True,
    )
    return (
        final_masses,
        total_transfer,
        vapor_pressure,
        final_gas,
    )


def _p4_sidecars(device: str) -> tuple[Any, Any, Any]:
    """Build fresh Warp sidecars from copied NumPy metadata."""
    runtime = _load_warp_runtime()
    wp = runtime.wp
    thermodynamics = runtime.ThermodynamicsConfig(
        modes=wp.array(
            P4_THERMODYNAMICS.modes.copy(), dtype=wp.int32, device=device
        ),
        parameters=wp.array(
            P4_THERMODYNAMICS.parameters.copy(), dtype=wp.float64, device=device
        ),
        molar_mass_reference=wp.array(
            P4_THERMODYNAMICS.molar_mass_reference.copy(),
            dtype=wp.float64,
            device=device,
        ),
    )
    activity_surface = runtime.CondensationActivitySurfaceConfig(
        activity_mode=0,
        surface_tension_mode=0,
        water_species_index=P4_THERMODYNAMICS.water_species_index,
        kappas=wp.array(
            P4_THERMODYNAMICS.kappas.copy(), dtype=wp.float64, device=device
        ),
        molar_mass_reference=wp.array(
            P4_THERMODYNAMICS.molar_mass_reference.copy(),
            dtype=wp.float64,
            device=device,
        ),
    )
    tension = wp.array(
        P4_THERMODYNAMICS.surface_tension.copy(),
        dtype=wp.float64,
        device=device,
    )
    return thermodynamics, activity_surface, tension


P4_MODE_PAIRS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)


def _assert_p4_gpu_matches_reference(
    case: P4CondensationCase,
    activity_mode: int,
    surface_mode: int,
    device: str,
) -> None:
    """Execute one P4 matrix element and assert parity plus ownership rules."""
    runtime = _load_warp_runtime()
    wp = runtime.wp
    (
        expected_final,
        expected_total,
        expected_vapor,
        expected_gas,
    ) = _p4_reference(case, activity_mode, surface_mode)
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    initial_mass = particles.masses.copy()
    initial_molar_mass = gas.molar_mass.copy()
    initial_partitioning = gas.partitioning.copy()
    gpu_particles = runtime.to_warp_particle_data(particles, device=device)
    gpu_gas = runtime.to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.zeros_like(expected_vapor),
    )
    thermodynamics, activity_surface, tension = _p4_sidecars(device)
    activity_surface = dataclasses.replace(
        activity_surface,
        activity_mode=activity_mode,
        surface_tension_mode=surface_mode,
    )
    _, total_transfer = runtime._condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=wp.array(case.temperature, dtype=wp.float64, device=device),
        pressure=wp.array(case.pressure, dtype=wp.float64, device=device),
        time_step=1000.0,
        surface_tension=tension,
        thermodynamics=thermodynamics,
        activity_surface=activity_surface,
    )
    final_mass = runtime.from_warp_particle_data(
        gpu_particles, sync=True
    ).masses
    total = total_transfer.numpy().copy()
    gas_concentration = gpu_gas.concentration.numpy().copy()
    vapor_pressure = gpu_gas.vapor_pressure.numpy().copy()
    molar_mass = gpu_gas.molar_mass.numpy().copy()
    partitioning = gpu_gas.partitioning.numpy().copy()

    npt.assert_allclose(
        total, expected_total, rtol=P4_PARITY_RTOL, atol=P4_PARITY_ATOL
    )
    npt.assert_allclose(
        final_mass, expected_final, rtol=P4_PARITY_RTOL, atol=P4_PARITY_ATOL
    )
    npt.assert_allclose(
        vapor_pressure, expected_vapor, rtol=P4_PARITY_RTOL, atol=P4_PARITY_ATOL
    )
    npt.assert_allclose(
        gas_concentration,
        expected_gas,
        rtol=P4_PARITY_RTOL,
        atol=P4_PARITY_ATOL,
    )
    npt.assert_array_equal(molar_mass, initial_molar_mass)
    npt.assert_array_equal(
        partitioning,
        np.broadcast_to(
            initial_partitioning.astype(np.int32), partitioning.shape
        ),
    )

    assert np.all(np.isfinite(final_mass)) and np.all(final_mass >= 0.0)
    assert np.all(np.isfinite(total)) and np.all(np.isfinite(vapor_pressure))
    npt.assert_allclose(
        final_mass,
        initial_mass + total,
        rtol=P4_INVARIANT_RTOL,
        atol=P4_INVARIANT_ATOL,
    )
    npt.assert_allclose(
        gas_concentration,
        expected_gas,
        rtol=P4_INVARIANT_RTOL,
        atol=P4_INVARIANT_ATOL,
    )
    clamp_index = case.clamp_index
    assert total[clamp_index] == -initial_mass[clamp_index]
    assert final_mass[clamp_index] == 0.0


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("case", P4_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(("activity_mode", "surface_mode"), P4_MODE_PAIRS)
def test_condensation_activity_surface_warp_cpu_matches_numpy_reference(
    warp_cpu_device: str,
    case: P4CondensationCase,
    activity_mode: int,
    surface_mode: int,
) -> None:
    """Warp CPU matches the independent P4 NumPy reference matrix."""
    _assert_p4_gpu_matches_reference(
        case, activity_mode, surface_mode, warp_cpu_device
    )


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("case", P4_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(("activity_mode", "surface_mode"), P4_MODE_PAIRS)
def test_condensation_activity_surface_cuda_matches_numpy_reference(
    cuda_device: str,
    case: P4CondensationCase,
    activity_mode: int,
    surface_mode: int,
) -> None:
    """Optional CUDA matches the independent P4 NumPy reference matrix."""
    _assert_p4_gpu_matches_reference(
        case, activity_mode, surface_mode, cuda_device
    )


@lru_cache(maxsize=None)
def _cpu_reference_final_masses(case_name: str, time_step: float) -> np.ndarray:
    """Cache CPU reference final masses for one case and timestep."""
    case = _get_condensation_stiffness_case(case_name)
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    temperature, pressure = _environment_inputs_for_case(case)
    mass_transfer = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        surface_tension=np.full(case.n_species, 0.072, dtype=np.float64),
        mass_accommodation=np.full(case.n_species, 1.0, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(
            case.n_species,
            2.0e-5,
            dtype=np.float64,
        ),
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
    )
    return _apply_particle_only_mass_transfer(
        particles.masses.copy(),
        mass_transfer,
    )


@lru_cache(maxsize=None)
def _recorded_explicit_final_masses_by_case(
    case_name: str,
) -> tuple[np.ndarray, ...]:
    """Cache recorded explicit baseline final masses for one named case."""
    case = _get_condensation_stiffness_case(case_name)
    records = _record_condensation_stiffness_trials(case, device="cpu")
    return tuple(record.final_masses.copy() for record in records)


def _copy_condensation_stiffness_trial_record(
    record: CondensationStiffnessTrialRecord,
) -> CondensationStiffnessTrialRecord:
    """Return a detached copy of one recorded-grid trial record."""
    return CondensationStiffnessTrialRecord(
        case_name=record.case_name,
        time_step=record.time_step,
        configured_time_step=record.configured_time_step,
        timestep_index=record.timestep_index,
        environment_input_mode=record.environment_input_mode,
        classification=record.classification,
        gas_unchanged=record.gas_unchanged,
        initial_gas_concentration=record.initial_gas_concentration.copy(),
        final_gas_concentration=record.final_gas_concentration.copy(),
        initial_vapor_pressure=record.initial_vapor_pressure.copy(),
        final_vapor_pressure=record.final_vapor_pressure.copy(),
        expected_vapor_pressure=record.expected_vapor_pressure.copy(),
        reuses_caller_mass_transfer_buffer=(
            record.reuses_caller_mass_transfer_buffer
        ),
        reuses_work_mass_transfer_buffer=(
            record.reuses_work_mass_transfer_buffer
        ),
        reuses_total_mass_transfer_buffer=(
            record.reuses_total_mass_transfer_buffer
        ),
        reuses_dynamic_viscosity_buffer=(
            record.reuses_dynamic_viscosity_buffer
        ),
        reuses_mean_free_path_buffer=record.reuses_mean_free_path_buffer,
        reuses_all_scratch_buffers=record.reuses_all_scratch_buffers,
        returned_total_is_scratch_total=record.returned_total_is_scratch_total,
        mass_transfer_has_nonzero_values=record.mass_transfer_has_nonzero_values,
        mass_transfer_changed_from_previous_trial=(
            record.mass_transfer_changed_from_previous_trial
        ),
        values_finite=record.values_finite,
        zero_mass_stable=record.zero_mass_stable,
        vapor_pressure_refreshed=record.vapor_pressure_refreshed,
        final_masses=record.final_masses.copy(),
        initial_masses=record.initial_masses.copy(),
        mass_transfer_values=record.mass_transfer_values.copy(),
        reference_final_masses=record.reference_final_masses.copy(),
        repeated_final_masses=record.repeated_final_masses.copy(),
        repeated_mass_transfer_values=(
            record.repeated_mass_transfer_values.copy()
        ),
        repeated_final_vapor_pressure=(
            record.repeated_final_vapor_pressure.copy()
        ),
    )


@lru_cache(maxsize=None)
def _cached_recorded_condensation_stiffness_trials(
    case_name: str,
    device: str,
) -> tuple[CondensationStiffnessTrialRecord, ...]:
    """Cache standard recorded-grid sweeps for mutation-free callers."""
    case = _get_condensation_stiffness_case(case_name)
    records = _record_condensation_stiffness_trials_uncached(case, device)
    return tuple(
        _copy_condensation_stiffness_trial_record(record) for record in records
    )


def _relative_mass_error(
    candidate_masses: np.ndarray,
    reference_masses: np.ndarray,
) -> float:
    """Return the max relative mass error with a finite nonzero denominator."""
    scale = np.maximum(np.abs(reference_masses), 1.0e-30)
    return float(np.max(np.abs(candidate_masses - reference_masses) / scale))


def _run_integration_candidate(
    case: CondensationStiffnessCase,
    candidate: CondensationIntegrationCandidate,
    time_step: float,
    scratch: CondensationCandidateScratch,
) -> CondensationCandidateResult:
    """Run one deterministic fixed-shape candidate with reusable scratch."""
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    temperature, pressure = _environment_inputs_for_case(case)
    scratch.mass_transfer.fill(0.0)
    scratch.work.fill(0.0)
    if scratch.accumulator is not None:
        scratch.accumulator.fill(0.0)

    if candidate.family == "fixed_count_explicit":
        substep_time_step = time_step / candidate.fixed_substeps
        assert scratch.accumulator is not None
        for _ in range(candidate.fixed_substeps):
            scratch.work[...] = _cpu_mass_transfer(
                particles,
                gas,
                vapor_pressure,
                surface_tension=np.full(
                    case.n_species, 0.072, dtype=np.float64
                ),
                mass_accommodation=np.full(
                    case.n_species,
                    1.0,
                    dtype=np.float64,
                ),
                diffusion_coefficient_vapor=np.full(
                    case.n_species,
                    2.0e-5,
                    dtype=np.float64,
                ),
                temperature=temperature,
                pressure=pressure,
                time_step=substep_time_step,
            )
            scratch.work[...] = np.maximum(scratch.work, -particles.masses)
            particles.masses += scratch.work
            scratch.accumulator += scratch.work
        scratch.mass_transfer[...] = scratch.accumulator
    elif candidate.family == "asymptotic_first_order":
        scratch.mass_transfer[...] = _cpu_mass_transfer(
            particles,
            gas,
            vapor_pressure,
            surface_tension=np.full(case.n_species, 0.072, dtype=np.float64),
            mass_accommodation=np.full(case.n_species, 1.0, dtype=np.float64),
            diffusion_coefficient_vapor=np.full(
                case.n_species,
                2.0e-5,
                dtype=np.float64,
            ),
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
        )
        positive_transfer = np.maximum(scratch.mass_transfer, 0.0)
        negative_transfer = np.minimum(scratch.mass_transfer, 0.0)
        reference_scale = np.maximum(particles.masses, 1.0e-30)
        relaxation_ratio = positive_transfer / reference_scale
        scratch.work[...] = (
            particles.masses * (1.0 - np.exp(-relaxation_ratio))
        ) + np.maximum(negative_transfer, -particles.masses)
        particles.masses += scratch.work
        scratch.mass_transfer[...] = scratch.work
    else:
        raise ValueError(f"Unknown candidate family: {candidate.family}")

    return CondensationCandidateResult(
        candidate_name=candidate.name,
        final_masses=particles.masses.copy(),
        mass_transfer=scratch.mass_transfer,
        work=scratch.work,
        accumulator=scratch.accumulator,
    )


@overload
def _run_gpu_step(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    device: str,
    surface_tension: np.ndarray | None = None,
    mass_accommodation: np.ndarray | None = None,
    diffusion_coefficient_vapor: np.ndarray | None = None,
    mass_transfer: Any | None = None,
    environment: Any | None = None,
    thermodynamics: ThermodynamicsConfig | None = None,
    activity_surface: CondensationActivitySurfaceConfig | None = None,
    scratch_buffers: Any | None = None,
    return_gas_state: Literal[False] = False,
) -> tuple[ParticleData, Any]: ...


@overload
def _run_gpu_step(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    device: str,
    surface_tension: np.ndarray | None = None,
    mass_accommodation: np.ndarray | None = None,
    diffusion_coefficient_vapor: np.ndarray | None = None,
    mass_transfer: Any | None = None,
    environment: Any | None = None,
    thermodynamics: ThermodynamicsConfig | None = None,
    activity_surface: CondensationActivitySurfaceConfig | None = None,
    scratch_buffers: Any | None = None,
    return_gas_state: Literal[True] = True,
) -> tuple[ParticleData, Any, GasData]: ...


def _run_gpu_step(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    device: str,
    surface_tension: np.ndarray | None = None,
    mass_accommodation: np.ndarray | None = None,
    diffusion_coefficient_vapor: np.ndarray | None = None,
    mass_transfer: Any | None = None,
    environment: Any | None = None,
    thermodynamics: ThermodynamicsConfig | None = None,
    activity_surface: CondensationActivitySurfaceConfig | None = None,
    scratch_buffers: Any | None = None,
    return_gas_state: bool = False,
) -> tuple[ParticleData, Any] | tuple[ParticleData, Any, GasData]:
    """Run GPU condensation step and return CPU particle data.

    Args:
        particles: CPU particle inputs.
        gas: CPU gas inputs.
        vapor_pressure: CPU vapor-pressure inputs.
        temperature: Scalar or Warp temperature inputs.
        pressure: Scalar or Warp pressure inputs.
        time_step: Timestep passed to the Warp kernel.
        device: Warp device name.
        surface_tension: Optional per-species surface tension array.
        mass_accommodation: Optional per-species accommodation array.
        diffusion_coefficient_vapor: Optional per-species diffusion array.
        mass_transfer: Optional caller-owned Warp mass-transfer buffer.
        environment: Optional explicit Warp environment container.
        thermodynamics: Optional supplied config; defaults to constant models.
        return_gas_state: If True, also round-trip the executed Warp gas state.

    Returns:
        CPU particle data plus the Warp mass-transfer buffer. When
        ``return_gas_state`` is True, also returns the executed CPU gas state.
    """
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    if thermodynamics is None:
        thermodynamics = _make_thermodynamics_config(gpu_gas)
    _, mass_transfer_buffer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        surface_tension=surface_tension,
        mass_accommodation=mass_accommodation,
        diffusion_coefficient_vapor=diffusion_coefficient_vapor,
        mass_transfer=mass_transfer,
        environment=environment,
        thermodynamics=thermodynamics,
        activity_surface=activity_surface,
        scratch_buffers=scratch_buffers,
    )
    cpu_particles = from_warp_particle_data(gpu_particles, sync=True)
    if not return_gas_state:
        return cpu_particles, mass_transfer_buffer
    cpu_gas = from_warp_gas_data(gpu_gas, name=gas.name, sync=False)
    return cpu_particles, mass_transfer_buffer, cpu_gas


PRODUCTION_PARITY_RTOL = 2.0e-10
PRODUCTION_PARITY_ATOL = 1.0e-30
PRODUCTION_INVENTORY_RTOL = 1.0e-12
PRODUCTION_INVENTORY_ATOL = 1.0e-30


def _assert_particle_gas_inventory_conserved(
    initial_masses: np.ndarray,
    particle_concentration: np.ndarray,
    initial_gas: np.ndarray,
    final_masses: np.ndarray,
    final_gas: np.ndarray,
) -> None:
    """Assert per-box, per-species particle-plus-gas conservation."""
    residual = np.sum(
        (final_masses - initial_masses) * particle_concentration[:, :, None],
        axis=1,
    ) + (final_gas - initial_gas)
    npt.assert_allclose(
        residual,
        np.zeros_like(residual),
        rtol=PRODUCTION_INVENTORY_RTOL,
        atol=PRODUCTION_INVENTORY_ATOL,
    )
    for values in (initial_masses, initial_gas, final_masses, final_gas):
        assert np.all(np.isfinite(values))
    assert np.all(final_masses >= 0.0)
    assert np.all(final_gas >= 0.0)


def _make_production_inventory_case() -> tuple[
    ParticleData,
    GasData,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build detached fp64 inputs for public multi-box inventory coverage."""
    particles = ParticleData(
        masses=np.array(
            [
                [
                    [2.0e-18, 4.0e-18, 6.0e-18],
                    [3.0e-18, 5.0e-18, 7.0e-18],
                    [4.0e-18, 6.0e-18, 8.0e-18],
                ],
                [
                    [5.0e-18, 7.0e-18, 9.0e-18],
                    [6.0e-18, 8.0e-18, 1.0e-17],
                    [7.0e-18, 9.0e-18, 1.1e-17],
                ],
            ],
            dtype=np.float64,
        ),
        concentration=np.array(
            [[1.0, 0.0, 2.0], [1.5, 0.0, 0.5]], dtype=np.float64
        ),
        charge=np.zeros((2, 3), dtype=np.float64),
        density=np.array([1000.0, 1200.0, 1400.0], dtype=np.float64),
        volume=np.full(2, 1.0e-6, dtype=np.float64),
    )
    gas = GasData(
        name=["uptake", "disabled", "evaporation"],
        molar_mass=np.array([0.018, 0.058, 0.098], dtype=np.float64),
        concentration=np.array(
            [[1.0e-17, 2.0e-17, 0.0], [2.0e-17, 3.0e-17, 0.0]],
            dtype=np.float64,
        ),
        partitioning=np.array([True, False, True]),
    )
    vapor_pressure = np.array(
        [[1.0e-15, 500.0, 2.0e3], [1.0e-15, 500.0, 2.0e3]],
        dtype=np.float64,
    )
    temperature = np.array([298.15, 303.15], dtype=np.float64)
    pressure = np.array([101325.0, 99000.0], dtype=np.float64)
    surface_tension = np.full(3, 0.072, dtype=np.float64)
    mass_accommodation = np.ones(3, dtype=np.float64)
    diffusion = np.full(3, 2.0e-5, dtype=np.float64)
    return (
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
    )


def _run_production_inventory_case(device: str) -> None:
    """Run public-hook inventory and CPU-oracle parity regression coverage."""
    runtime = _load_warp_runtime()
    (
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
    ) = _make_production_inventory_case()
    initial_masses = particles.masses.copy()
    initial_gas = gas.concentration.copy()
    particle_concentration = particles.concentration.copy()
    npt.assert_array_equal(initial_gas[:, 2], 0.0)
    expected_masses, _, _, expected_gas = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        0.1,
        partitioning=gas.partitioning,
        return_gas=True,
    )
    thermodynamics = runtime.ThermodynamicsConfig(
        modes=runtime.wp.zeros(3, dtype=runtime.wp.int32, device=device),
        parameters=runtime.wp.array(
            np.column_stack((vapor_pressure[0], np.zeros((3, 3)))),
            dtype=runtime.wp.float64,
            device=device,
        ),
        molar_mass_reference=runtime.wp.array(
            gas.molar_mass, dtype=runtime.wp.float64, device=device
        ),
    )
    final_particles, total_transfer, final_gas = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        runtime.wp.array(temperature, dtype=runtime.wp.float64, device=device),
        runtime.wp.array(pressure, dtype=runtime.wp.float64, device=device),
        0.1,
        device,
        surface_tension=runtime.wp.array(
            surface_tension, dtype=runtime.wp.float64, device=device
        ),
        mass_accommodation=runtime.wp.array(
            mass_accommodation, dtype=runtime.wp.float64, device=device
        ),
        diffusion_coefficient_vapor=runtime.wp.array(
            diffusion, dtype=runtime.wp.float64, device=device
        ),
        thermodynamics=thermodynamics,
        return_gas_state=True,
    )
    final_masses = final_particles.masses
    final_gas_concentration = final_gas.concentration
    total = total_transfer.numpy().copy()
    _assert_particle_gas_inventory_conserved(
        initial_masses,
        particle_concentration,
        initial_gas,
        final_masses,
        final_gas_concentration,
    )
    npt.assert_allclose(
        final_masses,
        expected_masses,
        rtol=PRODUCTION_PARITY_RTOL,
        atol=PRODUCTION_PARITY_ATOL,
    )
    npt.assert_allclose(
        final_gas_concentration,
        expected_gas,
        rtol=PRODUCTION_PARITY_RTOL,
        atol=PRODUCTION_PARITY_ATOL,
    )
    npt.assert_allclose(
        total,
        final_masses - initial_masses,
        rtol=PRODUCTION_PARITY_RTOL,
        atol=PRODUCTION_PARITY_ATOL,
    )
    npt.assert_array_equal(total[:, :, 1], 0.0)
    npt.assert_array_equal(final_masses[:, :, 1], initial_masses[:, :, 1])
    npt.assert_array_equal(final_gas_concentration[:, 1], initial_gas[:, 1])
    inactive = particle_concentration == 0.0
    assert np.any(total[:, :, 0][~inactive] > 0.0)
    assert np.any(total[:, :, 2][~inactive] < 0.0)
    assert np.any(final_gas_concentration[:, 2] > 0.0)
    npt.assert_array_equal(total[inactive], 0.0)
    npt.assert_array_equal(final_masses[inactive], initial_masses[inactive])


@pytest.mark.gpu_parity
def test_condensation_public_inventory_warp_cpu_matches_oracle(
    warp_cpu_device: str,
) -> None:
    """Public Warp CPU hook conserves inventory and matches its oracle."""
    _run_production_inventory_case(warp_cpu_device)


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_condensation_public_inventory_cuda_matches_oracle(
    cuda_device: str,
) -> None:
    """Public CUDA hook conserves inventory and matches its oracle."""
    _run_production_inventory_case(cuda_device)


def test_condensation_scratch_buffers_complete_sidecar_preserves_identity(
    warp_cpu_device: str,
) -> None:
    """A complete scratch sidecar returns its total buffer by identity."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device
    )
    initial_gas = gas.concentration.copy()
    _, returned, final_gas = _run_gpu_step(
        particles,
        gas,
        _make_vapor_pressure(1, 2),
        298.15,
        101325.0,
        0.1,
        warp_cpu_device,
        scratch_buffers=scratch,
        return_gas_state=True,
    )
    assert returned is scratch.total_mass_transfer
    assert scratch.work_mass_transfer.shape == particles.masses.shape
    assert np.all(np.isfinite(scratch.work_mass_transfer.numpy()))
    assert np.all(np.isfinite(scratch.total_mass_transfer.numpy()))
    assert np.all(scratch.total_mass_transfer.numpy() <= 0.0)
    npt.assert_array_equal(scratch.positive_mass_transfer_demand.numpy(), 0.0)
    assert np.all(scratch.negative_mass_transfer_release.numpy() >= 0.0)
    npt.assert_array_equal(scratch.positive_mass_transfer_scale.numpy(), 1.0)
    npt.assert_allclose(
        initial_gas - final_gas.concentration,
        np.sum(returned.numpy() * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        atol=1e-22,
    )


def test_validate_condensation_scratch_buffers_rejects_non_exact_sidecar_type(
    warp_cpu_device: str,
) -> None:
    """Scratch validation rejects non-dataclass candidates before allocation."""
    with pytest.raises(
        ValueError,
        match="scratch_buffers must be a CondensationScratchBuffers",
    ):
        validate_condensation_scratch_buffers(
            SimpleNamespace(),
            (1, 2, 2),
            wp.get_device(warp_cpu_device),
            "test_caller",
        )


@pytest.mark.parametrize(
    "fields",
    (
        ("work_mass_transfer",),
        ("total_mass_transfer",),
        ("dynamic_viscosity",),
        ("mean_free_path",),
        ("positive_mass_transfer_demand",),
        ("negative_mass_transfer_release",),
        ("positive_mass_transfer_scale",),
        ("work_mass_transfer", "dynamic_viscosity"),
    ),
)
def test_condensation_scratch_buffers_partial_sidecars_match_legacy(
    warp_cpu_device: str, fields: tuple[str, ...]
) -> None:
    """Partial sidecars preserve supplied identities and legacy results."""
    shape = (1, 2, 2)
    reference_particles = _make_particle_data(*shape)
    reference_gas = _make_gas_data(1, 2)
    reference_result, reference_transfer = _run_gpu_step(
        reference_particles,
        reference_gas,
        _make_vapor_pressure(1, 2),
        298.15,
        101325.0,
        0.1,
        warp_cpu_device,
    )
    particles = _make_particle_data(*shape)
    gas = _make_gas_data(1, 2)
    scratch = _make_condensation_scratch_buffers(
        shape, warp_cpu_device, fields=fields
    )
    result, returned = _run_gpu_step(
        particles,
        gas,
        _make_vapor_pressure(1, 2),
        298.15,
        101325.0,
        0.1,
        warp_cpu_device,
        scratch_buffers=scratch,
    )

    for field in fields:
        assert getattr(scratch, field) is not None
    expected_return = (
        scratch.total_mass_transfer
        if "total_mass_transfer" in fields
        else returned
    )
    assert returned is expected_return
    npt.assert_allclose(
        result.masses, reference_result.masses, rtol=1.0e-12, atol=1.0e-30
    )
    npt.assert_allclose(
        returned.numpy(), reference_transfer.numpy(), rtol=1.0e-12, atol=1.0e-30
    )


def test_condensation_scratch_property_sidecar_allows_legacy_transfer_buffer(
    warp_cpu_device: str,
) -> None:
    """Property-only scratch does not conflict with legacy transfer output."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    transfer = wp.full(
        particles.masses.shape,
        wp.float64(-1.0),
        dtype=wp.float64,
        device=warp_cpu_device,
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape,
        warp_cpu_device,
        fields=("dynamic_viscosity", "mean_free_path"),
    )
    _, returned = _run_gpu_step(
        particles,
        gas,
        _make_vapor_pressure(1, 2),
        298.15,
        101325.0,
        0.1,
        warp_cpu_device,
        mass_transfer=transfer,
        scratch_buffers=scratch,
    )
    assert returned is transfer
    assert scratch.dynamic_viscosity is not None
    assert scratch.mean_free_path is not None
    assert np.all(scratch.dynamic_viscosity.numpy() > 0.0)
    assert np.all(scratch.mean_free_path.numpy() > 0.0)


@pytest.mark.parametrize("field", ("work_mass_transfer", "total_mass_transfer"))
def test_condensation_scratch_transfer_overlap_is_atomic(
    warp_cpu_device: str, field: str
) -> None:
    """Legacy transfer output rejects either overlapping scratch transfer."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    transfer = wp.full(
        particles.masses.shape,
        wp.float64(-1.0),
        dtype=wp.float64,
        device=warp_cpu_device,
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device, fields=(field,)
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_transfer = transfer.numpy().copy()
    with pytest.raises(ValueError, match="conflicts with supplied scratch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            mass_transfer=transfer,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(transfer.numpy(), initial_transfer)


@pytest.mark.parametrize(
    "field",
    (
        "work_mass_transfer",
        "total_mass_transfer",
        "dynamic_viscosity",
        "mean_free_path",
        "positive_mass_transfer_demand",
        "negative_mass_transfer_release",
        "positive_mass_transfer_scale",
    ),
)
def test_condensation_scratch_buffer_wrong_dtype_is_atomic(
    warp_cpu_device: str, field: str
) -> None:
    """Non-float64 scratch fields fail before particle mutation."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device, fields=(field,)
    )
    shape = getattr(scratch, field).shape
    scratch = dataclasses.replace(
        scratch,
        **{
            field: wp.zeros(shape, dtype=wp.float32, device=warp_cpu_device),
        },
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    with pytest.raises(
        ValueError,
        match=f"scratch_buffers.{field} must use dtype",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)


@pytest.mark.parametrize(
    "field",
    (
        "work_mass_transfer",
        "total_mass_transfer",
        "dynamic_viscosity",
        "mean_free_path",
        "positive_mass_transfer_demand",
        "negative_mass_transfer_release",
        "positive_mass_transfer_scale",
    ),
)
def test_condensation_scratch_buffer_wrong_shape_is_atomic(
    warp_cpu_device: str, field: str
) -> None:
    """Each malformed supplied scratch field fails before a Warp launch."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device, fields=(field,)
    )
    malformed = wp.zeros((3,), dtype=wp.float64, device=warp_cpu_device)
    scratch = dataclasses.replace(scratch, **{field: malformed})
    initial_mass = gpu_particles.masses.numpy().copy()
    with pytest.raises(ValueError, match=f"scratch_buffers.{field} shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)


@pytest.mark.parametrize(
    "field",
    (
        "positive_mass_transfer_demand",
        "negative_mass_transfer_release",
        "positive_mass_transfer_scale",
    ),
)
def test_condensation_p2_scratch_non_warp_value_is_atomic(
    warp_cpu_device: str, field: str
) -> None:
    """P2 scratch fields reject non-Warp objects without changing state."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device
    )
    scratch = dataclasses.replace(
        scratch, **{field: np.full((1, 2), -1.0, dtype=np.float64)}
    )
    transfer = wp.full(
        particles.masses.shape,
        wp.float64(7.0),
        dtype=wp.float64,
        device=warp_cpu_device,
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_vapor = gpu_gas.vapor_pressure.numpy().copy()
    initial_transfer = transfer.numpy().copy()
    initial_scratch = {
        field_info.name: (
            value.copy()
            if isinstance(value, np.ndarray)
            else value.numpy().copy()
        )
        for field_info in dataclasses.fields(scratch)
        if (value := getattr(scratch, field_info.name)) is not None
    }

    with pytest.raises(ValueError, match=f"scratch_buffers.{field}"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            mass_transfer=transfer,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )

    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(gpu_gas.vapor_pressure.numpy(), initial_vapor)
    npt.assert_array_equal(transfer.numpy(), initial_transfer)
    for name, initial_value in initial_scratch.items():
        value = getattr(scratch, name)
        actual = value if isinstance(value, np.ndarray) else value.numpy()
        npt.assert_array_equal(actual, initial_value)


@pytest.mark.parametrize(
    "field",
    (
        "positive_mass_transfer_demand",
        "negative_mass_transfer_release",
        "positive_mass_transfer_scale",
    ),
)
def test_condensation_p2_scratch_wrong_device_is_atomic(field: str) -> None:
    """P2 scratch fields reject alternate-device storage before mutation."""
    runtime = _load_warp_runtime()
    if not runtime.cuda_available(runtime.wp):
        pytest.skip(runtime.CUDA_SKIP_REASON)
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device="cpu")
    gpu_gas = to_warp_gas_data(
        gas, device="cpu", vapor_pressure=_make_vapor_pressure(1, 2)
    )
    scratch = _make_condensation_scratch_buffers(particles.masses.shape, "cpu")
    scratch = dataclasses.replace(
        scratch,
        **{
            field: wp.full(
                (1, 2), wp.float64(-1.0), dtype=wp.float64, device="cuda"
            )
        },
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_vapor = gpu_gas.vapor_pressure.numpy().copy()

    with pytest.raises(ValueError, match=f"scratch_buffers.{field} device"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            scratch_buffers=scratch,
        )

    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(gpu_gas.vapor_pressure.numpy(), initial_vapor)


@pytest.mark.parametrize("invalid_value", (-1, 2))
def test_condensation_partitioning_nonbinary_mask_is_atomic(
    warp_cpu_device: str, invalid_value: int
) -> None:
    """A non-binary per-box mask fails before refreshing mutable state."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    gpu_gas.partitioning = wp.array(
        [[1, invalid_value]], dtype=wp.int32, device=warp_cpu_device
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_vapor = gpu_gas.vapor_pressure.numpy().copy()
    with pytest.raises(ValueError, match="gas.partitioning must contain"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
        )
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(gpu_gas.vapor_pressure.numpy(), initial_vapor)


@pytest.mark.parametrize(
    ("mask", "message"),
    (
        (np.array([1, 0], dtype=np.int32), "gas.partitioning"),
        (np.ones((1, 2), dtype=np.float64), "gas.partitioning"),
        (np.ones((1, 3), dtype=np.int32), "gas.partitioning"),
    ),
)
def test_condensation_partitioning_metadata_failure_is_atomic(
    warp_cpu_device: str,
    mask: np.ndarray,
    message: str,
) -> None:
    """Malformed partitioning metadata leaves every caller buffer untouched."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    if mask.dtype == np.int32:
        gpu_gas.partitioning = wp.array(
            mask, dtype=wp.int32, device=warp_cpu_device
        )
        gas_for_step = gpu_gas
    else:
        # Warp structs reject a wrong field dtype at assignment, so use the
        # entry point's duck-typed container contract to exercise preflight.
        gas_for_step = SimpleNamespace(
            molar_mass=gpu_gas.molar_mass,
            concentration=gpu_gas.concentration,
            vapor_pressure=gpu_gas.vapor_pressure,
            partitioning=wp.array(
                mask, dtype=wp.float64, device=warp_cpu_device
            ),
        )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape, warp_cpu_device
    )
    transfer = wp.full(
        particles.masses.shape,
        wp.float64(7.0),
        dtype=wp.float64,
        device=warp_cpu_device,
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    initial_vapor = gpu_gas.vapor_pressure.numpy().copy()
    initial_transfer = transfer.numpy().copy()
    initial_scratch = {
        field.name: getattr(scratch, field.name).numpy().copy()
        for field in dataclasses.fields(scratch)
    }

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gas_for_step,
            298.15,
            101325.0,
            0.1,
            mass_transfer=transfer,
            thermodynamics=_make_thermodynamics_config(gas_for_step),
            scratch_buffers=scratch,
        )

    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(gpu_gas.vapor_pressure.numpy(), initial_vapor)
    npt.assert_array_equal(transfer.numpy(), initial_transfer)
    for field in dataclasses.fields(scratch):
        npt.assert_array_equal(
            getattr(scratch, field.name).numpy(), initial_scratch[field.name]
        )


def test_condensation_partitioning_gates_disabled_species_and_inactive_slots(
    warp_cpu_device: str,
) -> None:
    """The raw and accumulated transfers are zeroed before application gates."""
    particles = _make_particle_data(2, 2, 2)
    particles.concentration[1, 1] = 0.0
    gas = _make_gas_data(2, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(2, 2)
    )
    gpu_gas.partitioning = wp.array(
        [[1, 0], [0, 1]], dtype=wp.int32, device=warp_cpu_device
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape,
        warp_cpu_device,
        fields=("work_mass_transfer", "total_mass_transfer"),
    )
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_gas = gpu_gas.concentration.numpy().copy()
    _, total = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        298.15,
        101325.0,
        0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        scratch_buffers=scratch,
    )
    work = scratch.work_mass_transfer.numpy()
    total_values = total.numpy()
    final_mass = gpu_particles.masses.numpy()
    for box_idx, species_idx in ((0, 1), (1, 0)):
        npt.assert_array_equal(work[box_idx, :, species_idx], 0.0)
        npt.assert_array_equal(total_values[box_idx, :, species_idx], 0.0)
        npt.assert_array_equal(
            final_mass[box_idx, :, species_idx],
            initial_mass[box_idx, :, species_idx],
        )
    npt.assert_array_equal(work[1, 1, :], 0.0)
    npt.assert_array_equal(total_values[1, 1, :], 0.0)
    npt.assert_array_equal(final_mass[1, 1, :], initial_mass[1, 1, :])
    npt.assert_allclose(
        initial_gas - gpu_gas.concentration.numpy(),
        np.sum(total_values * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        atol=1e-22,
    )
    npt.assert_array_equal(
        gpu_gas.concentration.numpy()[0, 1], initial_gas[0, 1]
    )
    npt.assert_array_equal(
        gpu_gas.concentration.numpy()[1, 0], initial_gas[1, 0]
    )


def test_condensation_partitioning_enabled_entry_matches_all_enabled_control(
    warp_cpu_device: str,
) -> None:
    """A binary enabled entry preserves the all-enabled raw proposal."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gated_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gated_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    control_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    control_gas = to_warp_gas_data(
        gas, device=warp_cpu_device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    gated_scratch = _make_condensation_scratch_buffers(
        particles.masses.shape,
        warp_cpu_device,
        fields=("work_mass_transfer",),
    )
    control_scratch = _make_condensation_scratch_buffers(
        particles.masses.shape,
        warp_cpu_device,
        fields=("work_mass_transfer",),
    )
    condensation_step_gpu(
        gated_particles,
        gated_gas,
        298.15,
        101325.0,
        0.1,
        thermodynamics=_make_thermodynamics_config(gated_gas),
        scratch_buffers=gated_scratch,
    )
    condensation_step_gpu(
        control_particles,
        control_gas,
        298.15,
        101325.0,
        0.1,
        thermodynamics=_make_thermodynamics_config(control_gas),
        scratch_buffers=control_scratch,
    )
    npt.assert_allclose(
        gated_scratch.work_mass_transfer.numpy(),
        control_scratch.work_mass_transfer.numpy(),
        rtol=1.0e-12,
        atol=1.0e-30,
    )


def _record_condensation_stiffness_trials(
    case: CondensationStiffnessCase,
    device: str,
    *,
    use_cache: bool = True,
) -> list[CondensationStiffnessTrialRecord]:
    """Return recorded-grid trials, reusing cached standard-path results."""
    if use_cache:
        return [
            _copy_condensation_stiffness_trial_record(record)
            for record in _cached_recorded_condensation_stiffness_trials(
                case.name,
                device,
            )
        ]
    return _record_condensation_stiffness_trials_uncached(case, device)


def _record_condensation_stiffness_trials_uncached(
    case: CondensationStiffnessCase,
    device: str,
    time_steps: tuple[float, ...] | None = None,
) -> list[CondensationStiffnessTrialRecord]:
    """Run the recorded timestep grid for one deterministic stiffness case.

    The helper rebuilds fresh inputs for each execution, while one complete
    caller-owned scratch sidecar is reused sequentially for the entire sweep.

    Args:
        case: Deterministic stiffness case to execute.
        device: Warp device name used for the sweep.

    Returns:
        Ordered recorded-grid trial records for the requested case.
    """
    recorded_timesteps = (
        time_steps or _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    )
    shape = (case.n_boxes, case.n_particles, case.n_species)
    scratch = _make_condensation_scratch_buffers(shape, device)
    previous_mass_transfer_values: np.ndarray | None = None
    environment_input_mode = "direct_warp_arrays"
    temperature: float | Any = case.temperature
    pressure: float | Any = case.pressure
    if case.n_boxes == 1:
        environment_input_mode = "scalar_inputs"
    else:
        temperature = wp.array(
            case.temperature_array(),
            dtype=wp.float64,
            device=device,
        )
        pressure = wp.array(
            case.pressure_array(),
            dtype=wp.float64,
            device=device,
        )

    records: list[CondensationStiffnessTrialRecord] = []
    for timestep_index, time_step in enumerate(recorded_timesteps):

        def run_once(
            trial_time_step: float,
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Any,
            tuple[bool, bool, bool, bool],
        ]:
            particles = case.build_particle_data()
            gas = case.build_gas_data()
            stale_vapor_pressure = case.build_vapor_pressure()
            gpu_particles = to_warp_particle_data(particles, device=device)
            gpu_gas = to_warp_gas_data(
                gas, device=device, vapor_pressure=stale_vapor_pressure
            )
            thermodynamics = _make_thermodynamics_config(gpu_gas)
            work_mass_transfer = scratch.work_mass_transfer
            total_mass_transfer = scratch.total_mass_transfer
            dynamic_viscosity = scratch.dynamic_viscosity
            mean_free_path = scratch.mean_free_path
            _, returned = condensation_step_gpu(
                gpu_particles,
                gpu_gas,
                temperature=temperature,
                pressure=pressure,
                time_step=trial_time_step,
                thermodynamics=thermodynamics,
                scratch_buffers=scratch,
            )
            return (
                particles.masses.copy(),
                gas.concentration.copy(),
                stale_vapor_pressure.copy(),
                from_warp_particle_data(gpu_particles, sync=True).masses.copy(),
                gpu_gas.concentration.numpy().copy(),
                gpu_gas.vapor_pressure.numpy().copy(),
                (returned, thermodynamics),
                (
                    scratch.work_mass_transfer is work_mass_transfer,
                    scratch.total_mass_transfer is total_mass_transfer,
                    scratch.dynamic_viscosity is dynamic_viscosity,
                    scratch.mean_free_path is mean_free_path,
                ),
            )

        (
            initial_masses,
            initial_gas_concentration,
            initial_vapor_pressure,
            final_masses,
            final_gas_concentration,
            final_vapor_pressure,
            first_runtime,
            scratch_identities,
        ) = run_once(time_step)
        returned_mass_transfer, thermodynamics = first_runtime
        mass_transfer_values = returned_mass_transfer.numpy().copy()
        (
            _,
            _,
            _,
            repeated_final_masses,
            _,
            repeated_final_vapor_pressure,
            repeated_runtime,
            repeated_scratch_identities,
        ) = run_once(time_step)
        repeated_mass_transfer = repeated_runtime[0].numpy().copy()
        temperature_array = (
            np.array([case.temperature], dtype=np.float64)
            if case.n_boxes == 1
            else case.temperature_array()
        )
        expected_vapor_pressure = _evaluate_vapor_pressure_config(
            thermodynamics, temperature_array
        )
        reference_particles = case.build_particle_data()
        reference_gas = case.build_gas_data()
        reference_final_masses, _, _ = _cpu_four_substep_oracle(
            reference_particles,
            reference_gas,
            initial_vapor_pressure,
            surface_tension=np.full(case.n_species, 0.072, dtype=np.float64),
            mass_accommodation=np.ones(case.n_species, dtype=np.float64),
            diffusion_coefficient_vapor=np.full(
                case.n_species, 2.0e-5, dtype=np.float64
            ),
            temperature=(
                case.temperature if case.n_boxes == 1 else temperature_array
            ),
            pressure=(
                case.pressure if case.n_boxes == 1 else case.pressure_array()
            ),
            time_step=time_step,
            thermodynamics=thermodynamics,
        )
        classification = _classify_particle_only_condensation_stiffness(
            case,
            initial_masses,
            final_masses,
            reference_gas,
            final_vapor_pressure,
            max_fractional_change=_RECORDED_STIFFNESS_THRESHOLD,
        )
        records.append(
            CondensationStiffnessTrialRecord(
                case_name=case.name,
                time_step=time_step,
                configured_time_step=recorded_timesteps[timestep_index],
                timestep_index=timestep_index,
                environment_input_mode=environment_input_mode,
                classification=classification,
                gas_unchanged=bool(
                    np.array_equal(
                        final_gas_concentration,
                        initial_gas_concentration,
                    )
                ),
                initial_gas_concentration=initial_gas_concentration,
                final_gas_concentration=final_gas_concentration,
                initial_vapor_pressure=initial_vapor_pressure,
                final_vapor_pressure=final_vapor_pressure,
                expected_vapor_pressure=expected_vapor_pressure,
                reuses_caller_mass_transfer_buffer=(
                    returned_mass_transfer is scratch.total_mass_transfer
                ),
                reuses_work_mass_transfer_buffer=scratch_identities[0],
                reuses_total_mass_transfer_buffer=scratch_identities[1],
                reuses_dynamic_viscosity_buffer=scratch_identities[2],
                reuses_mean_free_path_buffer=scratch_identities[3],
                reuses_all_scratch_buffers=all(scratch_identities)
                and all(repeated_scratch_identities),
                returned_total_is_scratch_total=(
                    returned_mass_transfer is scratch.total_mass_transfer
                ),
                mass_transfer_has_nonzero_values=bool(
                    np.any(mass_transfer_values != 0.0)
                ),
                mass_transfer_changed_from_previous_trial=(
                    previous_mass_transfer_values is not None
                    and not np.array_equal(
                        mass_transfer_values,
                        previous_mass_transfer_values,
                    )
                ),
                values_finite=_particle_values_are_finite(
                    final_masses, mass_transfer_values, final_vapor_pressure
                ),
                zero_mass_stable=_zero_mass_entries_remain_stable(
                    initial_masses, final_masses
                ),
                vapor_pressure_refreshed=bool(
                    np.array_equal(
                        final_vapor_pressure,
                        expected_vapor_pressure,
                    )
                    and not np.array_equal(
                        initial_vapor_pressure,
                        expected_vapor_pressure,
                    )
                ),
                final_masses=final_masses,
                initial_masses=initial_masses,
                mass_transfer_values=mass_transfer_values,
                reference_final_masses=reference_final_masses,
                repeated_final_masses=repeated_final_masses,
                repeated_mass_transfer_values=repeated_mass_transfer,
                repeated_final_vapor_pressure=repeated_final_vapor_pressure,
            )
        )
        previous_mass_transfer_values = mass_transfer_values

    return records


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("case_name", "time_step"),
    tuple(
        (case.name, time_step)
        for case in _make_condensation_stiffness_cases()
        for time_step in _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    ),
)
def test_condensation_production_stiffness_recorded_contract(
    warp_cpu_device: str,
    case_name: str,
    time_step: float,
) -> None:
    """Check the limited fixed-four production stiffness contract."""
    records = _record_condensation_stiffness_trials(
        _get_condensation_stiffness_case(case_name), warp_cpu_device
    )
    record = next(record for record in records if record.time_step == time_step)
    npt.assert_allclose(
        record.final_masses,
        record.reference_final_masses,
        rtol=5.0e-2,
        atol=0.0,
        err_msg=(
            "Recorded case-specific stiffness evidence is not a general "
            "parity or conservation tolerance."
        ),
    )
    positive_reference = (record.reference_final_masses > 0.0) & np.isfinite(
        record.reference_final_masses
    )
    if np.any(positive_reference):
        maximum_relative_error = np.max(
            np.abs(
                record.final_masses[positive_reference]
                - record.reference_final_masses[positive_reference]
            )
            / record.reference_final_masses[positive_reference]
        )
        assert maximum_relative_error <= 5.0e-2
    zero_reference = record.reference_final_masses == 0.0
    npt.assert_array_equal(
        record.final_masses[zero_reference],
        np.zeros(np.count_nonzero(zero_reference), dtype=np.float64),
    )
    assert record.environment_input_mode == (
        "scalar_inputs" if case_name != "droplet_like" else "direct_warp_arrays"
    )
    assert not record.gas_unchanged
    assert record.reuses_caller_mass_transfer_buffer
    assert record.reuses_work_mass_transfer_buffer
    assert record.reuses_total_mass_transfer_buffer
    assert record.reuses_dynamic_viscosity_buffer
    assert record.reuses_mean_free_path_buffer
    assert record.reuses_all_scratch_buffers
    assert record.returned_total_is_scratch_total
    assert record.mass_transfer_has_nonzero_values
    assert record.values_finite
    assert np.all(record.final_masses >= 0.0)
    assert record.zero_mass_stable
    assert record.vapor_pressure_refreshed
    particle_concentration = (
        _get_condensation_stiffness_case(case_name)
        .build_particle_data()
        .concentration
    )
    expected_final_gas = record.initial_gas_concentration - np.sum(
        record.mass_transfer_values * particle_concentration[:, :, None],
        axis=1,
    )
    npt.assert_allclose(
        record.final_gas_concentration,
        expected_final_gas,
        rtol=1.0e-12,
        atol=1.0e-30,
    )
    assert np.all(np.isfinite(record.final_gas_concentration))
    assert np.all(record.final_gas_concentration >= 0.0)
    npt.assert_array_equal(
        record.final_vapor_pressure,
        record.expected_vapor_pressure,
    )
    npt.assert_array_equal(record.final_masses, record.repeated_final_masses)
    npt.assert_array_equal(
        record.mass_transfer_values, record.repeated_mass_transfer_values
    )
    npt.assert_array_equal(
        record.final_vapor_pressure, record.repeated_final_vapor_pressure
    )


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_condensation_production_stiffness_cuda_slice(cuda_device: str) -> None:
    """Check one CUDA slice when the optional device is available."""
    case = _get_condensation_stiffness_case("droplet_like")
    record = _record_condensation_stiffness_trials_uncached(
        case, cuda_device, time_steps=(case.time_step,)
    )[0]
    npt.assert_allclose(
        record.final_masses,
        record.reference_final_masses,
        rtol=5.0e-2,
        atol=0.0,
        err_msg=(
            "CUDA stiffness evidence uses the recorded case-specific bound."
        ),
    )
    assert not record.gas_unchanged
    assert record.reuses_all_scratch_buffers
    assert record.reuses_work_mass_transfer_buffer
    assert record.reuses_total_mass_transfer_buffer
    assert record.reuses_dynamic_viscosity_buffer
    assert record.reuses_mean_free_path_buffer
    assert record.returned_total_is_scratch_total
    assert record.mass_transfer_has_nonzero_values
    assert record.values_finite
    assert np.all(record.final_masses >= 0.0)
    assert record.vapor_pressure_refreshed
    particle_concentration = case.build_particle_data().concentration
    expected_final_gas = record.initial_gas_concentration - np.sum(
        record.mass_transfer_values * particle_concentration[:, :, None],
        axis=1,
    )
    npt.assert_allclose(
        record.final_gas_concentration,
        expected_final_gas,
        rtol=1.0e-12,
        atol=1.0e-30,
    )
    assert np.all(np.isfinite(record.final_gas_concentration))
    assert np.all(record.final_gas_concentration >= 0.0)
    npt.assert_array_equal(
        record.final_vapor_pressure,
        record.expected_vapor_pressure,
    )
    npt.assert_array_equal(record.final_masses, record.repeated_final_masses)
    npt.assert_array_equal(
        record.final_vapor_pressure, record.repeated_final_vapor_pressure
    )


def test_condensation_step_gpu_signature_keeps_keyword_only_inputs() -> None:
    """Optional sidecars stay keyword-only after scalar positional inputs."""
    parameters = inspect.signature(_condensation_step_gpu).parameters
    parameter = parameters["environment"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["thermodynamics"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["activity_surface"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["scratch_buffers"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["latent_heat"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["energy_transfer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["thermal_work"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize(
    "sidecar_arguments",
    ["latent_heat", "thermal_work", "both"],
)
def test_zero_latent_or_thermal_work_sidecars_preserve_isothermal_condensation(
    device: str,
    n_species: int,
    sidecar_arguments: str,
) -> None:
    """Zero latent heat and deferred thermal work preserve transfer exactly."""
    particles = _make_particle_data(1, 2, n_species)
    gas = _make_gas_data(1, n_species)
    vapor_pressure = _make_vapor_pressure(1, n_species)
    sidecar_particles = to_warp_particle_data(particles, device=device)
    sidecar_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    legacy_particles = to_warp_particle_data(particles, device=device)
    legacy_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    latent_heat = wp.zeros(n_species, dtype=wp.float64, device=device)
    thermal_work = wp.array(
        np.linspace(0.0, 10.0, n_species), dtype=wp.float64, device=device
    )
    latent_snapshot = latent_heat.numpy().copy()
    thermal_snapshot = thermal_work.numpy().copy()
    sidecar_thermodynamics = _make_thermodynamics_config(sidecar_gas)
    legacy_thermodynamics = _make_thermodynamics_config(legacy_gas)
    sidecar_kwargs: dict[str, Any] = {}
    if sidecar_arguments in {"latent_heat", "both"}:
        sidecar_kwargs["latent_heat"] = latent_heat
    if sidecar_arguments in {"thermal_work", "both"}:
        sidecar_kwargs["thermal_work"] = thermal_work
    _, sidecar_transfer = _condensation_step_gpu(
        sidecar_particles,
        sidecar_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=sidecar_thermodynamics,
        **sidecar_kwargs,
    )
    _, legacy_transfer = _condensation_step_gpu(
        legacy_particles,
        legacy_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=legacy_thermodynamics,
    )
    npt.assert_array_equal(sidecar_transfer.numpy(), legacy_transfer.numpy())
    npt.assert_array_equal(
        sidecar_particles.masses.numpy(), legacy_particles.masses.numpy()
    )
    npt.assert_array_equal(
        sidecar_gas.concentration.numpy(), legacy_gas.concentration.numpy()
    )
    npt.assert_array_equal(
        sidecar_gas.vapor_pressure.numpy(), legacy_gas.vapor_pressure.numpy()
    )
    npt.assert_array_equal(latent_heat.numpy(), latent_snapshot)
    npt.assert_array_equal(thermal_work.numpy(), thermal_snapshot)


@pytest.mark.gpu_parity
def test_condensation_latent_heat_matches_four_substep_oracle(
    device: str,
) -> None:
    """Latent-corrected Warp transfers match the independent CPU oracle."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.1
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    vapor_pressure = _make_vapor_pressure(1, 2)
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.8], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.5e-5], dtype=np.float64)
    latent_values = np.array([2.2e6, 1.5e6], dtype=np.float64)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    expected_masses, expected_total, expected_work = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
        thermodynamics=thermodynamics,
        latent_heat=latent_values,
    )
    latent_heat = wp.array(latent_values, dtype=wp.float64, device=device)
    thermal_work = wp.full(2, 9.0, dtype=wp.float64, device=device)
    scratch = _make_condensation_scratch_buffers((1, 2, 2), device)
    latent_snapshot = latent_heat.numpy().copy()
    thermal_snapshot = thermal_work.numpy().copy()

    _, transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
        thermodynamics=thermodynamics,
        latent_heat=latent_heat,
        thermal_work=thermal_work,
        scratch_buffers=scratch,
    )

    assert transfer is scratch.total_mass_transfer
    scratch_arrays = (
        scratch.work_mass_transfer,
        scratch.total_mass_transfer,
        scratch.dynamic_viscosity,
        scratch.mean_free_path,
    )
    # Warp CPU and the NumPy oracle execute the same deterministic fp64 model.
    npt.assert_allclose(
        gpu_particles.masses.numpy(), expected_masses, rtol=1e-12, atol=0.0
    )
    npt.assert_allclose(transfer.numpy(), expected_total, rtol=1e-12, atol=0.0)
    npt.assert_allclose(
        scratch.work_mass_transfer.numpy(), expected_work, rtol=1e-12, atol=0.0
    )
    assert np.all(np.isfinite(gpu_particles.masses.numpy()))
    assert np.all(gpu_particles.masses.numpy() >= 0.0)
    npt.assert_array_equal(latent_heat.numpy(), latent_snapshot)
    npt.assert_array_equal(thermal_work.numpy(), thermal_snapshot)

    repeated_particles = to_warp_particle_data(particles, device=device)
    repeated_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    _, repeated_transfer = _condensation_step_gpu(
        repeated_particles,
        repeated_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
        thermodynamics=_make_thermodynamics_config(repeated_gas),
        latent_heat=latent_heat,
        thermal_work=thermal_work,
        scratch_buffers=scratch,
    )
    assert repeated_transfer is scratch.total_mass_transfer
    assert scratch.work_mass_transfer is scratch_arrays[0]
    assert scratch.total_mass_transfer is scratch_arrays[1]
    assert scratch.dynamic_viscosity is scratch_arrays[2]
    assert scratch.mean_free_path is scratch_arrays[3]
    npt.assert_allclose(
        repeated_particles.masses.numpy(), expected_masses, rtol=1e-12, atol=0.0
    )
    npt.assert_allclose(
        repeated_transfer.numpy(), expected_total, rtol=1e-12, atol=0.0
    )


def _assert_composed_condensation_route(
    device: str,
    *,
    use_environment: bool,
) -> None:
    """Assert one fresh composed latent/activity route against its oracle."""
    n_boxes, n_particles, n_species = 1, 2, 2
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.1
    particles = _make_particle_data(n_boxes, n_particles, n_species)
    gas = _make_gas_data(n_boxes, n_species)
    vapor_pressure = _make_vapor_pressure(n_boxes, n_species)
    initial_gas_concentration = gas.concentration.copy()
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.8], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.5e-5], dtype=np.float64)
    latent_values = np.array([2.2e6, 0.0], dtype=np.float64)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    thermodynamics = _make_mixed_thermodynamics_config(gpu_gas)
    activity_surface = _make_activity_surface_config(
        gpu_gas,
        int(ACTIVITY_MODE_KAPPA),
        int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
    )
    scratch = _make_condensation_scratch_buffers(particles.masses.shape, device)
    assert scratch.total_mass_transfer is not None
    transfer_identity = scratch.total_mass_transfer
    energy_transfer = wp.full(
        (n_boxes, n_species), wp.float64(17.0), dtype=wp.float64, device=device
    )
    energy_identity = energy_transfer
    expected_masses, expected_total, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
        thermodynamics=thermodynamics,
        activity_mode=ACTIVITY_MODE_KAPPA,
        surface_tension_mode=SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED,
        kappas=np.linspace(0.0, 0.5, n_species, dtype=np.float64),
        latent_heat=latent_values,
    )
    step_kwargs: dict[str, Any] = {
        "time_step": time_step,
        "surface_tension": wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        "mass_accommodation": wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        "diffusion_coefficient_vapor": wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
        "thermodynamics": thermodynamics,
        "activity_surface": activity_surface,
        "scratch_buffers": scratch,
        "latent_heat": wp.array(latent_values, dtype=wp.float64, device=device),
        "energy_transfer": energy_transfer,
    }
    if use_environment:
        environment = to_warp_environment_data(
            _make_environment_data(n_boxes, n_species, temperature, pressure),
            device=device,
        )
        step_kwargs.update(
            temperature=None,
            pressure=None,
            environment=environment,
        )
    else:
        step_kwargs.update(temperature=temperature, pressure=pressure)

    result = _condensation_step_gpu(gpu_particles, gpu_gas, **step_kwargs)

    assert len(result) == 2
    assert result[0] is gpu_particles
    assert result[1] is transfer_identity
    assert energy_transfer is energy_identity
    npt.assert_allclose(
        gpu_particles.masses.numpy(), expected_masses, rtol=1e-12, atol=0.0
    )
    npt.assert_allclose(result[1].numpy(), expected_total, rtol=1e-12, atol=0.0)
    expected_energy = expected_total.sum(axis=1) * latent_values[None, :]
    npt.assert_allclose(
        energy_transfer.numpy(), expected_energy, rtol=1e-12, atol=0.0
    )
    npt.assert_array_equal(energy_transfer.numpy()[:, 1], 0.0)
    npt.assert_allclose(
        initial_gas_concentration - gpu_gas.concentration.numpy(),
        np.sum(expected_total * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        # Subtracting a femtogram-scale delta from the 1e-6 gas fixture has
        # one fp64 rounding unit beyond the transfer accumulator.
        atol=3.0e-23,
    )
    assert np.all(np.isfinite(gpu_particles.masses.numpy()))
    assert np.all(gpu_particles.masses.numpy() >= 0.0)


@pytest.mark.gpu_parity
def test_condensation_composed_scalar_route_matches_four_substep_oracle(
    warp_cpu_device: str,
) -> None:
    """The scalar route preserves composed latent/activity diagnostics."""
    _assert_composed_condensation_route(warp_cpu_device, use_environment=False)


@pytest.mark.gpu_parity
def test_condensation_composed_environment_route_matches_four_substep_oracle(
    warp_cpu_device: str,
) -> None:
    """The explicit-environment route preserves composed diagnostics."""
    _assert_composed_condensation_route(warp_cpu_device, use_environment=True)


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_condensation_composed_cuda_matches_four_substep_oracle(
    cuda_device: str,
) -> None:
    """Optional CUDA matches the composed four-substep CPU oracle."""
    _assert_composed_condensation_route(cuda_device, use_environment=False)


@pytest.mark.gpu_parity
def test_condensation_step_gpu_energy_transfer_reuses_and_overwrites_output(
    device: str,
) -> None:
    """Energy output is caller-owned, overwritten, and not a return value."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    vapor_pressure = _make_vapor_pressure(1, 2)
    latent_values = np.array([2.2e6, 1.5e6], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    energy_transfer = wp.full((1, 2), 17.0, dtype=wp.float64, device=device)
    output_identity = energy_transfer
    result = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        energy_transfer=energy_transfer,
    )
    assert len(result) == 2
    assert result[0] is gpu_particles
    assert energy_transfer is output_identity
    assert np.any(energy_transfer.numpy() != 17.0)
    expected_energy = result[1].numpy().sum(axis=1) * latent_values[None, :]
    npt.assert_allclose(
        energy_transfer.numpy(), expected_energy, rtol=1e-12, atol=1e-18
    )

    energy_transfer = wp.array(
        [[np.nan, np.inf]], dtype=wp.float64, device=device
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    _, total = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        energy_transfer=energy_transfer,
    )
    assert np.all(np.isfinite(energy_transfer.numpy()))
    npt.assert_allclose(
        energy_transfer.numpy(),
        total.numpy().sum(axis=1) * latent_values[None, :],
        rtol=1e-12,
        atol=1e-18,
    )


def test_condensation_step_gpu_without_energy_transfer_skips_energy_kernels(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting energy output preserves the legacy launch profile."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    result = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
    )

    assert len(result) == 2
    assert (
        condensation_module._clear_energy_transfer_kernel
        not in launched_kernels
    )
    assert (
        condensation_module._accumulate_energy_transfer_kernel
        not in launched_kernels
    )


def test_condensation_step_gpu_nonzero_execution_launches_four_p2_sequences(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each public substep finalizes, accumulates, and couples P2 transfer."""
    particles = _make_particle_data(1, 1, 1)
    gas = _make_gas_data(1, 1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 1)
    )
    scratch = _make_condensation_scratch_buffers(
        (1, 1, 1),
        device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    p2_sidecars = tuple(
        getattr(scratch, name)
        for name in (
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        )
    )
    initial_gas = gpu_gas.concentration.numpy().copy()
    launched_kernels: list[Any] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launched_kernels.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    _, total = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        scratch_buffers=scratch,
    )

    p2_kernels = (
        condensation_module._bound_evaporation_candidate_kernel,
        condensation_module._reduce_inventory_candidates_kernel,
        condensation_module._scale_inventory_uptake_kernel,
        condensation_module._finalize_and_apply_inventory_transfer_kernel,
        condensation_module._accumulate_finalized_mass_transfer_kernel,
        condensation_module._couple_finalized_transfer_to_gas_kernel,
    )
    for kernel in p2_kernels:
        assert launched_kernels.count(kernel) == 4
    npt.assert_allclose(
        initial_gas - gpu_gas.concentration.numpy(),
        np.sum(total.numpy() * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        atol=1e-22,
    )
    assert all(np.all(np.isfinite(sidecar.numpy())) for sidecar in p2_sidecars)


@pytest.mark.gpu_parity
def test_condensation_step_gpu_energy_transfer_aggregates_by_box_and_species(
    device: str,
) -> None:
    """Energy reduction isolates each box/species while summing particles."""
    particles = _make_particle_data(2, 2, 2)
    gas = _make_gas_data(2, 2)
    gas.concentration[:] = np.array([[1.0e-2, 2.0e-2], [3.0e-2, 4.0e-2]])
    vapor_pressure = _make_vapor_pressure(2, 2)
    latent_values = np.array([1.1e6, 2.3e6], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    energy_transfer = wp.zeros((2, 2), dtype=wp.float64, device=device)
    _, total = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        energy_transfer=energy_transfer,
    )
    expected = total.numpy().sum(axis=1) * latent_values[None, :]
    npt.assert_allclose(
        energy_transfer.numpy(), expected, rtol=1e-12, atol=1e-18
    )
    assert not np.allclose(energy_transfer.numpy(), expected.sum(axis=0))


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("gas_concentration", "time_step"),
    [
        (1.0e-2, 0.1),
        (1.0e-6, 0.1),
        (1.0e-2, 0.0),
    ],
    ids=("condensation", "evaporation", "zero-transfer"),
)
def test_condensation_energy_transfer_matches_four_substep_oracle(
    device: str,
    gas_concentration: float,
    time_step: float,
) -> None:
    """Energy output uses the signed bounded four-substep transfer."""
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gas.concentration.fill(gas_concentration)
    vapor_pressure = _make_vapor_pressure(1, 2)
    latent_values = np.array([1.1e6, 2.3e6], dtype=np.float64)
    expected_final, expected_total, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension=np.full(2, 0.072, dtype=np.float64),
        mass_accommodation=np.full(2, 1.0, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(2, 2.0e-5, dtype=np.float64),
        temperature=298.15,
        pressure=101325.0,
        time_step=time_step,
        latent_heat=latent_values,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    energy_transfer = wp.full(
        (1, 2), wp.float64(13.0), dtype=wp.float64, device=device
    )

    _, total = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=time_step,
        thermodynamics=_make_thermodynamics_config(gpu_gas),
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        energy_transfer=energy_transfer,
    )

    expected_energy = expected_total.sum(axis=1) * latent_values[None, :]
    npt.assert_allclose(
        gpu_particles.masses.numpy(), expected_final, rtol=1e-12, atol=1e-18
    )
    npt.assert_allclose(total.numpy(), expected_total, rtol=1e-12, atol=1e-18)
    npt.assert_allclose(
        energy_transfer.numpy(), expected_energy, rtol=1e-12, atol=1e-18
    )


@pytest.mark.parametrize(
    ("energy_transfer", "match"),
    [
        (None, "requires latent_heat"),
        ([0.0, 0.0], "must be a Warp array"),
        (np.zeros((1, 2), dtype=np.float32), "must use dtype"),
        (np.zeros((2,), dtype=np.float64), "shape"),
        (np.zeros((1, 1), dtype=np.float64), "shape"),
    ],
    ids=["missing-latent", "non-warp", "float32", "rank", "shape"],
)
def test_condensation_energy_transfer_preflight_is_atomic(
    device: str,
    energy_transfer: Any,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid energy output metadata fails before allocations or mutation."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    if energy_transfer is None:
        energy_transfer = wp.full((1, 2), 19.0, dtype=wp.float64, device=device)
    if isinstance(energy_transfer, np.ndarray):
        energy_transfer = wp.array(
            energy_transfer,
            dtype=wp.float32
            if energy_transfer.dtype == np.float32
            else wp.float64,
            device=device,
        )
    latent_heat = wp.ones(2, dtype=wp.float64, device=device)
    scratch = _make_condensation_scratch_buffers((1, 1, 2), device)
    assert scratch.work_mass_transfer is not None
    assert scratch.total_mass_transfer is not None
    assert scratch.dynamic_viscosity is not None
    assert scratch.mean_free_path is not None
    output_arrays = (
        (energy_transfer,) if hasattr(energy_transfer, "numpy") else ()
    )
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.concentration,
        gpu_gas.vapor_pressure,
        *output_arrays,
        scratch.work_mass_transfer,
        scratch.total_mass_transfer,
        scratch.dynamic_viscosity,
        scratch.mean_free_path,
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)

    def fail_after_preflight(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid energy output passed preflight")

    monkeypatch.setattr(
        condensation_module, "_ensure_environment_arrays", fail_after_preflight
    )
    with pytest.raises(ValueError, match=match):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            latent_heat=None
            if match == "requires latent_heat"
            else latent_heat,
            energy_transfer=energy_transfer,
            scratch_buffers=scratch,
        )
    for array, expected in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), expected)


@pytest.mark.parametrize(
    "alias_name",
    ("concentration", "vapor_pressure"),
)
def test_condensation_energy_transfer_alias_preflight_is_atomic(
    device: str,
    alias_name: str,
) -> None:
    """Aliased energy output fails before it can mutate mutable gas state."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    energy_transfer = getattr(gpu_gas, alias_name)
    latent_heat = wp.ones(2, dtype=wp.float64, device=device)
    scratch = _make_condensation_scratch_buffers((1, 1, 2), device)
    assert scratch.work_mass_transfer is not None
    assert scratch.total_mass_transfer is not None
    assert scratch.dynamic_viscosity is not None
    assert scratch.mean_free_path is not None
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.concentration,
        gpu_gas.vapor_pressure,
        scratch.work_mass_transfer,
        scratch.total_mass_transfer,
        scratch.dynamic_viscosity,
        scratch.mean_free_path,
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)

    with pytest.raises(ValueError, match="must not overlap mutable"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            latent_heat=latent_heat,
            energy_transfer=energy_transfer,
            scratch_buffers=scratch,
        )

    assert energy_transfer is getattr(gpu_gas, alias_name)
    for array, expected in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -1.0))
@pytest.mark.parametrize(
    "sidecar_name",
    (
        "latent_heat",
        "thermal_work",
        "surface_tension",
        "mass_accommodation",
        "diffusion_coefficient_vapor",
    ),
)
def test_condensation_cuda_invalid_species_sidecar_is_atomic(
    cuda_device: str,
    invalid_value: float,
    sidecar_name: str,
) -> None:
    """CUDA rejects invalid sidecars before caller-owned state is changed."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=cuda_device)
    gpu_gas = to_warp_gas_data(
        gas, device=cuda_device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    sidecars = {
        "latent_heat": wp.ones(2, dtype=wp.float64, device=cuda_device),
        "thermal_work": wp.ones(2, dtype=wp.float64, device=cuda_device),
        "surface_tension": wp.full(
            2, 0.072, dtype=wp.float64, device=cuda_device
        ),
        "mass_accommodation": wp.ones(2, dtype=wp.float64, device=cuda_device),
        "diffusion_coefficient_vapor": wp.full(
            2, 2.0e-5, dtype=wp.float64, device=cuda_device
        ),
    }
    sidecars[sidecar_name] = wp.array(
        [invalid_value, 1.0], dtype=wp.float64, device=cuda_device
    )
    energy_transfer = wp.full(
        (1, 2), 19.0, dtype=wp.float64, device=cuda_device
    )
    mass_transfer = wp.full(
        (1, 1, 2), 13.0, dtype=wp.float64, device=cuda_device
    )
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.vapor_pressure,
        energy_transfer,
        mass_transfer,
        *sidecars.values(),
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)

    with pytest.raises(ValueError, match=f"{sidecar_name} must be finite"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            energy_transfer=energy_transfer,
            mass_transfer=mass_transfer,
            **sidecars,
        )

    for array, expected in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), expected)


def test_condensation_energy_transfer_rejects_thermodynamic_parameters_alias(
    device: str,
) -> None:
    """Energy output cannot clear thermodynamic parameters before refresh."""
    particles = _make_particle_data(4, 1, 4)
    gas = _make_gas_data(4, 4)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(4, 4)
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    latent_heat = wp.ones(4, dtype=wp.float64, device=device)
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.vapor_pressure,
        thermodynamics.parameters,
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)

    with pytest.raises(ValueError, match="must not overlap mutable"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=thermodynamics,
            latent_heat=latent_heat,
            energy_transfer=thermodynamics.parameters,
        )
    for array, expected in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), expected)


def test_condensation_energy_transfer_rejects_partial_storage_overlap(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct Warp arrays with overlapping storage are rejected."""
    energy_transfer = wp.zeros((1, 2), dtype=wp.float64, device=device)
    mutable_state = wp.zeros((1, 2), dtype=wp.float64, device=device)

    def overlapping_ranges(array: Any) -> tuple[int, int]:
        """Return deterministic partially overlapping byte ranges."""
        if array is energy_transfer:
            return 0, 16
        return 8, 24

    monkeypatch.setattr(
        condensation_module,
        "_warp_array_memory_range",
        overlapping_ranges,
    )

    with pytest.raises(ValueError, match="must not overlap mutable"):
        condensation_module._validate_energy_transfer_ownership(
            energy_transfer,
            (mutable_state,),
        )


@pytest.mark.cuda
def test_condensation_energy_transfer_device_mismatch_is_atomic(
    warp_cpu_device: str,
    cuda_device: str,
) -> None:
    """A cross-device energy output fails before caller state is changed."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=_make_vapor_pressure(1, 2),
    )
    latent_heat = wp.ones(2, dtype=wp.float64, device=warp_cpu_device)
    energy_transfer = wp.full(
        (1, 2), wp.float64(19.0), dtype=wp.float64, device=cuda_device
    )
    state_arrays = (
        gpu_particles.masses,
        gpu_gas.vapor_pressure,
        energy_transfer,
    )
    snapshots = tuple(array.numpy().copy() for array in state_arrays)

    with pytest.raises(ValueError, match="energy_transfer device"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            latent_heat=latent_heat,
            energy_transfer=energy_transfer,
        )

    for array, expected in zip(state_arrays, snapshots, strict=True):
        npt.assert_array_equal(array.numpy(), expected)


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_condensation_energy_transfer_cuda_matches_box_species_oracle(
    cuda_device: str,
) -> None:
    """Optional CUDA energy reduction matches the multi-box CPU-side oracle."""
    test_condensation_step_gpu_energy_transfer_aggregates_by_box_and_species(
        cuda_device
    )


@pytest.mark.gpu_parity
def test_condensation_mixed_latent_heat_with_activity_matches_oracle(
    device: str,
) -> None:
    """Mixed latent species use shared activity/Kelvin surface pressure."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.1
    particles = _make_particle_data(1, 2, 2)
    gas = _make_gas_data(1, 2)
    gas.concentration[:] = np.array([[1.0e-2, 2.0e-2]])
    vapor_pressure = _make_vapor_pressure(1, 2)
    surface_tension = np.array([0.04, 0.09], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.8], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.5e-5], dtype=np.float64)
    latent_values = np.array([0.0, 2.2e6], dtype=np.float64)
    kappas = np.array([0.0, 0.5], dtype=np.float64)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    expected_masses, expected_total, expected_work = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
        thermodynamics=thermodynamics,
        activity_mode=int(ACTIVITY_MODE_KAPPA),
        surface_tension_mode=int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        kappas=kappas,
        latent_heat=latent_values,
    )
    _, _, isothermal_work = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
        thermodynamics=thermodynamics,
        activity_mode=int(ACTIVITY_MODE_KAPPA),
        surface_tension_mode=int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        kappas=kappas,
    )
    activity_surface = CondensationActivitySurfaceConfig(
        activity_mode=int(ACTIVITY_MODE_KAPPA),
        surface_tension_mode=int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        water_species_index=0,
        kappas=wp.array(kappas, dtype=wp.float64, device=device),
        molar_mass_reference=wp.array(
            gas.molar_mass, dtype=wp.float64, device=device
        ),
    )
    scratch = _make_condensation_scratch_buffers((1, 2, 2), device)

    _, total = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
        thermodynamics=thermodynamics,
        activity_surface=activity_surface,
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        scratch_buffers=scratch,
    )

    npt.assert_allclose(
        gpu_particles.masses.numpy(), expected_masses, rtol=1e-12, atol=0.0
    )
    npt.assert_allclose(total.numpy(), expected_total, rtol=1e-12, atol=0.0)
    npt.assert_allclose(
        scratch.work_mass_transfer.numpy(), expected_work, rtol=1e-12, atol=0.0
    )
    assert np.all(gpu_particles.masses.numpy() >= 0.0)
    assert np.all(np.isfinite(gpu_particles.masses.numpy()))
    assert np.all(expected_work[:, :, 1] < isothermal_work[:, :, 1])
    assert np.all(
        scratch.work_mass_transfer.numpy()[:, :, 1] < isothermal_work[:, :, 1]
    )


def test_condensation_isolated_zero_latent_species_matches_baseline(
    device: str,
) -> None:
    """A zero-latent branch stays exactly isothermal when isolated."""
    particles = _make_particle_data(1, 1, 2)
    particles.masses[:, :, 1] = 0.0
    gas = _make_gas_data(1, 2)
    gas.concentration[:, 1] = 0.0
    vapor_pressure = _make_vapor_pressure(1, 2)
    baseline_particles = to_warp_particle_data(particles, device=device)
    baseline_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    latent_particles = to_warp_particle_data(particles, device=device)
    latent_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )

    _, baseline_total = _condensation_step_gpu(
        baseline_particles,
        baseline_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(baseline_gas),
    )
    _, latent_total = _condensation_step_gpu(
        latent_particles,
        latent_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=_make_thermodynamics_config(latent_gas),
        latent_heat=wp.array([0.0, 2.2e6], dtype=wp.float64, device=device),
    )

    npt.assert_array_equal(
        latent_particles.masses.numpy()[:, :, 0],
        baseline_particles.masses.numpy()[:, :, 0],
    )
    npt.assert_array_equal(
        latent_total.numpy()[:, :, 0], baseline_total.numpy()[:, :, 0]
    )


@pytest.mark.parametrize("sidecar_name", ["latent_heat", "thermal_work"])
@pytest.mark.parametrize(
    "values",
    [
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([-np.inf, 1.0]),
        np.array([-1.0, 1.0]),
        np.ones((1, 2)),
        np.ones(1),
        np.array([1.0, 2.0], dtype=np.float32),
        [1.0, 2.0],
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "negative",
        "rank",
        "length",
        "float32",
        "non-warp",
    ],
)
def test_invalid_latent_sidecar_fails_before_allocation_or_mutation(
    device: str,
    sidecar_name: str,
    values: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid latent metadata fails atomically before fallback work begins."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=np.full((1, 2), 17.0)
    )
    if isinstance(values, list):
        candidate = values
    else:
        dtype = wp.float32 if values.dtype == np.float32 else wp.float64
        candidate = wp.array(values, dtype=dtype, device=device)
    counterpart = wp.array([0.0, 1.0], dtype=wp.float64, device=device)
    transfer = wp.full((1, 1, 2), 13.0, dtype=wp.float64, device=device)
    scratch = _make_condensation_scratch_buffers(
        (1, 1, 2),
        device,
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    snapshots = tuple(
        array.numpy().copy()
        for array in (
            gpu_particles.masses,
            gpu_gas.concentration,
            gpu_gas.vapor_pressure,
            counterpart,
            transfer,
            scratch.work_mass_transfer,
            scratch.total_mass_transfer,
            scratch.dynamic_viscosity,
            scratch.mean_free_path,
        )
    )
    kwargs = {sidecar_name: candidate}
    other_name = (
        "thermal_work" if sidecar_name == "latent_heat" else "latent_heat"
    )
    kwargs[other_name] = counterpart

    def fail_before_sidecar_preflight(*args: Any, **kwargs: Any) -> None:
        """Fail if invalid sidecars reach allocation, normalization, or launch.

        Raises:
            AssertionError: Always, if atomic sidecar preflight is bypassed.
        """
        raise AssertionError("invalid sidecar passed atomic preflight")

    monkeypatch.setattr(
        condensation_module,
        "_ensure_environment_arrays",
        fail_before_sidecar_preflight,
    )
    monkeypatch.setattr(
        condensation_module.wp,
        "zeros",
        fail_before_sidecar_preflight,
    )
    monkeypatch.setattr(
        condensation_module.wp,
        "full",
        fail_before_sidecar_preflight,
    )
    monkeypatch.setattr(
        condensation_module.wp,
        "launch",
        fail_before_sidecar_preflight,
    )
    with pytest.raises(ValueError, match=sidecar_name):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            mass_transfer=transfer,
            thermodynamics=thermodynamics,
            scratch_buffers=scratch,
            **kwargs,
        )
    for array, expected in zip(
        (
            gpu_particles.masses,
            gpu_gas.concentration,
            gpu_gas.vapor_pressure,
            counterpart,
            transfer,
            scratch.work_mass_transfer,
            scratch.total_mass_transfer,
            scratch.dynamic_viscosity,
            scratch.mean_free_path,
        ),
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(array.numpy(), expected)


@pytest.mark.parametrize("invalid_time_step", [True, -0.1, np.nan, np.inf])
def test_invalid_time_step_fails_before_mutating_caller_outputs(
    device: str,
    invalid_time_step: float | bool,
) -> None:
    """Invalid timesteps leave supplied output and physical state unchanged."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=np.full((1, 2), 17.0)
    )
    transfer = wp.full((1, 1, 2), 13.0, dtype=wp.float64, device=device)
    masses_before = gpu_particles.masses.numpy().copy()
    vapor_pressure_before = gpu_gas.vapor_pressure.numpy().copy()
    transfer_before = transfer.numpy().copy()

    with pytest.raises(ValueError, match="time_step"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=invalid_time_step,
            mass_transfer=transfer,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
        )

    npt.assert_array_equal(gpu_particles.masses.numpy(), masses_before)
    npt.assert_array_equal(
        gpu_gas.vapor_pressure.numpy(), vapor_pressure_before
    )
    npt.assert_array_equal(transfer.numpy(), transfer_before)


@pytest.mark.cuda
@pytest.mark.parametrize("sidecar_name", ["latent_heat", "thermal_work"])
def test_latent_sidecar_device_mismatch_raises_value_error(
    warp_cpu_device: str,
    cuda_device: str,
    sidecar_name: str,
) -> None:
    """Reject a latent sidecar that is not on the particle device."""
    particles = _make_particle_data(1, 1, 2)
    gas = _make_gas_data(1, 2)
    gpu_particles = to_warp_particle_data(particles, device=warp_cpu_device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=warp_cpu_device,
        vapor_pressure=_make_vapor_pressure(1, 2),
    )
    sidecar = wp.array([0.0, 1.0], dtype=wp.float64, device=cuda_device)
    counterpart = wp.array([0.0, 1.0], dtype=wp.float64, device=warp_cpu_device)
    kwargs = {sidecar_name: sidecar}
    other_name = (
        "thermal_work" if sidecar_name == "latent_heat" else "latent_heat"
    )
    kwargs[other_name] = counterpart

    with pytest.raises(ValueError, match=sidecar_name):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            **kwargs,
        )


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("activity_mode", "surface_mode"),
    [
        (int(ACTIVITY_MODE_IDEAL), int(SURFACE_TENSION_MODE_STATIC)),
        (
            int(ACTIVITY_MODE_IDEAL),
            int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        ),
        (int(ACTIVITY_MODE_KAPPA), int(SURFACE_TENSION_MODE_STATIC)),
        (
            int(ACTIVITY_MODE_KAPPA),
            int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        ),
    ],
)
def test_condensation_activity_surface_modes_are_finite_and_water_only(
    device: str,
    activity_mode: int,
    surface_mode: int,
) -> None:
    """Configured modes affect water only and retain finite clamped masses."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    tension = wp.array([0.04, 0.09], dtype=wp.float64, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    activity_surface = _make_activity_surface_config(
        gpu_gas, activity_mode, surface_mode
    )
    _, configured_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        surface_tension=tension,
        activity_surface=activity_surface,
    )
    configured = configured_transfer.numpy()
    final_masses = from_warp_particle_data(gpu_particles, sync=True).masses
    assert np.all(np.isfinite(configured))
    assert np.all(final_masses >= 0.0)

    # The non-water transfer is exactly the static legacy result when the
    # surface mode is static, proving activity never applies to that vapor.
    if surface_mode == int(SURFACE_TENSION_MODE_STATIC):
        legacy_particles = to_warp_particle_data(particles, device=device)
        legacy_gas = to_warp_gas_data(
            gas, device=device, vapor_pressure=vapor_pressure
        )
        _, legacy_transfer = condensation_step_gpu(
            legacy_particles,
            legacy_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            surface_tension=wp.array(
                [0.04, 0.09], dtype=wp.float64, device=device
            ),
        )
        npt.assert_allclose(
            configured[:, :, 1],
            legacy_transfer.numpy()[:, :, 1],
            rtol=1e-12,
            atol=0.0,
        )


def test_condensation_activity_surface_validation_is_frozen_and_atomic(
    device: str,
) -> None:
    """Malformed sidecars fail before launch and valid bindings are frozen."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=_make_vapor_pressure(1, 2)
    )
    sidecar = _make_activity_surface_config(
        gpu_gas, int(ACTIVITY_MODE_IDEAL), int(SURFACE_TENSION_MODE_STATIC)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, sidecar).activity_mode = 1
    with pytest.raises(ValueError, match="activity_surface"):
        validate_condensation_activity_surface_config(
            object(),
            2,
            gpu_particles.masses.device,
            gpu_gas.molar_mass,
            "test",
        )


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("activity_mode", "surface_mode"),
    [
        (int(ACTIVITY_MODE_IDEAL), int(SURFACE_TENSION_MODE_STATIC)),
        (
            int(ACTIVITY_MODE_IDEAL),
            int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        ),
        (int(ACTIVITY_MODE_KAPPA), int(SURFACE_TENSION_MODE_STATIC)),
        (
            int(ACTIVITY_MODE_KAPPA),
            int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        ),
    ],
)
def test_condensation_activity_surface_matches_independent_reference(
    device: str,
    activity_mode: int,
    surface_mode: int,
) -> None:
    """Each activity/tension mode matches the independent NumPy transfer."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=2)
    particles.masses[0, 0, :] = [1.0e-18, 9.0e-18]
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    tension_values = np.array([0.035, 0.095], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    sidecar = _make_activity_surface_config(
        gpu_gas, activity_mode, surface_mode
    )
    tension = wp.array(tension_values, dtype=wp.float64, device=device)

    _, transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        surface_tension=tension,
        thermodynamics=thermodynamics,
        activity_surface=sidecar,
    )
    _, expected, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension=tension_values,
        mass_accommodation=np.ones(2, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(2, 2.0e-5, dtype=np.float64),
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=thermodynamics,
        activity_mode=activity_mode,
        surface_tension_mode=surface_mode,
        water_species_index=0,
        kappas=sidecar.kappas.numpy(),
    )

    npt.assert_allclose(transfer.numpy(), expected, rtol=1.0e-10, atol=0.0)


@pytest.mark.gpu_parity
def test_condensation_activity_surface_multibox_uses_current_composition(
    device: str,
) -> None:
    """Weighted kappa transfer uses each box's temperature and composition."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=2)
    particles.masses[:, 0, :] = [[1.0e-18, 9.0e-18], [8.0e-18, 2.0e-18]]
    gas = _make_gas_data(n_boxes=2, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=2)
    temperatures = np.array([268.15, 303.15], dtype=np.float64)
    pressures = np.array([101325.0, 99500.0], dtype=np.float64)
    tension_values = np.array([0.035, 0.095], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    thermodynamics = _make_mixed_thermodynamics_config(gpu_gas)
    sidecar = _make_activity_surface_config(
        gpu_gas,
        int(ACTIVITY_MODE_KAPPA),
        int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
    )

    _, transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=wp.array(temperatures, dtype=wp.float64, device=device),
        pressure=wp.array(pressures, dtype=wp.float64, device=device),
        time_step=0.1,
        surface_tension=wp.array(
            tension_values, dtype=wp.float64, device=device
        ),
        thermodynamics=thermodynamics,
        activity_surface=sidecar,
    )
    _, expected, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension=tension_values,
        mass_accommodation=np.ones(2, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(2, 2.0e-5, dtype=np.float64),
        temperature=temperatures,
        pressure=pressures,
        time_step=0.1,
        thermodynamics=thermodynamics,
        activity_mode=int(ACTIVITY_MODE_KAPPA),
        surface_tension_mode=int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
        kappas=sidecar.kappas.numpy(),
    )

    npt.assert_allclose(transfer.numpy(), expected, rtol=1.0e-10, atol=0.0)
    npt.assert_allclose(
        gpu_gas.vapor_pressure.numpy(),
        _evaluate_vapor_pressure_config(thermodynamics, temperatures),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert not np.array_equal(transfer.numpy()[0], transfer.numpy()[1])


def test_condensation_activity_surface_invalid_sidecar_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid sidecars preserve caller buffers after mask-only preflight."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=np.full((1, 2), 17.0)
    )
    sidecar = _make_activity_surface_config(
        gpu_gas, int(ACTIVITY_MODE_IDEAL), int(SURFACE_TENSION_MODE_STATIC)
    )
    sidecar = dataclasses.replace(
        sidecar,
        kappas=wp.array([-1.0, 0.5], dtype=wp.float64, device=device),
    )
    tension = wp.array([0.04, 0.09], dtype=wp.float64, device=device)
    transfer = wp.full((1, 1, 2), 13.0, dtype=wp.float64, device=device)
    snapshots = tuple(
        values.numpy().copy()
        for values in (
            gpu_particles.masses,
            gpu_gas.vapor_pressure,
            sidecar.kappas,
            sidecar.molar_mass_reference,
            tension,
            transfer,
        )
    )

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            original_launch(kernel, *args, **kwargs)
            return
        raise AssertionError("invalid sidecar launched a non-preflight kernel")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)
    with pytest.raises(ValueError, match="kappas"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            surface_tension=tension,
            mass_transfer=transfer,
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            activity_surface=sidecar,
        )
    for values, expected in zip(
        (
            gpu_particles.masses,
            gpu_gas.vapor_pressure,
            sidecar.kappas,
            sidecar.molar_mass_reference,
            tension,
            transfer,
        ),
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(values.numpy(), expected)


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "surface_mode",
    [
        int(SURFACE_TENSION_MODE_STATIC),
        int(SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED),
    ],
)
@pytest.mark.parametrize("particle_case", ["zero_total_mass", "no_water"])
def test_condensation_activity_surface_handles_inactive_composition_edges(
    device: str,
    surface_mode: int,
    particle_case: str,
) -> None:
    """Configured modes retain finite, clamped edge-case mass transfers."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=2)
    if particle_case == "zero_total_mass":
        particles.masses[0, 0, :] = 0.0
    else:
        particles.masses[0, 0, :] = [0.0, 9.0e-18]
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    tension_values = np.array([0.035, 0.095], dtype=np.float64)
    transfers: dict[int, np.ndarray] = {}
    final_masses: dict[int, np.ndarray] = {}
    for activity_mode in (int(ACTIVITY_MODE_IDEAL), int(ACTIVITY_MODE_KAPPA)):
        gpu_particles = to_warp_particle_data(particles, device=device)
        gpu_gas = to_warp_gas_data(
            gas, device=device, vapor_pressure=vapor_pressure
        )
        sidecar = _make_activity_surface_config(
            gpu_gas, activity_mode, surface_mode
        )
        _, transfer = _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            surface_tension=wp.array(
                tension_values, dtype=wp.float64, device=device
            ),
            thermodynamics=_make_thermodynamics_config(gpu_gas),
            activity_surface=sidecar,
        )
        transfers[activity_mode] = transfer.numpy()
        final_masses[activity_mode] = from_warp_particle_data(
            gpu_particles, sync=True
        ).masses

    for activity_mode in transfers:
        assert np.all(np.isfinite(transfers[activity_mode]))
        assert np.all(final_masses[activity_mode] >= 0.0)
    if particle_case == "zero_total_mass":
        npt.assert_array_equal(
            transfers[int(ACTIVITY_MODE_IDEAL)],
            np.zeros_like(transfers[int(ACTIVITY_MODE_IDEAL)]),
        )
        npt.assert_array_equal(
            final_masses[int(ACTIVITY_MODE_KAPPA)],
            np.zeros_like(final_masses[int(ACTIVITY_MODE_KAPPA)]),
        )
    else:
        npt.assert_allclose(
            transfers[int(ACTIVITY_MODE_IDEAL)][:, :, 1],
            transfers[int(ACTIVITY_MODE_KAPPA)][:, :, 1],
            rtol=1.0e-12,
            atol=0.0,
        )


def test_condensation_step_gpu_scalar_positional_call_remains_valid(
    device: str,
) -> None:
    """Legacy positional scalar callers remain source-compatible."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    _, mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        298.15,
        101325.0,
        0.1,
    )
    wp.synchronize()

    assert mass_transfer.shape == (1, 2, 1)


def test_condensation_step_gpu_requires_thermodynamics_sidecar(
    device: str,
) -> None:
    """The public boundary rejects an omitted thermodynamic sidecar."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=_make_vapor_pressure(1, 1),
    )

    with pytest.raises(ValueError, match="thermodynamics"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, 101325.0),
        (298.15, None),
        (None, 101325.0),
    ],
)
def test_condensation_step_gpu_rejects_mixed_environment_inputs(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Mixed scalar and environment inputs raise a stable contract error."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    with pytest.raises(
        ValueError,
        match="direct temperature/pressure inputs with environment",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_accepts_explicit_environment(
    device: str,
) -> None:
    """Pure ``environment=...`` execution succeeds when inputs are valid."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    _, scalar_mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
    )
    scalar_result = np.asarray(scalar_mass_transfer.numpy()).copy()

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, environment_mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=None,
        pressure=None,
        time_step=0.1,
        environment=environment,
    )

    npt.assert_allclose(environment_mass_transfer.numpy(), scalar_result)


def test_condensation_step_gpu_uniform_direct_arrays_match_scalar_results(
    device: str,
) -> None:
    """Uniform per-box direct arrays preserve legacy scalar physics."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    _, scalar_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        device=device,
    )
    _, uniform_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array([298.15, 298.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 101325.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(
        uniform_mass_transfer.numpy(),
        scalar_mass_transfer.numpy(),
        rtol=_ENVIRONMENT_ARRAY_RTOL,
    )


def test_condensation_stiffness_case_builds_named_regimes() -> None:
    """Stiffness catalog exposes the expected deterministic baseline cases."""
    cases = _make_condensation_stiffness_cases()

    assert [case.name for case in cases] == [
        "nanometer",
        "accumulation_mode",
        "droplet_like",
    ]

    for case in cases:
        particles = case.build_particle_data()
        gas = case.build_gas_data()
        vapor_pressure = case.build_vapor_pressure()

        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
        assert particles.masses.shape == (
            case.n_boxes,
            case.n_particles,
            case.n_species,
        )
        assert particles.masses.dtype == np.float64
        assert gas.concentration.dtype == np.float64
        assert vapor_pressure.dtype == np.float64


def test_condensation_stiffness_case_zero_mass_preserves_fixed_shape() -> None:
    """Zero-mass helper coverage preserves deterministic fixed-shape outputs."""
    case = CondensationStiffnessCase(
        name="zero_mass_edge",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
        zero_mass_particles=((0, 1),),
    )

    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
    assert particles.masses.shape == (1, 2, 2)
    npt.assert_allclose(particles.masses[0, 1, :], 0.0)


def test_condensation_stiffness_case_direct_arrays_match_scalar_inputs(
    device: str,
) -> None:
    """Representative case supports scalar and direct ``(n_boxes,)`` inputs."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "droplet_like"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    _, scalar_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=case.time_step,
        device=device,
    )
    _, array_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            case.temperature_array(),
            dtype=wp.float64,
            device=device,
        ),
        pressure=wp.array(
            case.pressure_array(),
            dtype=wp.float64,
            device=device,
        ),
        time_step=case.time_step,
        device=device,
    )

    assert scalar_mass_transfer.shape == array_mass_transfer.shape
    assert array_mass_transfer.dtype == wp.float64
    assert array_mass_transfer.shape == (
        case.n_boxes,
        case.n_particles,
        case.n_species,
    )
    npt.assert_allclose(
        array_mass_transfer.numpy()[0],
        scalar_mass_transfer.numpy()[0],
        rtol=1.0e-10,
    )
    npt.assert_allclose(
        array_mass_transfer.numpy()[1],
        scalar_mass_transfer.numpy()[1],
        rtol=1.0e-10,
    )


def test_stiffness_metrics_reject_shape_metadata_mismatch() -> None:
    """Metadata validation fails clearly on declared shape mismatches."""
    case = CondensationStiffnessCase(
        name="shape_mismatch",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    particles.masses = np.zeros((1, 3, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="particle masses shape does not match declared case metadata",
    ):
        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)


def test_stiffness_metrics_reject_dtype_metadata_mismatch() -> None:
    """Metadata validation fails clearly on dtype mismatches."""
    case = CondensationStiffnessCase(
        name="dtype_mismatch",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure().astype(np.float32)

    with pytest.raises(TypeError, match="vapor pressure must use np.float64"):
        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)


def test_fractional_mass_change_zero_mass_returns_documented_result() -> None:
    """Zero-initial-mass entries report zero fractional change."""
    initial = np.array([[[0.0, 2.0e-18]]], dtype=np.float64)
    final = np.array([[[0.0, 3.0e-18]]], dtype=np.float64)

    fractional_change = _fractional_mass_change_per_bin(initial, final)

    npt.assert_allclose(fractional_change, [[[0.0, 0.5]]])
    assert _zero_mass_entries_remain_stable(initial, final)


def test_fractional_mass_change_rejects_shape_mismatch() -> None:
    """Fractional mass change helper rejects mismatched shapes."""
    initial = np.zeros((1, 1, 1), dtype=np.float64)
    final = np.zeros((1, 2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="initial and final masses must have matching shape",
    ):
        _fractional_mass_change_per_bin(initial, final)


def test_zero_mass_entries_remain_stable_rejects_shape_mismatch() -> None:
    """Zero-mass stability helper rejects mismatched shapes."""
    initial = np.zeros((1, 1, 1), dtype=np.float64)
    final = np.zeros((1, 2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="initial and final masses must have matching shape",
    ):
        _zero_mass_entries_remain_stable(initial, final)


def test_condensation_stiffness_metric_helpers_detect_invalid_values() -> None:
    """Metric helpers expose non-negativity and finiteness checks."""
    good = np.array([0.0, 1.0], dtype=np.float64)
    bad = np.array([1.0, np.inf], dtype=np.float64)

    assert _particle_mass_is_nonnegative(good)
    assert not _particle_mass_is_nonnegative(np.array([-1.0], dtype=np.float64))
    assert _particle_values_are_finite(good)
    assert not _particle_values_are_finite(good, bad)


def test_condensation_stiffness_classification_threshold_boundary() -> None:
    """Exact threshold equality remains classified as stable."""
    case = CondensationStiffnessCase(
        name="threshold_boundary",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[1.5e-18]]], dtype=np.float64)
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        gas,
        vapor_pressure,
        max_fractional_change=0.5,
    )

    assert classification.label == "stable"
    assert classification.max_fractional_mass_change == pytest.approx(0.5)
    assert classification.particle_only_update


def test_large_fractional_change_is_unstable() -> None:
    """Large fractional mass changes are classified as unstable."""
    case = CondensationStiffnessCase(
        name="large_change",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=1.0,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[3.0e-18]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=0.5,
    )

    assert classification.label == "unstable"
    assert classification.max_fractional_mass_change > classification.threshold
    assert classification.particle_only_update


def test_zero_mass_growth_is_unstable() -> None:
    """Zero-initial-mass growth is unstable even if fractional change is
    zero.
    """
    case = CondensationStiffnessCase(
        name="zero_mass_growth",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[0.0]]], dtype=np.float64)
    final = np.array([[[1.0e-21]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=0.0,
    )

    assert classification.label == "unstable"
    assert classification.zero_mass_change_stable is False
    assert classification.max_fractional_mass_change == 0.0


def test_nonfinite_values_are_unstable() -> None:
    """Non-finite particle results are unstable regardless of threshold."""
    case = CondensationStiffnessCase(
        name="nonfinite_values",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[np.nan]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=10.0,
    )

    assert classification.label == "unstable"
    assert classification.values_finite is False


def test_condensation_stiffness_classification_explicitly_marks_particle_only(
    device: str,
) -> None:
    """Classification exposes the particle-only caveat without gas claims."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "accumulation_mode"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    initial_gas = gas.concentration.copy()
    initial_masses = particles.masses.copy()

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=case.time_step,
        device=device,
    )
    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial_masses,
        gpu_result.masses,
        gas,
        vapor_pressure,
        max_fractional_change=10.0,
    )

    assert classification.particle_only_update is True
    npt.assert_allclose(gas.concentration, initial_gas)


def test_run_gpu_step_can_round_trip_executed_gas_state(device: str) -> None:
    """Helper can observe the executed Warp gas state directly."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)

    gpu_result, mass_transfer, executed_gas = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        device=device,
        return_gas_state=True,
    )

    assert mass_transfer.shape == (1, 1, 1)
    assert gpu_result.masses.shape == particles.masses.shape
    npt.assert_allclose(
        gas.concentration - executed_gas.concentration,
        np.sum(
            mass_transfer.numpy() * particles.concentration[:, :, None], axis=1
        ),
        rtol=1e-12,
        atol=1e-22,
    )
    assert executed_gas is not gas


def _recorded_condensation_trials_detect_executed_gas_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Recorded trials fail the gas-unchanged flag on executed-gas mutation."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    original_run_gpu_step = _run_gpu_step
    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "nanometer"
    )

    def _mutating_run_gpu_step(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        particle_result, mass_transfer, executed_gas = original_run_gpu_step(
            *args,
            **kwargs,
        )
        mutated_gas = GasData(
            name=list(executed_gas.name),
            molar_mass=executed_gas.molar_mass.copy(),
            concentration=executed_gas.concentration.copy(),
            partitioning=executed_gas.partitioning.copy(),
        )
        mutated_gas.concentration[0, 0] += 1.0
        return particle_result, mass_transfer, mutated_gas

    monkeypatch.setitem(globals(), "_run_gpu_step", _mutating_run_gpu_step)

    records = _record_condensation_stiffness_trials(
        case,
        device=device,
        use_cache=False,
    )

    assert not any(record.gas_unchanged for record in records)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_missing_scalar_inputs_without_environment_are_rejected(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Scalar-mode calls require both temperature and pressure."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            298.15,
            101325.0,
            "direct temperature/pressure inputs with environment",
        ),
        (
            None,
            None,
            r"\(n_boxes,\)",
        ),
    ],
)
def test_condensation_step_gpu_contract_errors_short_circuit_before_helpers(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
    message: str,
) -> None:
    """Contract errors follow the mask-only preflight before normal work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment_data = _make_environment_data(n_boxes=1, n_species=1)
    if temperature is None and pressure is None:
        environment_data.temperature = np.array([298.15, 299.15])
    environment = to_warp_environment_data(environment_data, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )

    assert calls == ["partitioning_validation"]


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_missing_scalar_inputs_short_circuit_before_helpers(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Missing direct inputs fail before buffer preparation or launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == ["partitioning_validation"]


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (0.0, 101325.0, "temperature must be finite and > 0"),
        (298.15, 0.0, "pressure must be finite and > 0"),
        (float("nan"), 101325.0, "temperature must be finite and > 0"),
    ],
)
def test_invalid_scalar_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float,
    pressure: float,
    message: str,
) -> None:
    """Invalid scalar domains fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == ["partitioning_validation"]


def test_invalid_environment_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment arrays fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    environment.pressure = wp.array([0.0], dtype=wp.float64, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )

    assert calls == ["partitioning_validation"]


@pytest.mark.parametrize("field_name", ["temperature", "pressure"])
def test_missing_environment_field_short_circuits_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    field_name: str,
) -> None:
    """Malformed environment payloads raise stable errors before launch."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    original_masses = np.asarray(gpu_particles.masses.numpy()).copy()
    mass_transfer = wp.full(
        (1, 1, 1),
        wp.float64(5.0),
        dtype=wp.float64,
        device=device,
    )
    original_mass_transfer = np.asarray(mass_transfer.numpy()).copy()

    class _MalformedEnvironment:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0], dtype=wp.float64, device=device
            )

    environment = _MalformedEnvironment()
    delattr(environment, field_name)
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match=rf"environment\.{field_name} must be a Warp array with shape "
        r"\(n_boxes,\)",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            mass_transfer=mass_transfer,
            environment=environment,
        )

    assert calls == ["partitioning_validation"]
    npt.assert_allclose(gpu_particles.masses.numpy(), original_masses)
    npt.assert_allclose(mass_transfer.numpy(), original_mass_transfer)


@pytest.mark.parametrize(
    ("temperature_values", "pressure_values", "message"),
    [
        (
            np.array([298.15, 0.0], dtype=np.float64),
            np.array([101325.0, 101325.0], dtype=np.float64),
            "temperature must be finite and > 0",
        ),
        (
            np.array([298.15, 298.15], dtype=np.float64),
            np.array([101325.0, np.nan], dtype=np.float64),
            "pressure must be finite and > 0",
        ),
    ],
)
def test_invalid_direct_array_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature_values: np.ndarray,
    pressure_values: np.ndarray,
    message: str,
) -> None:
    """Invalid direct array domains fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=wp.array(
                temperature_values,
                dtype=wp.float64,
                device=device,
            ),
            pressure=wp.array(
                pressure_values,
                dtype=wp.float64,
                device=device,
            ),
            time_step=0.1,
        )

    assert calls == ["partitioning_validation"]


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            np.array([298.15], dtype=np.float64),
            101325.0,
            "temperature must be a scalar or Warp array with shape",
        ),
        (
            298.15,
            np.array([101325.0], dtype=np.float64),
            "pressure must be a scalar or Warp array with shape",
        ),
    ],
)
def test_condensation_step_gpu_rejects_direct_non_warp_arrays_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
    message: str,
) -> None:
    """Unsupported direct non-Warp arrays fail before launch or mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    original_masses = np.asarray(gpu_particles.masses.numpy()).copy()
    mass_transfer = wp.full(
        (1, 1, 1),
        wp.float64(9.0),
        dtype=wp.float64,
        device=device,
    )
    original_mass_transfer = np.asarray(mass_transfer.numpy()).copy()
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            mass_transfer=mass_transfer,
        )

    assert calls == ["partitioning_validation"]
    npt.assert_allclose(gpu_particles.masses.numpy(), original_masses)
    npt.assert_allclose(mass_transfer.numpy(), original_mass_transfer)


@pytest.mark.parametrize(
    ("field_name", "field_values", "message"),
    [
        (
            "temperature",
            np.array([np.nan, 298.15], dtype=np.float64),
            "environment.temperature must be finite and > 0",
        ),
        (
            "pressure",
            np.array([101325.0, 0.0], dtype=np.float64),
            "environment.pressure must be finite and > 0",
        ),
    ],
)
def test_invalid_environment_array_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    field_name: str,
    field_values: np.ndarray,
    message: str,
) -> None:
    """Invalid environment arrays fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    setattr(
        environment,
        field_name,
        wp.array(field_values, dtype=wp.float64, device=device),
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    original_launch = condensation_module.wp.launch

    def _unexpected_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        if kernel is condensation_module._validate_partitioning_values_kernel:
            calls.append("partitioning_validation")
            original_launch(kernel, *args, **kwargs)
            return
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )

    assert calls == ["partitioning_validation"]


def test_condensation_step_gpu_accepts_direct_environment_arrays(
    device: str,
) -> None:
    """Direct ``(n_boxes,)`` Warp-array inputs match scalar results."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    expected_masses, expected, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        # The wrapper's default constant thermodynamics sidecar refreshes both
        # boxes to 800 Pa before the direct-environment transfer calculation.
        np.full_like(vapor_pressure, 800.0),
        np.array([0.072], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0e-5], dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
    )

    gpu_result, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            temperature_values, dtype=wp.float64, device=device
        ),
        pressure=wp.array(pressure_values, dtype=wp.float64, device=device),
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(
        mass_transfer.numpy(), expected, rtol=_ENVIRONMENT_ARRAY_RTOL
    )
    npt.assert_allclose(gpu_result.masses, expected_masses)


def test_explicit_environment_matches_direct_arrays(
    device: str,
) -> None:
    """Accepted ``environment=...`` arrays match direct-array inputs."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _, direct_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            temperature_values,
            dtype=wp.float64,
            device=device,
        ),
        pressure=wp.array(
            pressure_values,
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )
    _, environment_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    npt.assert_allclose(
        environment_mass_transfer.numpy(),
        direct_mass_transfer.numpy(),
        rtol=_ENVIRONMENT_ARRAY_RTOL,
    )


def test_success_does_not_mutate_environment_inputs(
    device: str,
) -> None:
    """Successful ``environment=...`` execution preserves inputs."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    npt.assert_allclose(environment.temperature.numpy(), temperature_values)
    npt.assert_allclose(environment.pressure.numpy(), pressure_values)


@pytest.mark.parametrize(
    ("temperature_input", "pressure_input"),
    [
        (
            298.15,
            np.array([101325.0, 100800.0], dtype=np.float64),
        ),
        (
            np.array([298.15, 301.15], dtype=np.float64),
            101325.0,
        ),
    ],
)
def test_condensation_step_gpu_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature_input: float | np.ndarray,
    pressure_input: float | np.ndarray,
) -> None:
    """Hybrid direct inputs match the CPU reference path."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = (
        np.full((2,), temperature_input, dtype=np.float64)
        if isinstance(temperature_input, float)
        else temperature_input
    )
    pressure_values = (
        np.full((2,), pressure_input, dtype=np.float64)
        if isinstance(pressure_input, float)
        else pressure_input
    )
    _, expected, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        np.full_like(vapor_pressure, 800.0),
        np.array([0.072], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0e-5], dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
    )

    temperature = temperature_input
    if isinstance(temperature_input, np.ndarray):
        temperature = wp.array(
            temperature_input, dtype=wp.float64, device=device
        )
    pressure = pressure_input
    if isinstance(pressure_input, np.ndarray):
        pressure = wp.array(pressure_input, dtype=wp.float64, device=device)

    _, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(
        mass_transfer.numpy(), expected, rtol=_ENVIRONMENT_ARRAY_RTOL
    )


def test_condensation_step_gpu_preserves_direct_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Direct Warp arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_dtypes: list[tuple[Any, Any]] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if (
            getattr(kernel, "key", "")
            == "_prepare_environment_properties_kernel"
        ):
            launch_dtypes.append((inputs[0].dtype, inputs[1].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=wp.array([298.15, 301.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 100800.0], dtype=wp.float64, device=device
        ),
        time_step=0.1,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)] * 4


def test_condensation_step_gpu_non_uniform_environment_matches_cpu(
    device: str,
) -> None:
    """Non-uniform explicit environment inputs reach box-local physics."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 308.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 98000.0], dtype=np.float64)
    _, expected, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        np.full_like(vapor_pressure, 800.0),
        np.array([0.072], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0e-5], dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
    )
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    assert not np.allclose(expected[0], expected[1], rtol=1.0e-6, atol=0.0)
    npt.assert_allclose(
        mass_transfer.numpy(), expected, rtol=_ENVIRONMENT_ARRAY_RTOL
    )


def test_condensation_step_gpu_preserves_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explicit environment arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    class _EnvironmentLike:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15, 301.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )

    environment = _EnvironmentLike()
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_dtypes: list[tuple[Any, Any]] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if (
            getattr(kernel, "key", "")
            == "_prepare_environment_properties_kernel"
        ):
            launch_dtypes.append((inputs[0].dtype, inputs[1].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=None,
        pressure=None,
        time_step=0.1,
        environment=environment,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)] * 4


def test_condensation_step_gpu_environment_shape_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays must match ``(n_boxes,)`` before launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=device
    )
    environment.temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_environment_device_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays on the wrong device fail before launch work."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    environment = to_warp_environment_data(
        _make_environment_data(1, 1),
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="environment.temperature device"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_direct_temperature_shape_mismatch_raises(
    device: str,
) -> None:
    """Direct temperature arrays must match ``(n_boxes,)`` before launch."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    temperature = wp.array([298.15], dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=101325.0,
            time_step=0.1,
        )


def test_condensation_step_gpu_direct_pressure_device_mismatch_raises(
    device: str,
) -> None:
    """Direct pressure arrays on the wrong device fail before launch."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    pressure = wp.array([101325.0], dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="pressure device"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=pressure,
            time_step=0.1,
        )


def test_condensation_step_gpu_prepares_box_properties_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Condensation precomputes box properties once per entry-point call."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_names: list[str] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launch_names.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
    )

    assert launch_names.count("_prepare_environment_properties_kernel") == 4


def test_condensation_step_gpu_zero_timestep_runs_four_ordered_cycles(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Zero timesteps clear totals and still schedule four physics cycles."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    initial_masses = particles.masses.copy()
    initial_gas = gas.concentration.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=_make_vapor_pressure(1, 1),
    )
    total = wp.full(
        particles.masses.shape,
        wp.float64(17.0),
        dtype=wp.float64,
        device=device,
    )
    scratch = _make_condensation_scratch_buffers(
        particles.masses.shape,
        device,
        fields=(
            "positive_mass_transfer_demand",
            "negative_mass_transfer_release",
            "positive_mass_transfer_scale",
        ),
    )
    launch_names: list[str] = []
    transfer_time_steps: list[float] = []
    latent_enabled_flags: list[int] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        name = getattr(kernel, "key", str(kernel))
        launch_names.append(name)
        if name == "condensation_mass_transfer_kernel":
            latent_enabled_flags.append(int(kwargs["inputs"][17]))
            transfer_time_steps.append(float(kwargs["inputs"][23]))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    _, returned = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.0,
        mass_transfer=total,
        latent_heat=wp.array([2.2e6], dtype=wp.float64, device=device),
        scratch_buffers=scratch,
    )

    assert returned is total
    assert (
        launch_names
        == [
            "_validate_partitioning_values_kernel",
            "_clear_mass_transfer_kernel",
        ]
        + [
            "_refresh_vapor_pressure_kernel",
            "_prepare_environment_properties_kernel",
            "condensation_mass_transfer_kernel",
            "_gate_mass_transfer_kernel",
            "_bound_evaporation_candidate_kernel",
            "_reduce_inventory_candidates_kernel",
            "_scale_inventory_uptake_kernel",
            "_finalize_and_apply_inventory_transfer_kernel",
            "_reduce_finalized_transfer_to_gas_kernel",
            "apply_mass_transfer_kernel",
            "_accumulate_finalized_mass_transfer_kernel",
            "_couple_finalized_transfer_to_gas_kernel",
        ]
        * 4
    )
    assert transfer_time_steps == [0.0] * 4
    assert latent_enabled_flags == [1] * 4
    npt.assert_array_equal(gpu_particles.masses.numpy(), initial_masses)
    npt.assert_array_equal(gpu_gas.concentration.numpy(), initial_gas)
    npt.assert_array_equal(total.numpy(), np.zeros_like(initial_masses))
    npt.assert_array_equal(scratch.positive_mass_transfer_demand.numpy(), 0.0)
    npt.assert_array_equal(scratch.negative_mass_transfer_release.numpy(), 0.0)
    npt.assert_array_equal(scratch.positive_mass_transfer_scale.numpy(), 1.0)


@pytest.mark.gpu_parity
def test_condensation_step_gpu_matches_cpu_single_box(device: str) -> None:
    """GPU condensation matches CPU for a single box."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.8], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.5e-5], dtype=np.float64)

    expected_masses, _, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


@pytest.mark.gpu_parity
def test_condensation_step_gpu_multi_box_matches_cpu(device: str) -> None:
    """GPU condensation matches CPU for multiple boxes."""
    temperature = 300.0
    pressure = 100000.0
    time_step = 0.5
    particles = _make_particle_data(n_boxes=3, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=3, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=3, n_species=2)
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([0.9, 0.7], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.7e-5], dtype=np.float64)

    expected_masses, _, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


def test_apply_mass_transfer_kernel_clamps_negative(device: str) -> None:
    """Masses clamp to non-negative when evaporation exceeds mass."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 10.0
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = np.full((1, 1), 1.0e6, dtype=np.float64)
    surface_tension = np.array([0.072], dtype=np.float64)
    mass_accommodation = np.array([1.0], dtype=np.float64)
    diffusion = np.array([2.0e-5], dtype=np.float64)

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    assert np.all(gpu_result.masses >= 0.0)


@pytest.mark.parametrize(
    "use_scratch",
    (False, True),
    ids=("legacy_total", "scratch_work_and_total"),
)
def test_condensation_forced_evaporation_accumulates_applied_transfer(
    device: str,
    use_scratch: bool,
) -> None:
    """Forced evaporation reports bounded applied mass and energy transfer."""
    case = P4_CASES[0]
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    initial_mass = particles.masses.copy()
    latent_values = np.array([2.2e6, 1.5e6], dtype=np.float64)
    expected_mass, expected_total, expected_work = _cpu_four_substep_oracle(
        particles,
        gas,
        _p4_vapor_pressure(case.temperature),
        surface_tension=P4_THERMODYNAMICS.surface_tension,
        mass_accommodation=np.ones(2, dtype=np.float64),
        diffusion_coefficient_vapor=np.full(2, 2.0e-5, dtype=np.float64),
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=1000.0,
        kappas=P4_THERMODYNAMICS.kappas,
        latent_heat=latent_values,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.zeros((1, 2), dtype=np.float64),
    )
    thermodynamics, activity_surface, tension = _p4_sidecars(device)
    scratch = (
        _make_condensation_scratch_buffers(particles.masses.shape, device)
        if use_scratch
        else None
    )
    legacy_total = (
        None
        if use_scratch
        else wp.full(
            particles.masses.shape,
            wp.float64(7.0),
            dtype=wp.float64,
            device=device,
        )
    )
    energy_transfer = wp.full(
        (1, 2), wp.float64(31.0), dtype=wp.float64, device=device
    )

    _, returned_total = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=wp.array(case.temperature, dtype=wp.float64, device=device),
        pressure=wp.array(case.pressure, dtype=wp.float64, device=device),
        time_step=1000.0,
        surface_tension=tension,
        thermodynamics=thermodynamics,
        activity_surface=activity_surface,
        mass_transfer=legacy_total,
        scratch_buffers=scratch,
        latent_heat=wp.array(latent_values, dtype=wp.float64, device=device),
        energy_transfer=energy_transfer,
    )

    expected_total_buffer = (
        scratch.total_mass_transfer if scratch is not None else legacy_total
    )
    assert returned_total is expected_total_buffer
    assert returned_total.device == wp.get_device(device)
    npt.assert_allclose(
        gpu_particles.masses.numpy(), expected_mass, rtol=1.0e-10
    )
    npt.assert_allclose(returned_total.numpy(), expected_total, rtol=1.0e-10)
    npt.assert_allclose(
        energy_transfer.numpy(),
        expected_total.sum(axis=1) * latent_values[None, :],
        rtol=1e-12,
        atol=1e-18,
    )
    assert not np.array_equal(
        energy_transfer.numpy(),
        expected_work.sum(axis=1) * latent_values[None, :],
    )
    npt.assert_allclose(
        gpu_particles.masses.numpy(),
        initial_mass + returned_total.numpy(),
        rtol=1e-12,
    )
    assert np.all(np.isfinite(gpu_particles.masses.numpy()))
    assert np.all(gpu_particles.masses.numpy() >= 0.0)
    npt.assert_allclose(
        gas.concentration - gpu_gas.concentration.numpy(),
        np.sum(expected_total * particles.concentration[:, :, None], axis=1),
        rtol=1e-12,
        atol=1e-23,
    )
    if scratch is not None:
        assert scratch.work_mass_transfer is not None
        npt.assert_allclose(
            scratch.work_mass_transfer.numpy(), expected_work, rtol=1.0e-10
        )


def test_condensation_skips_inactive_particles(device: str) -> None:
    """Inactive particles retain their masses."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.concentration[0, 1] = 0.0
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)

    initial_mass = particles.masses[0, 1, 0]
    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
    )

    assert gpu_result.masses[0, 1, 0] == pytest.approx(initial_mass)


@pytest.mark.gpu_parity
def test_condensation_multi_species_parity(device: str) -> None:
    """Multi-species GPU condensation matches CPU."""
    temperature = 295.0
    pressure = 100500.0
    time_step = 0.8
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=3)
    gas = _make_gas_data(n_boxes=1, n_species=3)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=3)
    surface_tension = np.array([0.072, 0.08, 0.1], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.9, 0.7], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.7e-5, 1.2e-5], dtype=np.float64)

    expected_masses, _, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


def test_condensation_step_gpu_refreshes_before_mass_transfer(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Refresh runs per substep after one float32 cast and before transfer."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=2, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.full((2, 2), 1.0, dtype=np.float64),
    )
    thermodynamics = _make_mixed_thermodynamics_config(gpu_gas)
    temperature = wp.array([273.15, 301.15], dtype=wp.float32, device=device)
    pressure = wp.array([101325.0, 100800.0], dtype=wp.float32, device=device)
    launch_names: list[str] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launch_names.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    _, transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature,
        pressure,
        0.1,
        thermodynamics=thermodynamics,
    )

    expected_pressure = _evaluate_vapor_pressure_config(
        thermodynamics,
        np.array([273.15, 301.15], dtype=np.float32).astype(np.float64),
    )
    npt.assert_allclose(
        gpu_gas.vapor_pressure.numpy(),
        expected_pressure,
        rtol=2.0e-7,
    )
    assert np.all(np.isfinite(transfer.numpy()))
    assert launch_names.count("_refresh_vapor_pressure_kernel") == 4
    assert launch_names.index("_copy_temperature_to_float64_kernel") < (
        launch_names.index("_refresh_vapor_pressure_kernel")
    )
    assert (
        launch_names.index("_refresh_vapor_pressure_kernel")
        < (launch_names.index("_prepare_environment_properties_kernel"))
        < launch_names.index("condensation_mass_transfer_kernel")
    )


def test_condensation_step_gpu_refreshes_reused_gas_at_new_temperature(
    device: str,
) -> None:
    """A reused gas sidecar is overwritten from each current temperature."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=2, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.full((2, 2), 1.0, dtype=np.float64),
    )
    thermodynamics = _make_mixed_thermodynamics_config(gpu_gas)
    pressure = wp.full(2, 101325.0, dtype=wp.float64, device=device)
    first_temperature = wp.array(
        [273.15, 280.0], dtype=wp.float64, device=device
    )
    second_temperature = wp.array(
        [290.0, 305.0], dtype=wp.float64, device=device
    )

    _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        first_temperature,
        pressure,
        0.1,
        thermodynamics=thermodynamics,
    )
    first_pressure = gpu_gas.vapor_pressure.numpy().copy()
    _, second_transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        second_temperature,
        pressure,
        0.1,
        thermodynamics=thermodynamics,
    )
    second_pressure = gpu_gas.vapor_pressure.numpy().copy()

    npt.assert_allclose(
        first_pressure,
        _evaluate_vapor_pressure_config(
            thermodynamics, np.array([273.15, 280.0])
        ),
    )
    npt.assert_allclose(
        second_pressure,
        _evaluate_vapor_pressure_config(
            thermodynamics, np.array([290.0, 305.0])
        ),
    )
    assert not np.array_equal(first_pressure, second_pressure)
    assert np.all(np.isfinite(second_transfer.numpy()))


@pytest.mark.parametrize("input_kind", ["scalar", "array", "environment"])
def test_condensation_step_gpu_refreshes_stale_pressure_for_each_input_kind(
    input_kind: str,
    device: str,
) -> None:
    """Refresh stale pressure and match CPU transfer for each input contract."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=2)
    gas = _make_gas_data(n_boxes=2, n_species=2)
    temperature_values = np.array([278.15, 303.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    if input_kind == "scalar":
        temperature_values.fill(298.15)
        pressure_values.fill(101325.0)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.full((2, 2), 1.0, dtype=np.float64),
    )
    thermodynamics = _make_mixed_thermodynamics_config(gpu_gas)
    if input_kind == "environment":
        environment = to_warp_environment_data(
            EnvironmentData(
                temperature=temperature_values,
                pressure=pressure_values,
                saturation_ratio=np.ones((2, 2), dtype=np.float64),
            ),
            device=device,
        )
        temperature: float | Any | None = None
        pressure: float | Any | None = None
    elif input_kind == "array":
        environment = None
        temperature = wp.array(
            temperature_values, dtype=wp.float64, device=device
        )
        pressure = wp.array(pressure_values, dtype=wp.float64, device=device)
    else:
        environment = None
        temperature = float(temperature_values[0])
        pressure = float(pressure_values[0])

    _, transfer = _condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature,
        pressure,
        0.1,
        environment=environment,
        thermodynamics=thermodynamics,
    )
    expected_pressure = _evaluate_vapor_pressure_config(
        thermodynamics,
        temperature_values,
    )
    _, expected_transfer, _ = _cpu_four_substep_oracle(
        particles,
        gas,
        np.full((2, 2), 1.0, dtype=np.float64),
        np.full(2, 0.072, dtype=np.float64),
        np.full(2, 1.0, dtype=np.float64),
        np.full(2, 2.0e-5, dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
        thermodynamics=thermodynamics,
    )

    npt.assert_allclose(gpu_gas.vapor_pressure.numpy(), expected_pressure)
    npt.assert_allclose(transfer.numpy(), expected_transfer, rtol=1.0e-10)


@pytest.mark.parametrize(
    ("failure_kind", "match"),
    [
        ("thermodynamics", "thermodynamics"),
        ("temperature", "temperature"),
        ("mass_transfer", "mass_transfer shape"),
    ],
)
def test_condensation_step_gpu_pre_refresh_failures_are_atomic(
    failure_kind: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Validation failures launch no refresh and preserve caller-owned state."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.array([[17.0]], dtype=np.float64),
    )
    thermodynamics: ThermodynamicsConfig | None = _make_thermodynamics_config(
        gpu_gas
    )
    temperature: float = 298.15
    mass_transfer: Any | None = None
    if failure_kind == "thermodynamics":
        thermodynamics = None
    elif failure_kind == "temperature":
        temperature = float("nan")
    else:
        mass_transfer = wp.zeros((1, 1, 2), dtype=wp.float64, device=device)

    initial_mass = gpu_particles.masses.numpy().copy()
    initial_pressure = gpu_gas.vapor_pressure.numpy().copy()
    launch_names: list[str] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launch_names.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    with pytest.raises(ValueError, match=match):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature,
            101325.0,
            0.1,
            mass_transfer=mass_transfer,
            thermodynamics=thermodynamics,
        )
    assert "_refresh_vapor_pressure_kernel" not in launch_names
    npt.assert_allclose(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_allclose(gpu_gas.vapor_pressure.numpy(), initial_pressure)


def test_condensation_step_gpu_invalid_optional_buffer_does_not_refresh(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Optional-buffer failures leave gas and particle data untouched."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=np.array([[17.0]], dtype=np.float64),
    )
    thermodynamics = _make_thermodynamics_config(gpu_gas)
    initial_mass = gpu_particles.masses.numpy()
    initial_pressure = gpu_gas.vapor_pressure.numpy()
    launch_names: list[str] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launch_names.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)
    invalid_transfer = wp.zeros((1, 1, 2), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="mass_transfer shape"):
        _condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            298.15,
            101325.0,
            0.1,
            mass_transfer=invalid_transfer,
            thermodynamics=thermodynamics,
        )
    assert "_refresh_vapor_pressure_kernel" not in launch_names
    npt.assert_allclose(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_allclose(gpu_gas.vapor_pressure.numpy(), initial_pressure)


@pytest.mark.parametrize("case_name", ["nanometer", "droplet_like"])
def test_fixed_count_candidate_is_deterministic_for_named_stiffness_case(
    case_name: str,
) -> None:
    """Fixed-count candidate produces identical outputs on repeated runs."""
    case = _get_condensation_stiffness_case(case_name)
    candidate = _get_integration_candidate("fixed_count_substeps_4")
    time_step = _RECORDED_TIMESTEP_GRID_BY_CASE[case_name][-1]

    first = _run_integration_candidate(
        case,
        candidate,
        time_step,
        _make_candidate_scratch(case, candidate),
    )
    second = _run_integration_candidate(
        case,
        candidate,
        time_step,
        _make_candidate_scratch(case, candidate),
    )

    npt.assert_array_equal(first.final_masses, second.final_masses)
    npt.assert_array_equal(first.mass_transfer, second.mass_transfer)


@pytest.mark.parametrize("case_name", ["nanometer", "droplet_like"])
def test_asymptotic_candidate_is_deterministic_for_named_stiffness_case(
    case_name: str,
) -> None:
    """Asymptotic candidate produces identical outputs on repeated runs."""
    case = _get_condensation_stiffness_case(case_name)
    candidate = _get_integration_candidate("asymptotic_relaxation")
    time_step = _RECORDED_TIMESTEP_GRID_BY_CASE[case_name][-1]

    first = _run_integration_candidate(
        case,
        candidate,
        time_step,
        _make_candidate_scratch(case, candidate),
    )
    second = _run_integration_candidate(
        case,
        candidate,
        time_step,
        _make_candidate_scratch(case, candidate),
    )

    npt.assert_array_equal(first.final_masses, second.final_masses)
    npt.assert_array_equal(first.mass_transfer, second.mass_transfer)


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
@pytest.mark.parametrize(
    "candidate",
    _CONDENSATION_INTEGRATION_CANDIDATES,
    ids=lambda candidate: candidate.name,
)
def test_candidate_outputs_are_finite_and_non_negative(
    case: CondensationStiffnessCase,
    candidate: CondensationIntegrationCandidate,
) -> None:
    """Candidate outputs stay finite and particle-only masses stay
    non-negative.
    """
    result = _run_integration_candidate(
        case,
        candidate,
        case.time_step,
        _make_candidate_scratch(case, candidate),
    )

    assert np.all(np.isfinite(result.final_masses))
    assert np.all(result.final_masses >= 0.0)
    assert np.all(np.isfinite(result.mass_transfer))


@pytest.mark.parametrize(
    "candidate",
    _CONDENSATION_INTEGRATION_CANDIDATES,
    ids=lambda candidate: candidate.name,
)
def test_candidate_reuses_mass_transfer_and_fixed_shape_scratch_buffers(
    candidate: CondensationIntegrationCandidate,
) -> None:
    """Candidate execution reuses caller-owned fixed-shape scratch arrays."""
    case = _get_condensation_stiffness_case("droplet_like")
    scratch = _make_candidate_scratch(case, candidate)
    mass_transfer_id = id(scratch.mass_transfer)
    work_id = id(scratch.work)
    accumulator_id = (
        id(scratch.accumulator) if scratch.accumulator is not None else None
    )
    shape = (case.n_boxes, case.n_particles, case.n_species)

    first = _run_integration_candidate(case, candidate, case.time_step, scratch)
    second = _run_integration_candidate(
        case, candidate, case.time_step, scratch
    )

    assert id(first.mass_transfer) == mass_transfer_id
    assert id(second.mass_transfer) == mass_transfer_id
    assert id(first.work) == work_id
    assert id(second.work) == work_id
    assert first.mass_transfer.shape == shape
    assert second.mass_transfer.shape == shape
    if scratch.accumulator is None:
        assert first.accumulator is None
        assert second.accumulator is None
    else:
        assert accumulator_id is not None
        first_accumulator = first.accumulator
        second_accumulator = second.accumulator
        assert first_accumulator is not None
        assert second_accumulator is not None
        assert id(first_accumulator) == accumulator_id
        assert id(second_accumulator) == accumulator_id
        assert first_accumulator.shape == shape
        assert second_accumulator.shape == shape


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
@pytest.mark.parametrize(
    "candidate",
    _CONDENSATION_INTEGRATION_CANDIDATES,
    ids=lambda candidate: candidate.name,
)
def test_candidate_matches_cpu_reference_with_documented_tolerance(
    case: CondensationStiffnessCase,
    candidate: CondensationIntegrationCandidate,
) -> None:
    """Candidate stays within its documented CPU-reference agreement bound."""
    reference = _cpu_reference_final_masses(case.name, case.time_step)
    result = _run_integration_candidate(
        case,
        candidate,
        case.time_step,
        _make_candidate_scratch(case, candidate),
    )

    npt.assert_allclose(
        result.final_masses,
        reference,
        rtol=candidate.reference_rtol,
        atol=0.0,
        err_msg=(
            f"{candidate.name} should stay within rtol="
            f"{candidate.reference_rtol} for {case.name}"
        ),
    )


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
@pytest.mark.parametrize(
    "candidate",
    _CONDENSATION_INTEGRATION_CANDIDATES,
    ids=lambda candidate: candidate.name,
)
def test_candidate_preserves_recorded_explicit_baseline_ordering_or_error_bound(
    case: CondensationStiffnessCase,
    candidate: CondensationIntegrationCandidate,
) -> None:
    """Candidate stays within a documented error bound across the recorded
    grid.
    """
    recorded_timesteps = _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    explicit_baseline = _recorded_explicit_final_masses_by_case(case.name)

    relative_errors = []
    for time_step, baseline_masses in zip(
        recorded_timesteps,
        explicit_baseline,
        strict=True,
    ):
        result = _run_integration_candidate(
            case,
            candidate,
            time_step,
            _make_candidate_scratch(case, candidate),
        )
        relative_errors.append(
            _relative_mass_error(result.final_masses, baseline_masses)
        )

    assert max(relative_errors) <= candidate.baseline_relative_error_bound


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
def test_condensation_stiffness_recorded_grid_uses_one_evidence_rule(
    case: CondensationStiffnessCase,
    device: str,
) -> None:
    """Recorded sweeps apply one shared evidence-backed stability rule."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    records = _record_condensation_stiffness_trials(case, device=device)

    assert {record.classification.threshold for record in records} == {
        _RECORDED_STIFFNESS_THRESHOLD
    }
    for record in records:
        assert record.classification.label == "stable"
        assert record.time_step == record.configured_time_step
        assert record.classification.mass_nonnegative
        assert record.classification.values_finite
        assert record.classification.particle_only_update
        assert record.classification.zero_mass_change_stable
        npt.assert_allclose(
            record.initial_gas_concentration - record.final_gas_concentration,
            np.sum(
                record.mass_transfer_values
                * case.build_particle_data().concentration[:, :, None],
                axis=1,
            ),
            rtol=1e-12,
            atol=1e-22,
        )
        assert record.reuses_caller_mass_transfer_buffer
        assert record.mass_transfer_has_nonzero_values
        assert np.all(np.isfinite(record.final_masses))
        assert np.all(record.final_masses >= 0.0)


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
def test_condensation_stiffness_recorded_grid_matches_configured_timesteps(
    case: CondensationStiffnessCase,
    device: str,
) -> None:
    """Recorded sweeps preserve configured timestep count, order, and mode."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    records = _record_condensation_stiffness_trials(case, device=device)
    configured_timesteps = _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]

    assert len(records) == len(configured_timesteps)
    assert [record.timestep_index for record in records] == list(
        range(len(configured_timesteps))
    )
    assert [record.time_step for record in records] == list(
        configured_timesteps
    )
    assert [record.configured_time_step for record in records] == list(
        configured_timesteps
    )
    if case.n_boxes == 1:
        assert {record.environment_input_mode for record in records} == {
            "scalar_inputs"
        }
    else:
        assert {record.environment_input_mode for record in records} == {
            "direct_warp_arrays"
        }


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
def test_condensation_stiffness_recorded_grid_reports_measured_change(
    case: CondensationStiffnessCase,
    device: str,
) -> None:
    """Recorded sweeps report the measured fractional change consistently."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    records = _record_condensation_stiffness_trials(case, device=device)

    assert all(
        record.classification.max_fractional_mass_change
        == pytest.approx(_RECORDED_STIFFNESS_THRESHOLD)
        for record in records
    )


def test_condensation_stiffness_recorded_grid_cache_matches_uncached_execution(
    device: str,
) -> None:
    """Cached recorded-grid helpers preserve uncached standard-path results."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    case = _get_condensation_stiffness_case("droplet_like")
    cached_records = _record_condensation_stiffness_trials(case, device=device)
    uncached_records = _record_condensation_stiffness_trials(
        case,
        device=device,
        use_cache=False,
    )

    assert len(cached_records) == len(uncached_records)
    for cached, uncached in zip(cached_records, uncached_records, strict=True):
        assert cached.case_name == uncached.case_name
        assert cached.time_step == uncached.time_step
        assert cached.configured_time_step == uncached.configured_time_step
        assert cached.timestep_index == uncached.timestep_index
        assert cached.environment_input_mode == uncached.environment_input_mode
        assert (
            cached.classification.max_fractional_mass_change
            == uncached.classification.max_fractional_mass_change
        )
        assert cached.classification.label == uncached.classification.label
        assert cached.gas_unchanged == uncached.gas_unchanged
        assert (
            cached.reuses_caller_mass_transfer_buffer
            == uncached.reuses_caller_mass_transfer_buffer
        )
        assert (
            cached.mass_transfer_has_nonzero_values
            == uncached.mass_transfer_has_nonzero_values
        )
        assert (
            cached.mass_transfer_changed_from_previous_trial
            == uncached.mass_transfer_changed_from_previous_trial
        )
        npt.assert_allclose(cached.final_masses, uncached.final_masses)
        npt.assert_allclose(cached.initial_masses, uncached.initial_masses)
        npt.assert_allclose(
            cached.mass_transfer_values,
            uncached.mass_transfer_values,
        )


@pytest.mark.gpu_parity
@pytest.mark.cuda
def test_condensation_stiffness_recorded_grid_cuda_contract_parity(
    device: str,
) -> None:
    """CUDA, when available, preserves the nanometer recorded-grid contract."""
    if device == "cpu":
        pytest.skip("CUDA parity only runs on CUDA devices")
    if not cuda_available(wp):
        pytest.skip("CUDA is unavailable")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "nanometer"
    )
    records = _record_condensation_stiffness_trials(case, device=device)

    assert len(records) == len(_RECORDED_TIMESTEP_GRID_BY_CASE[case.name])
    assert [record.timestep_index for record in records] == [0, 1, 2]
    assert [record.time_step for record in records] == list(
        _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    )
    assert [record.configured_time_step for record in records] == list(
        _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    )
    assert {record.environment_input_mode for record in records} == {
        "scalar_inputs"
    }
    assert all(record.reuses_caller_mass_transfer_buffer for record in records)
    for record in records:
        npt.assert_allclose(
            record.initial_gas_concentration - record.final_gas_concentration,
            np.sum(
                record.mass_transfer_values
                * case.build_particle_data().concentration[:, :, None],
                axis=1,
            ),
            rtol=1e-12,
            atol=1e-22,
        )
    assert all(record.case_name == case.name for record in records)
    assert all(record.classification.label == "stable" for record in records)
    assert all(record.classification.mass_nonnegative for record in records)
    assert all(record.classification.values_finite for record in records)
    assert all(
        record.classification.zero_mass_change_stable for record in records
    )
    assert all(record.classification.particle_only_update for record in records)
    assert all(
        record.classification.threshold == _RECORDED_STIFFNESS_THRESHOLD
        for record in records
    )
    assert all(record.mass_transfer_has_nonzero_values for record in records)
    assert all(np.all(np.isfinite(record.final_masses)) for record in records)
    assert all(np.all(record.final_masses >= 0.0) for record in records)
    assert all(
        record.classification.max_fractional_mass_change
        == pytest.approx(_RECORDED_STIFFNESS_THRESHOLD)
        for record in records
    )
    assert any(
        record.mass_transfer_changed_from_previous_trial
        for record in records[1:]
    )


def test_condensation_stiffness_invalid_environment_inputs_do_not_mutate_case(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Pre-launch failures leave deterministic case inputs unchanged."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "droplet_like"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    original_masses = particles.masses.copy()
    original_gas = gas.concentration.copy()
    original_vapor_pressure = vapor_pressure.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match="temperature must be finite and > 0"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=wp.array(
                [case.temperature, 0.0],
                dtype=wp.float64,
                device=device,
            ),
            pressure=wp.array(
                case.pressure_array(),
                dtype=wp.float64,
                device=device,
            ),
            time_step=case.time_step,
        )

    assert calls == []
    npt.assert_allclose(particles.masses, original_masses)
    npt.assert_allclose(gas.concentration, original_gas)
    npt.assert_allclose(vapor_pressure, original_vapor_pressure)


def test_condensation_step_gpu_reuses_mass_transfer_buffer(
    device: str,
) -> None:
    """Preallocated mass transfer buffer is reused."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.zeros(
        (1, 2, 2),
        dtype=wp.float64,
        device=device,
    )
    _, returned_buffer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        mass_transfer=mass_transfer,
    )
    assert returned_buffer is mass_transfer
    assert np.any(returned_buffer.numpy() != 0.0)


def test_rejected_inputs_leave_mass_transfer_buffer_unchanged(
    device: str,
) -> None:
    """Rejected inputs do not mutate caller-owned mass-transfer buffers."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.full(
        (1, 2, 2),
        wp.float64(7.0),
        dtype=wp.float64,
        device=device,
    )
    original = np.asarray(mass_transfer.numpy()).copy()
    bad_temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=bad_temperature,
            pressure=101325.0,
            time_step=1.0,
            mass_transfer=mass_transfer,
        )

    npt.assert_allclose(mass_transfer.numpy(), original)


def test_condensation_step_gpu_rejects_mismatched_mass_transfer_shape(
    device: str,
) -> None:
    """Mismatched mass transfer shape raises ValueError."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.zeros(
        (1, 2, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="mass_transfer shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            mass_transfer=mass_transfer,
        )


def test_validate_species_array_rejects_length_mismatch(device: str) -> None:
    """Validation helper rejects arrays with wrong length."""
    array = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="length 3 does not match n_species"):
        _validate_species_array("surface_tension", array, 2, array.device)


def test_validate_species_array_rejects_device_mismatch(device: str) -> None:
    """Validation helper rejects arrays on a different device."""
    array = wp.zeros(2, dtype=wp.float64, device=device)
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")
    with pytest.raises(ValueError, match="device does not match particle"):
        _validate_species_array(
            "surface_tension",
            array,
            2,
            wp.get_device(wrong_device),
        )


def test_validate_species_array_rejects_rank_mismatch(device: str) -> None:
    """Validation helper rejects arrays with more than one dimension."""
    array = wp.zeros((1, 2), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="must be a 1D array"):
        _validate_species_array("surface_tension", array, 2, array.device)


def test_validate_mass_transfer_buffer_rejects_shape(device: str) -> None:
    """Validation helper rejects mass transfer buffers with bad shape."""
    buffer = wp.zeros((1, 2, 3), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="mass_transfer shape"):
        _validate_mass_transfer_buffer(buffer, (1, 2, 2), buffer.device)


def test_validate_mass_transfer_buffer_rejects_device(device: str) -> None:
    """Validation helper rejects mass transfer buffers on wrong device."""
    buffer = wp.zeros((1, 2, 2), dtype=wp.float64, device=device)
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")
    with pytest.raises(ValueError, match="buffer device does not match"):
        _validate_mass_transfer_buffer(
            buffer,
            (1, 2, 2),
            wp.get_device(wrong_device),
        )


def test_condensation_validation_helpers_accept_valid_inputs(
    device: str,
) -> None:
    """Validation helpers accept correctly shaped on-device buffers."""
    species_array = wp.zeros(2, dtype=wp.float64, device=device)
    mass_transfer = wp.zeros((1, 2, 2), dtype=wp.float64, device=device)

    _validate_species_array(
        "surface_tension",
        species_array,
        2,
        species_array.device,
    )
    _validate_mass_transfer_buffer(
        mass_transfer,
        (1, 2, 2),
        mass_transfer.device,
    )


def test_condensation_step_gpu_rejects_particle_length_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle arrays with incorrect lengths."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    bad_density = wp.zeros(3, dtype=wp.float64, device=device)
    gpu_particles.density = bad_density

    with pytest.raises(ValueError, match="particle density length"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_zero_volume_particle_short_circuits_with_arrays(
    device: str,
) -> None:
    """Zero-volume particles remain unchanged with per-box array inputs."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    particles.masses[1, 0, :] = 0.0
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    gpu_result, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array([298.15, 304.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 99500.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )

    assert gpu_result.masses[1, 0, 0] == pytest.approx(0.0)
    assert mass_transfer.numpy()[1, 0, 0] == pytest.approx(0.0)


def test_condensation_step_gpu_rejects_particle_concentration_shape(
    device: str,
) -> None:
    """Condensation rejects particle concentration shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_particles.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="particle concentration shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_molar_mass_length(
    device: str,
) -> None:
    """Condensation rejects gas molar mass length mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.molar_mass = wp.zeros(3, dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match="n_species mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_concentration_shape(
    device: str,
) -> None:
    """Condensation rejects gas concentration shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="gas concentration shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_vapor_pressure_shape(
    device: str,
) -> None:
    """Condensation rejects vapor pressure shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.vapor_pressure = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="vapor pressure shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects gas arrays on a different device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.molar_mass = wp.zeros(2, dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="gas molar mass device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_particle_concentration_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle concentration on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.concentration = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(
        ValueError,
        match="particle concentration device mismatch",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_particle_density_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle density on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.density = wp.zeros(2, dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="particle density device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_concentration_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects gas concentration on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.concentration = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="gas concentration device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_vapor_pressure_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects vapor pressure on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.vapor_pressure = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="gas vapor pressure device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


@pytest.mark.gpu_parity
def test_particle_radius_from_volume_wp_matches_numpy(device: str) -> None:
    """Warp helper for radius matches NumPy calculation."""
    volumes = np.array([1.0e-18, 8.0e-18], dtype=np.float64)
    expected = np.cbrt(3.0 * volumes / (4.0 * np.pi))
    volumes_wp = wp.array(volumes, dtype=wp.float64, device=device)
    radii_wp = wp.zeros(len(volumes), dtype=wp.float64, device=device)

    @wp.kernel
    def _radius_kernel(
        total_volume: Any,
        radii_out: Any,
    ) -> None:
        idx = wp.tid()
        radii_out[idx] = particle_radius_from_volume_wp(total_volume[idx])

    wp.launch(
        _radius_kernel,
        dim=len(volumes),
        inputs=[volumes_wp, radii_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(radii_wp.numpy(), expected, rtol=1.0e-8)
