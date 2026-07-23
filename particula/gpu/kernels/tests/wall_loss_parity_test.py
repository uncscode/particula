"""Parity and statistical validation for direct neutral and charged GPU wall loss."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp

wp: Any = None
try:
    import warp as wp
except ImportError:
    pass


def _warp() -> Any:
    """Import Warp at runtime so deselected tests remain collection-safe."""
    return pytest.importorskip("warp")


def _available_warp_devices() -> list[Any]:
    """Return CPU plus an optional CUDA parameter without a missing-CUDA row."""
    if wp is None:
        return ["cpu"]
    from particula.gpu.tests.cuda_availability import warp_devices

    return [
        pytest.param(candidate, marks=pytest.mark.cuda)
        if candidate == "cuda"
        else candidate
        for candidate in warp_devices(wp)
    ]


@pytest.fixture(params=_available_warp_devices())
def device(request: pytest.FixtureRequest) -> str:
    """Provide each available Warp device."""
    return request.param


def _config(geometry: str):
    """Construct a concrete-module-only neutral wall-loss configuration."""
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    if geometry == "spherical":
        return NeutralWallLossConfig("spherical", 1.0e-3, chamber_radius=0.75)
    return NeutralWallLossConfig(
        "rectangular", 1.0e-3, chamber_dimensions=(1.2, 0.8, 0.6)
    )


def _charged_config(geometry: str, device: str, potential: float, field: float):
    """Construct charged geometry with an explicitly caller-owned field."""
    runtime_wp = _warp()
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    if geometry == "spherical":
        return NeutralWallLossConfig(
            "spherical",
            1.0e-3,
            chamber_radius=0.75,
            mode="charged",
            wall_potential=potential,
            wall_electric_field=field,
        )
    return NeutralWallLossConfig(
        "rectangular",
        1.0e-3,
        chamber_dimensions=(1.2, 0.8, 0.6),
        mode="charged",
        wall_potential=potential,
        wall_electric_field=runtime_wp.array(
            [field, -2.0 * field, 0.5 * field],
            dtype=runtime_wp.float64,
            device=device,
        ),
    )


def _masses_for_radius(radius: float) -> np.ndarray:
    """Return two-species masses whose density-weighted volume has radius."""
    volume = 4.0 * np.pi * radius**3 / 3.0
    return np.array([0.4 * volume * 1000.0, 0.6 * volume * 1500.0])


def _particles(device: str, n_boxes: int = 2):
    """Build explicit fp64 complete, inactive, and unusable fixed slots."""
    runtime_wp = _warp()
    from particula.gpu import WarpParticleData

    masses = np.zeros((n_boxes, 4, 2), dtype=np.float64)
    for box in range(n_boxes):
        masses[box, 0] = _masses_for_radius(20.0e-9)
        masses[box, 1] = _masses_for_radius(2.0e-6)
        masses[box, 3, 0] = np.nextafter(0.0, 1.0)
    particles = WarpParticleData()
    particles.masses = runtime_wp.array(
        masses, dtype=runtime_wp.float64, device=device
    )
    particles.concentration = runtime_wp.array(
        np.tile(np.array([1.0, 1.0, 0.0, 1.0]), (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.charge = runtime_wp.array(
        np.tile(np.array([1.0, -2.0, 3.0, 4.0]), (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.density = runtime_wp.array(
        [1000.0, 1500.0], dtype=runtime_wp.float64, device=device
    )
    particles.volume = runtime_wp.array(
        np.ones(n_boxes), dtype=runtime_wp.float64, device=device
    )
    return particles


def _one_species_particles(device: str, n_boxes: int = 2):
    """Build one-species fp64 complete, inactive, and unusable slots."""
    runtime_wp = _warp()
    from particula.gpu import WarpParticleData

    masses = np.zeros((n_boxes, 4, 1), dtype=np.float64)
    for box in range(n_boxes):
        masses[box, 0, 0] = _masses_for_radius(20.0e-9).sum()
        masses[box, 1, 0] = _masses_for_radius(2.0e-6).sum()
        masses[box, 3, 0] = np.nextafter(0.0, 1.0)
    particles = WarpParticleData()
    particles.masses = runtime_wp.array(
        masses, dtype=runtime_wp.float64, device=device
    )
    particles.concentration = runtime_wp.array(
        np.tile(np.array([1.0, 1.0, 0.0, 1.0]), (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.charge = runtime_wp.array(
        np.tile(np.array([1.0, -2.0, 3.0, 4.0]), (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.density = runtime_wp.array(
        [1000.0], dtype=runtime_wp.float64, device=device
    )
    particles.volume = runtime_wp.array(
        np.ones(n_boxes), dtype=runtime_wp.float64, device=device
    )
    return particles


@pytest.fixture(params=[_particles, _one_species_particles])
def particle_factory(request: pytest.FixtureRequest) -> Any:
    """Provide multi- and one-species complete-slot parity fixtures."""
    return request.param


def _environment(device: str, n_boxes: int) -> tuple[Any, Any]:
    """Return explicit per-box float64 temperature and pressure arrays."""
    runtime_wp = _warp()
    return (
        runtime_wp.array(
            np.linspace(285.0, 310.0, n_boxes),
            dtype=runtime_wp.float64,
            device=device,
        ),
        runtime_wp.array(
            np.linspace(95000.0, 101325.0, n_boxes),
            dtype=runtime_wp.float64,
            device=device,
        ),
    )


def _cpu_oracle(
    particles: Any, temperature: np.ndarray, pressure: np.ndarray, config: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate eligible complete-slot rates without importing GPU helpers."""
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_rectangle_wall_loss_coefficient_via_system_state,
        get_spherical_wall_loss_coefficient_via_system_state,
    )

    masses = particles.masses.numpy()
    concentration = particles.concentration.numpy()
    density = particles.density.numpy()
    eligible = np.zeros(concentration.shape, dtype=bool)
    coefficients = np.zeros(concentration.shape, dtype=np.float64)
    for box, particle in np.ndindex(concentration.shape):
        total_mass = np.sum(masses[box, particle])
        total_volume = np.sum(masses[box, particle] / density)
        if (
            concentration[box, particle] <= 0
            or total_mass <= 0
            or total_volume <= 0
        ):
            continue
        radius = (3.0 * total_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        effective_density = total_mass / total_volume
        if not (
            np.isfinite(radius)
            and radius > 0
            and np.isfinite(effective_density)
            and effective_density > 0
        ):
            continue
        if config.geometry == "spherical":
            coefficient = get_spherical_wall_loss_coefficient_via_system_state(
                config.wall_eddy_diffusivity,
                radius,
                effective_density,
                float(temperature[box]),
                float(pressure[box]),
                float(config.chamber_radius),
            )
        else:
            coefficient = get_rectangle_wall_loss_coefficient_via_system_state(
                config.wall_eddy_diffusivity,
                radius,
                effective_density,
                float(temperature[box]),
                float(pressure[box]),
                cast(tuple[float, float, float], config.chamber_dimensions),
            )
        if coefficient > 0 and not np.isnan(coefficient):
            eligible[box, particle] = True
            coefficients[box, particle] = coefficient
    return eligible, coefficients


def _charged_cpu_oracle(
    particles: Any, temperature: np.ndarray, pressure: np.ndarray, config: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate charged rates with the independent CPU strategy authority."""
    from particula.dynamics.wall_loss import ChargedWallLossStrategy

    masses = particles.masses.numpy()
    concentration = particles.concentration.numpy()
    charge = particles.charge.numpy()
    density = particles.density.numpy()
    field = config.wall_electric_field
    field_value: float | tuple[float, float, float]
    if config.geometry == "rectangular":
        field_value = tuple(float(value) for value in field.numpy())
    else:
        field_value = float(field)
    strategy = ChargedWallLossStrategy(
        config.wall_eddy_diffusivity,
        config.geometry,
        chamber_radius=config.chamber_radius,
        chamber_dimensions=config.chamber_dimensions,
        wall_potential=config.wall_potential,
        wall_electric_field=field_value,
        distribution_type="particle_resolved",
    )
    eligible = np.zeros(concentration.shape, dtype=bool)
    coefficients = np.zeros(concentration.shape, dtype=np.float64)
    for box in range(concentration.shape[0]):
        total_mass = masses[box].sum(axis=1)
        total_volume = (masses[box] / density).sum(axis=1)
        usable = (
            (concentration[box] > 0.0)
            & (total_mass > 0.0)
            & (total_volume > 0.0)
        )
        radius = np.zeros_like(total_mass)
        particle_density = np.zeros_like(total_mass)
        radius[usable] = (3.0 * total_volume[usable] / (4.0 * np.pi)) ** (
            1.0 / 3.0
        )
        particle_density[usable] = total_mass[usable] / total_volume[usable]
        usable &= np.isfinite(radius) & np.isfinite(particle_density)
        usable &= (radius > 0.0) & (particle_density > 0.0)
        if np.any(usable):
            rates = strategy.compute_coefficient_from_arrays(
                radius[usable],
                particle_density[usable],
                charge[box, usable],
                float(temperature[box]),
                float(pressure[box]),
            )
            valid = np.isfinite(rates) & (rates > 0.0)
            positions = np.flatnonzero(usable)
            eligible[box, positions[valid]] = True
            coefficients[box, positions[valid]] = rates[valid]
    return eligible, coefficients


if wp is not None:
    from particula.gpu.dynamics.wall_loss_funcs import (
        _combine_charged_wall_loss_coefficient_wp,
        _electric_field_drift_wp,
        _geometry_scale_wp,
        _image_charge_enhancement_wp,
        _resolve_rectangular_electric_field_wp,
        _resolve_spherical_electric_field_wp,
        rectangle_wall_loss_coefficient_wp,
        spherical_wall_loss_coefficient_wp,
    )
    from particula.gpu.properties import (
        effective_density_wp,
        particle_radius_from_volume_wp,
    )
    from particula.util.constants import (
        BOLTZMANN_CONSTANT,
        GAS_CONSTANT,
        MOLECULAR_WEIGHT_AIR,
        REF_TEMPERATURE_STP,
        REF_VISCOSITY_AIR_STP,
        SUTHERLAND_CONSTANT,
    )

    _BOLTZMANN_CONSTANT = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
    _GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
    _MOLECULAR_WEIGHT_AIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
    _REF_TEMPERATURE_STP = wp.constant(wp.float64(REF_TEMPERATURE_STP))
    _REF_VISCOSITY_AIR_STP = wp.constant(wp.float64(REF_VISCOSITY_AIR_STP))
    _SUTHERLAND_CONSTANT = wp.constant(wp.float64(SUTHERLAND_CONSTANT))
    from particula.util.constants import (
        ELECTRIC_PERMITTIVITY,
        ELEMENTARY_CHARGE_VALUE,
    )

    _ELEMENTARY_CHARGE_VALUE = wp.constant(wp.float64(ELEMENTARY_CHARGE_VALUE))
    _ELECTRIC_PERMITTIVITY = wp.constant(wp.float64(ELECTRIC_PERMITTIVITY))

    @wp.kernel
    def _wall_loss_rate_diagnostic(
        masses: wp.array3d[wp.float64],
        concentration: wp.array2d[wp.float64],
        density: wp.array[wp.float64],
        temperature: wp.array[wp.float64],
        pressure: wp.array[wp.float64],
        eddy: wp.float64,
        radius: wp.float64,
        length: wp.float64,
        width: wp.float64,
        height: wp.float64,
        geometry: wp.int32,
        n_species: wp.int32,
        rates: wp.array2d[wp.float64],
        eligible: wp.array2d[wp.int32],
    ) -> None:
        """Record production-equivalent coefficients without RNG or mutation."""
        box, particle = wp.tid()
        if concentration[box, particle] <= 0.0:
            return
        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species in range(n_species):
            mass = masses[box, particle, species]
            total_mass += mass
            total_volume += mass / density[species]
        if total_mass <= 0.0 or total_volume <= 0.0:
            return
        particle_radius = particle_radius_from_volume_wp(total_volume)
        particle_density = effective_density_wp(total_mass, total_volume)
        if (
            not wp.isfinite(particle_radius)
            or particle_radius <= 0.0
            or not wp.isfinite(particle_density)
            or particle_density <= 0.0
        ):
            return
        coefficient = wp.float64(0.0)
        if geometry == 0:
            coefficient = spherical_wall_loss_coefficient_wp(
                eddy,
                particle_radius,
                particle_density,
                temperature[box],
                pressure[box],
                radius,
                _BOLTZMANN_CONSTANT,
                _GAS_CONSTANT,
                _MOLECULAR_WEIGHT_AIR,
                _REF_VISCOSITY_AIR_STP,
                _REF_TEMPERATURE_STP,
                _SUTHERLAND_CONSTANT,
            )
        else:
            coefficient = rectangle_wall_loss_coefficient_wp(
                eddy,
                particle_radius,
                particle_density,
                temperature[box],
                pressure[box],
                length,
                width,
                height,
                _BOLTZMANN_CONSTANT,
                _GAS_CONSTANT,
                _MOLECULAR_WEIGHT_AIR,
                _REF_VISCOSITY_AIR_STP,
                _REF_TEMPERATURE_STP,
                _SUTHERLAND_CONSTANT,
            )
        if coefficient > 0.0 and wp.isfinite(coefficient):
            eligible[box, particle] = 1
            rates[box, particle] = coefficient

    @wp.kernel
    def _charged_wall_loss_rate_diagnostic(  # noqa: C901
        masses: wp.array3d[wp.float64],
        concentration: wp.array2d[wp.float64],
        charge: wp.array2d[wp.float64],
        density: wp.array[wp.float64],
        temperature: wp.array[wp.float64],
        pressure: wp.array[wp.float64],
        field: wp.array[wp.float64],
        eddy: wp.float64,
        radius: wp.float64,
        length: wp.float64,
        width: wp.float64,
        height: wp.float64,
        potential: wp.float64,
        geometry: wp.int32,
        n_species: wp.int32,
        rates: wp.array2d[wp.float64],
        eligible: wp.array2d[wp.int32],
    ) -> None:
        """Record charged coefficients without consuming RNG or mutating slots."""
        box, particle = wp.tid()
        if concentration[box, particle] <= 0.0:
            return
        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species in range(n_species):
            total_mass += masses[box, particle, species]
            total_volume += masses[box, particle, species] / density[species]
        if total_mass <= 0.0 or total_volume <= 0.0:
            return
        particle_radius = particle_radius_from_volume_wp(total_volume)
        particle_density = effective_density_wp(total_mass, total_volume)
        if particle_radius <= 0.0 or particle_density <= 0.0:
            return
        neutral = wp.float64(0.0)
        scale = _geometry_scale_wp(geometry, radius, length, width, height)
        resolved_field = wp.float64(0.0)
        if geometry == 0:
            neutral = spherical_wall_loss_coefficient_wp(
                eddy,
                particle_radius,
                particle_density,
                temperature[box],
                pressure[box],
                radius,
                _BOLTZMANN_CONSTANT,
                _GAS_CONSTANT,
                _MOLECULAR_WEIGHT_AIR,
                _REF_VISCOSITY_AIR_STP,
                _REF_TEMPERATURE_STP,
                _SUTHERLAND_CONSTANT,
            )
            resolved_field = _resolve_spherical_electric_field_wp(
                field[0], potential, scale
            )
        else:
            neutral = rectangle_wall_loss_coefficient_wp(
                eddy,
                particle_radius,
                particle_density,
                temperature[box],
                pressure[box],
                length,
                width,
                height,
                _BOLTZMANN_CONSTANT,
                _GAS_CONSTANT,
                _MOLECULAR_WEIGHT_AIR,
                _REF_VISCOSITY_AIR_STP,
                _REF_TEMPERATURE_STP,
                _SUTHERLAND_CONSTANT,
            )
            resolved_field = _resolve_rectangular_electric_field_wp(
                field[0], field[1], field[2], potential, scale
            )
        coefficient = neutral
        if charge[box, particle] != 0.0:
            enhancement = _image_charge_enhancement_wp(
                particle_radius,
                charge[box, particle],
                temperature[box],
                _ELEMENTARY_CHARGE_VALUE,
                _ELECTRIC_PERMITTIVITY,
                _BOLTZMANN_CONSTANT,
            )
            drift = _electric_field_drift_wp(
                particle_radius,
                charge[box, particle],
                temperature[box],
                resolved_field,
                scale,
                _ELEMENTARY_CHARGE_VALUE,
                _REF_VISCOSITY_AIR_STP,
                _REF_TEMPERATURE_STP,
                _SUTHERLAND_CONSTANT,
            )
            coefficient = _combine_charged_wall_loss_coefficient_wp(
                neutral, enhancement, drift
            )
        if coefficient > 0.0 and not wp.isnan(coefficient):
            eligible[box, particle] = 1
            rates[box, particle] = coefficient


@pytest.mark.gpu_parity
@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
@pytest.mark.parametrize("n_boxes", [1, 2])
def test_complete_slot_rates_match_independent_cpu_oracle(
    device: str, geometry: str, n_boxes: int, particle_factory: Any
) -> None:
    """Compare non-mutating Warp complete-slot rates against CPU equations."""
    runtime_wp = _warp()
    particles = particle_factory(device, n_boxes)
    temperature, pressure = _environment(device, n_boxes)
    config = _config(geometry)
    expected_mask, expected_rates = _cpu_oracle(
        particles, temperature.numpy(), pressure.numpy(), config
    )
    before = (
        particles.masses.numpy().copy(),
        particles.concentration.numpy().copy(),
        particles.charge.numpy().copy(),
    )
    rates = runtime_wp.zeros(
        expected_mask.shape, dtype=runtime_wp.float64, device=device
    )
    actual_mask = runtime_wp.zeros(
        expected_mask.shape, dtype=runtime_wp.int32, device=device
    )
    dimensions = config.chamber_dimensions or (0.0, 0.0, 0.0)
    runtime_wp.launch(
        _wall_loss_rate_diagnostic,
        dim=expected_mask.shape,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            temperature,
            pressure,
            config.wall_eddy_diffusivity,
            config.chamber_radius or 0.0,
            *dimensions,
            0 if geometry == "spherical" else 1,
            particles.masses.shape[2],
            rates,
            actual_mask,
        ],
        device=device,
    )
    runtime_wp.synchronize()
    npt.assert_array_equal(actual_mask.numpy().astype(bool), expected_mask)
    assert np.all(np.isfinite(rates.numpy()[expected_mask]))
    # CPU Debye integration omits its zero-endpoint trapezoid interval.
    tolerance = 1.002e-3 if geometry == "spherical" else 1.0e-10
    npt.assert_allclose(
        rates.numpy()[expected_mask],
        expected_rates[expected_mask],
        rtol=tolerance,
        atol=1.0e-20,
    )
    assert not np.any(expected_mask[:, 2:])
    npt.assert_array_equal(particles.masses.numpy(), before[0])
    npt.assert_array_equal(particles.concentration.numpy(), before[1])
    npt.assert_array_equal(particles.charge.numpy(), before[2])


def _charged_matrix_particles(device: str, n_boxes: int) -> Any:
    """Build four physical scales plus inactive and unusable protected slots."""
    runtime_wp = _warp()
    from particula.gpu import WarpParticleData

    radii = (2.0e-9, 50.0e-9, 1.0e-6, 50.0e-6)
    masses = np.zeros((n_boxes, 7, 2), dtype=np.float64)
    for box in range(n_boxes):
        for particle, particle_radius in enumerate(radii):
            masses[box, particle] = _masses_for_radius(particle_radius)
        masses[box, 6, 0] = np.nextafter(0.0, 1.0)
    particles = WarpParticleData()
    particles.masses = runtime_wp.array(
        masses, dtype=runtime_wp.float64, device=device
    )
    particles.concentration = runtime_wp.array(
        np.tile([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.charge = runtime_wp.array(
        np.tile([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0], (n_boxes, 1)),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.density = runtime_wp.array(
        [1000.0, 1500.0], dtype=runtime_wp.float64, device=device
    )
    particles.volume = runtime_wp.array(
        np.ones(n_boxes), dtype=runtime_wp.float64, device=device
    )
    return particles


def _charged_diagnostic(
    particles: Any, temperature: Any, pressure: Any, config: Any, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run the non-mutating charged Warp diagnostic and return bulk results."""
    runtime_wp = _warp()
    shape = particles.concentration.shape
    rates = runtime_wp.zeros(shape, dtype=runtime_wp.float64, device=device)
    eligible = runtime_wp.zeros(shape, dtype=runtime_wp.int32, device=device)
    dimensions = config.chamber_dimensions or (0.0, 0.0, 0.0)
    if config.geometry == "spherical":
        field = runtime_wp.array(
            [config.wall_electric_field, 0.0, 0.0],
            dtype=runtime_wp.float64,
            device=device,
        )
    else:
        field = config.wall_electric_field
    runtime_wp.launch(
        _charged_wall_loss_rate_diagnostic,
        dim=shape,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            temperature,
            pressure,
            field,
            config.wall_eddy_diffusivity,
            config.chamber_radius or 0.0,
            *dimensions,
            config.wall_potential,
            0 if config.geometry == "spherical" else 1,
            particles.masses.shape[2],
            rates,
            eligible,
        ],
        device=device,
    )
    runtime_wp.synchronize()
    return eligible.numpy().astype(bool), rates.numpy()


def _neutral_diagnostic(
    particles: Any, temperature: Any, pressure: Any, config: Any, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run the non-mutating neutral Warp diagnostic and return bulk results."""
    runtime_wp = _warp()
    shape = particles.concentration.shape
    rates = runtime_wp.zeros(shape, dtype=runtime_wp.float64, device=device)
    eligible = runtime_wp.zeros(shape, dtype=runtime_wp.int32, device=device)
    dimensions = config.chamber_dimensions or (0.0, 0.0, 0.0)
    runtime_wp.launch(
        _wall_loss_rate_diagnostic,
        dim=shape,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            temperature,
            pressure,
            config.wall_eddy_diffusivity,
            config.chamber_radius or 0.0,
            *dimensions,
            0 if config.geometry == "spherical" else 1,
            particles.masses.shape[2],
            rates,
            eligible,
        ],
        device=device,
    )
    runtime_wp.synchronize()
    return eligible.numpy().astype(bool), rates.numpy()


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("geometry", "n_boxes", "potential", "field"),
    [
        ("spherical", 1, 0.0, 0.0),
        ("spherical", 2, -5.0, 3.0),
        ("rectangular", 1, 4.0, -2.0),
        ("rectangular", 2, -3.0, 2.5),
    ],
)
def test_charged_coefficients_match_independent_cpu_oracle(
    device: str, geometry: str, n_boxes: int, potential: float, field: float
) -> None:
    """Charged diagnostics match CPU strategy rates without changing owners."""
    particles = _charged_matrix_particles(device, n_boxes)
    temperature, pressure = _environment(device, n_boxes)
    config = _charged_config(geometry, device, potential, field)
    expected_mask, expected_rates = _charged_cpu_oracle(
        particles, temperature.numpy(), pressure.numpy(), config
    )
    particle_before = {
        name: getattr(particles, name).numpy().copy()
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    field_before = (
        config.wall_electric_field.numpy().copy()
        if geometry == "rectangular"
        else None
    )
    actual_mask, actual_rates = _charged_diagnostic(
        particles, temperature, pressure, config, device
    )
    npt.assert_array_equal(actual_mask, expected_mask)
    tolerance = 1.002e-3 if geometry == "spherical" else 1.0e-6
    npt.assert_allclose(
        expected_rates[expected_mask],
        actual_rates[expected_mask],
        rtol=tolerance,
        atol=1.0e-20 if geometry == "spherical" else 0.0,
    )
    for name, values in particle_before.items():
        npt.assert_array_equal(getattr(particles, name).numpy(), values)
    if field_before is not None:
        npt.assert_array_equal(config.wall_electric_field.numpy(), field_before)


@pytest.mark.gpu_parity
@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_zero_charge_charged_mode_exactly_matches_neutral_diagnostics_and_state(
    device: str, geometry: str
) -> None:
    """Configured charged fields retain exact neutral coefficients and draws."""
    runtime_wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    neutral_particles = _charged_matrix_particles(device, 2)
    charged_particles = _charged_matrix_particles(device, 2)
    zeros = np.zeros(neutral_particles.charge.shape, dtype=np.float64)
    for particles in (neutral_particles, charged_particles):
        particles.charge = runtime_wp.array(
            zeros, dtype=runtime_wp.float64, device=device
        )
    temperature, pressure = _environment(device, 2)
    neutral_config = _config(geometry)
    charged_config = _charged_config(
        geometry, device, potential=-9.0, field=4.0
    )
    neutral_mask, neutral_rates = _neutral_diagnostic(
        neutral_particles, temperature, pressure, neutral_config, device
    )
    charged_mask, charged_rates = _charged_diagnostic(
        charged_particles, temperature, pressure, charged_config, device
    )
    npt.assert_array_equal(charged_mask, neutral_mask)
    npt.assert_array_equal(charged_rates, neutral_rates)
    neutral_rng = runtime_wp.array(
        [17, 31], dtype=runtime_wp.uint32, device=device
    )
    charged_rng = runtime_wp.array(
        [17, 31], dtype=runtime_wp.uint32, device=device
    )
    field_before = (
        charged_config.wall_electric_field.numpy().copy()
        if geometry == "rectangular"
        else None
    )
    wall_loss_step_gpu(
        neutral_particles,
        temperature,
        pressure,
        1.0,
        config=neutral_config,
        rng_states=neutral_rng,
    )
    wall_loss_step_gpu(
        charged_particles,
        temperature,
        pressure,
        1.0,
        config=charged_config,
        rng_states=charged_rng,
    )
    for name in ("masses", "concentration", "charge", "density", "volume"):
        npt.assert_array_equal(
            getattr(charged_particles, name).numpy(),
            getattr(neutral_particles, name).numpy(),
        )
    npt.assert_array_equal(charged_rng.numpy(), neutral_rng.numpy())
    if field_before is not None:
        npt.assert_array_equal(
            charged_config.wall_electric_field.numpy(), field_before
        )


def _homogeneous_particles(device: str, active: int = 32):
    """Create one-box active trials followed by two deliberately inactive slots."""
    runtime_wp = _warp()
    particles = _particles(device, 1)
    masses = np.zeros((1, active + 2, 2), dtype=np.float64)
    masses[0, :active] = _masses_for_radius(200.0e-9)
    particles.masses = runtime_wp.array(
        masses, dtype=runtime_wp.float64, device=device
    )
    particles.concentration = runtime_wp.array(
        np.array([[*np.ones(active), 0.0, 1.0]]),
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.charge = runtime_wp.array(
        np.arange(active + 2, dtype=np.float64)[None, :],
        dtype=runtime_wp.float64,
        device=device,
    )
    return particles


def _survival_time(config: Any, device: str) -> float:
    """Choose a timestep yielding an interior half-survival probability."""
    particles = _homogeneous_particles(device)
    mask, rates = _cpu_oracle(
        particles, np.array([298.15]), np.array([101325.0]), config
    )
    assert mask[0, 0] and np.isfinite(rates[0, 0]) and rates[0, 0] > 0
    return float(-np.log(0.5) / rates[0, 0])


def _inclusive_binomial_interval(
    n: int, probability: float, alpha: float
) -> tuple[int, int]:
    """Return inclusive equal-tail exact-binomial acceptance bounds."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    from scipy.stats import binom

    lower = int(np.ceil(binom.ppf(alpha / 2.0, n, probability)))
    upper = int(np.floor(binom.isf(alpha / 2.0, n, probability)))
    return lower, upper


def _is_within_inclusive_binomial_interval(
    observed: int, n: int, probability: float, alpha: float
) -> bool:
    """Return whether an observed count falls inside the exact interval."""
    lower, upper = _inclusive_binomial_interval(n, probability, alpha)
    return lower <= observed <= upper


def _charged_homogeneous_particles(
    device: str, radius: float, charge: float, active: int
) -> Any:
    """Create homogeneous charged trials and two protected inactive sentinels."""
    runtime_wp = _warp()
    from particula.gpu import WarpParticleData

    particles = WarpParticleData()
    masses = np.zeros((1, active + 2, 2), dtype=np.float64)
    masses[0, :active] = _masses_for_radius(radius)
    particles.masses = runtime_wp.array(
        masses, dtype=runtime_wp.float64, device=device
    )
    particles.concentration = runtime_wp.array(
        [[*np.ones(active), 0.0, 1.0]], dtype=runtime_wp.float64, device=device
    )
    particles.charge = runtime_wp.array(
        [[*np.full(active, charge), 13.0, -17.0]],
        dtype=runtime_wp.float64,
        device=device,
    )
    particles.density = runtime_wp.array(
        [1000.0, 1500.0], dtype=runtime_wp.float64, device=device
    )
    particles.volume = runtime_wp.array(
        [1.0], dtype=runtime_wp.float64, device=device
    )
    return particles


def _charged_half_survival_time(
    config: Any, device: str, radius: float
) -> float:
    """Derive one charged timestep targeting a 0.5 survival probability."""
    particles = _charged_homogeneous_particles(device, radius, 2.0, active=1)
    mask, rates = _charged_cpu_oracle(
        particles, np.array([298.15]), np.array([101325.0]), config
    )
    assert mask[0, 0] and rates[0, 0] > 0.0
    return float(-np.log(0.5) / rates[0, 0])


@pytest.mark.parametrize(
    ("n", "probability", "alpha", "message"),
    [
        (0, 0.5, 0.1, "n must be positive"),
        (1, -0.1, 0.1, "probability must be in"),
        (1, 1.1, 0.1, "probability must be in"),
        (1, 0.5, 0.0, "alpha must be in"),
        (1, 0.5, 1.0, "alpha must be in"),
    ],
)
def test_inclusive_binomial_interval_rejects_invalid_parameters(
    n: int, probability: float, alpha: float, message: str
) -> None:
    """Reject invalid exact-binomial helper parameters with stable messages."""
    with pytest.raises(ValueError, match=message):
        _inclusive_binomial_interval(n, probability, alpha)


def test_inclusive_binomial_interval_accepts_exact_endpoints_only() -> None:
    """Inclusive interval endpoints pass while adjacent counts fail."""
    lower, upper = _inclusive_binomial_interval(4096, 0.5, 1.25e-7)
    assert _is_within_inclusive_binomial_interval(lower, 4096, 0.5, 1.25e-7)
    assert _is_within_inclusive_binomial_interval(upper, 4096, 0.5, 1.25e-7)
    assert not _is_within_inclusive_binomial_interval(
        lower - 1, 4096, 0.5, 1.25e-7
    )
    assert not _is_within_inclusive_binomial_interval(
        upper + 1, 4096, 0.5, 1.25e-7
    )


@pytest.mark.gpu_parity
@pytest.mark.stochastic
@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
@pytest.mark.parametrize("radius", [2.0e-9, 50.0e-9, 1.0e-6, 50.0e-6])
def test_charged_survival_matches_frozen_exact_binomial_interval(
    device: str, geometry: str, radius: float
) -> None:
    """Validate each charged geometry-radius stratum from 4,096 observations."""
    from particula.gpu.kernels import wall_loss_step_gpu

    config = _charged_config(
        geometry,
        device,
        potential=0.0 if geometry == "spherical" else 2.0,
        field=0.0 if geometry == "spherical" else 3.0,
    )
    time_step = _charged_half_survival_time(config, device, radius)
    active, fixtures = 256, 16
    observed = 0
    for seed in range(fixtures):
        particles = _charged_homogeneous_particles(device, radius, 2.0, active)
        sentinels = tuple(
            getattr(particles, name).numpy()[0, active:].copy()
            for name in ("masses", "concentration", "charge")
        )
        wall_loss_step_gpu(
            particles, 298.15, 101325.0, time_step, config=config, rng_seed=seed
        )
        observed += int(
            np.count_nonzero(particles.concentration.numpy()[0, :active])
        )
        for name, expected in zip(
            ("masses", "concentration", "charge"), sentinels, strict=True
        ):
            npt.assert_array_equal(
                getattr(particles, name).numpy()[0, active:], expected
            )
    assert _is_within_inclusive_binomial_interval(
        observed, active * fixtures, 0.5, 1.25e-7
    )


@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_charged_persistent_sidecar_survival_and_noop_lifecycle(
    device: str,
) -> None:
    """Validate persistent charged state without reset and exact no-op paths."""
    runtime_wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    radius, active, steps = 50.0e-9, 128, 3
    config = _charged_config("spherical", device, potential=0.0, field=0.0)
    time_step = _charged_half_survival_time(config, device, radius)
    particles = _charged_homogeneous_particles(device, radius, 2.0, active)
    states = runtime_wp.zeros(1, dtype=runtime_wp.uint32, device=device)
    owner = states
    initialized_state: np.ndarray | None = None
    previous_state: np.ndarray | None = None
    for step in range(steps):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            time_step,
            config=config,
            rng_seed=73,
            rng_states=states,
            initialize_rng=step == 0,
        )
        assert states is owner
        current_state = states.numpy().copy()
        if step == 0:
            initialized_state = current_state
        else:
            assert previous_state is not None
            assert not np.array_equal(current_state, previous_state)
        previous_state = current_state
    assert initialized_state is not None
    reset_particles = _charged_homogeneous_particles(
        device, radius, 2.0, active
    )
    reset_states = runtime_wp.array(
        [91], dtype=runtime_wp.uint32, device=device
    )
    wall_loss_step_gpu(
        reset_particles,
        298.15,
        101325.0,
        time_step,
        config=config,
        rng_seed=73,
        rng_states=reset_states,
        initialize_rng=True,
    )
    npt.assert_array_equal(reset_states.numpy(), initialized_state)
    observed = int(
        np.count_nonzero(particles.concentration.numpy()[0, :active])
    )
    assert _is_within_inclusive_binomial_interval(
        observed, active, 0.5**steps, 1.25e-7
    )
    before = states.numpy().copy()
    wall_loss_step_gpu(
        particles, 298.15, 101325.0, 0.0, config=config, rng_states=states
    )
    npt.assert_array_equal(states.numpy(), before)


@pytest.mark.gpu_parity
def test_charged_opposing_field_does_not_reset_supplied_sidecar(
    device: str,
) -> None:
    """Keep a supplied sidecar unchanged when clipped charged rates draw never."""
    runtime_wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    field = runtime_wp.array(
        [1.0e100, 0.0, 0.0], dtype=runtime_wp.float64, device=device
    )
    config = NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_potential=-2.0e100,
        wall_electric_field=field,
    )
    particles = _charged_homogeneous_particles(
        device, radius=50.0e-9, charge=1.0, active=1
    )
    states = runtime_wp.array([31], dtype=runtime_wp.uint32, device=device)
    particle_before = {
        name: getattr(particles, name).numpy().copy()
        for name in ("masses", "concentration", "charge")
    }
    state_before = states.numpy().copy()

    wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0,
        config=config,
        rng_seed=73,
        rng_states=states,
        initialize_rng=True,
    )

    for name, expected in particle_before.items():
        npt.assert_array_equal(getattr(particles, name).numpy(), expected)
    npt.assert_array_equal(states.numpy(), state_before)


@pytest.mark.gpu_parity
@pytest.mark.stochastic
@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_fresh_seed_survival_matches_binomial_expectation(
    device: str, geometry: str
) -> None:
    """Validate 100 independent omitted-sidecar trials without RNG replay."""
    from particula.gpu.kernels import wall_loss_step_gpu

    active = 32
    config = _config(geometry)
    time_step = _survival_time(config, device)
    observed = 0
    for seed in range(100):
        particles = _homogeneous_particles(device, active)
        inactive = (
            particles.masses.numpy()[0, active:].copy(),
            particles.concentration.numpy()[0, active:].copy(),
            particles.charge.numpy()[0, active:].copy(),
        )
        wall_loss_step_gpu(
            particles, 298.15, 101325.0, time_step, config=config, rng_seed=seed
        )
        observed += int(
            np.count_nonzero(particles.concentration.numpy()[0, :active])
        )
        npt.assert_array_equal(
            particles.masses.numpy()[0, active:], inactive[0]
        )
        npt.assert_array_equal(
            particles.concentration.numpy()[0, active:], inactive[1]
        )
        npt.assert_array_equal(
            particles.charge.numpy()[0, active:], inactive[2]
        )
    total_trials = active * 100
    expected = total_trials * 0.5
    bound = 3.0 * np.sqrt(total_trials * 0.5 * 0.5)
    assert abs(observed - expected) <= bound


@pytest.mark.gpu_parity
@pytest.mark.stochastic
@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_persistent_sidecar_survival_and_lifecycle(
    device: str, geometry: str
) -> None:
    """Validate 100 persistent-sidecar trials and in-trial non-reset behavior."""
    runtime_wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    active, n_steps = 32, 3
    config = _config(geometry)
    time_step = _survival_time(config, device)
    observed = 0
    for seed in range(100):
        particles = _homogeneous_particles(device, active)
        states = runtime_wp.zeros(1, dtype=runtime_wp.uint32, device=device)
        owner = states
        for step in range(n_steps):
            before = states.numpy().copy()
            eligible = np.any(particles.concentration.numpy()[0, :active] > 0)
            returned = wall_loss_step_gpu(
                particles,
                298.15,
                101325.0,
                time_step,
                config=config,
                rng_seed=seed,
                rng_states=states,
                initialize_rng=step == 0,
            )
            assert returned is particles and states is owner
            if eligible:
                assert not np.array_equal(states.numpy(), before)
            else:
                npt.assert_array_equal(states.numpy(), before)
        observed += int(
            np.count_nonzero(particles.concentration.numpy()[0, :active])
        )
    total_trials = active * 100
    probability = 0.5**n_steps
    expected = total_trials * probability
    bound = 3.0 * np.sqrt(total_trials * probability * (1.0 - probability))
    assert abs(observed - expected) <= bound


@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_zero_time_and_all_inactive_are_exact_noops(device: str) -> None:
    """Keep exact write-free no-op assertions separate from aggregate evidence."""
    runtime_wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    for time_step, inactive in ((0.0, False), (1.0, True)):
        particles = _particles(device, 1)
        if inactive:
            particles.concentration = runtime_wp.zeros(
                (1, 4), dtype=runtime_wp.float64, device=device
            )
        states = runtime_wp.array([17], dtype=runtime_wp.uint32, device=device)
        particle_owner = particles
        field_names = ("masses", "concentration", "charge", "density", "volume")
        snapshot = {
            name: (
                getattr(particles, name),
                getattr(particles, name).numpy().copy(),
            )
            for name in field_names
        }
        state_owner = states
        state_snapshot = states.numpy().copy()
        returned = wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            time_step,
            config=_config("spherical"),
            rng_states=states,
        )
        assert returned is particle_owner
        assert states is state_owner
        for name in field_names:
            field_owner, expected = snapshot[name]
            assert getattr(particles, name) is field_owner
            npt.assert_array_equal(getattr(particles, name).numpy(), expected)
        npt.assert_array_equal(states.numpy(), state_snapshot)


def test_wall_loss_lazy_import_surface() -> None:
    """Keep the direct step lazy and config concrete without requiring Warp."""
    from particula.gpu import kernels

    assert "wall_loss_step_gpu" in kernels.__all__
    assert kernels._SYMBOL_TO_MODULE["wall_loss_step_gpu"] == (
        "particula.gpu.kernels.wall_loss"
    )
    assert "NeutralWallLossConfig" not in kernels.__all__
    assert not hasattr(kernels, "NeutralWallLossConfig")


def test_wall_loss_lazy_symbol_matches_concrete_step() -> None:
    """Resolve the lazy direct step only after the runtime Warp guard."""
    _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig
    from particula.gpu.kernels.wall_loss import (
        wall_loss_step_gpu as module_step,
    )

    assert wall_loss_step_gpu is module_step
    assert NeutralWallLossConfig is not None
