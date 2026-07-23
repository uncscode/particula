"""CPU parity tests for Warp neutral wall-loss coefficient helpers."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp: Any = None
try:
    import warp as wp
except ImportError:
    pass

pytestmark = (
    [pytest.mark.warp, pytest.mark.skip(reason="Warp not installed")]
    if wp is None
    else pytest.mark.warp
)

if wp is not None:
    from particula.dynamics.properties.wall_loss_coefficient import (  # noqa: E402
        get_particle_settling_velocity_via_system_state,
        get_rectangle_wall_loss_coefficient_via_system_state,
        get_spherical_wall_loss_coefficient_via_system_state,
    )
    from particula.gpu.dynamics.wall_loss_funcs import (  # noqa: E402
        _combine_charged_wall_loss_coefficient_wp,
        _coulomb_self_potential_ratio_wp,
        _electric_field_drift_wp,
        _geometry_scale_wp,
        _image_charge_enhancement_wp,
        _resolve_rectangular_electric_field_wp,
        _resolve_spherical_electric_field_wp,
        rectangle_wall_loss_coefficient_wp,
        spherical_wall_loss_coefficient_wp,
    )
    from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402
    from particula.util.constants import (  # noqa: E402
        BOLTZMANN_CONSTANT,
        ELECTRIC_PERMITTIVITY,
        ELEMENTARY_CHARGE_VALUE,
        GAS_CONSTANT,
        MOLECULAR_WEIGHT_AIR,
        REF_TEMPERATURE_STP,
        REF_VISCOSITY_AIR_STP,
        SUTHERLAND_CONSTANT,
    )


def _available_warp_devices() -> list[Any]:
    """Return collection-safe Warp device parameters."""
    if wp is None:
        return ["cpu"]
    return [
        pytest.param(candidate, marks=pytest.mark.cuda)
        if candidate == "cuda"
        else candidate
        for candidate in warp_devices(wp)
    ]


if wp is not None:

    @wp.kernel
    def _spherical_wall_loss_kernel(
        eddy_diffusivities: Any,
        particle_radii: Any,
        particle_densities: Any,
        temperatures: Any,
        pressures: Any,
        chamber_radii: Any,
        boltzmann_constant: wp.float64,
        gas_constant: wp.float64,
        molecular_weight_air: wp.float64,
        ref_viscosity: wp.float64,
        ref_temperature: wp.float64,
        sutherland_constant: wp.float64,
        result: Any,
    ) -> None:
        """Calculate one spherical wall-loss coefficient per lane."""
        tid = wp.tid()
        result[tid] = spherical_wall_loss_coefficient_wp(
            eddy_diffusivities[tid],
            particle_radii[tid],
            particle_densities[tid],
            temperatures[tid],
            pressures[tid],
            chamber_radii[tid],
            boltzmann_constant,
            gas_constant,
            molecular_weight_air,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )

    @wp.kernel
    def _rectangle_wall_loss_kernel(
        eddy_diffusivities: Any,
        particle_radii: Any,
        particle_densities: Any,
        temperatures: Any,
        pressures: Any,
        chamber_lengths: Any,
        chamber_widths: Any,
        chamber_heights: Any,
        boltzmann_constant: wp.float64,
        gas_constant: wp.float64,
        molecular_weight_air: wp.float64,
        ref_viscosity: wp.float64,
        ref_temperature: wp.float64,
        sutherland_constant: wp.float64,
        result: Any,
    ) -> None:
        """Calculate one rectangular wall-loss coefficient per lane."""
        tid = wp.tid()
        result[tid] = rectangle_wall_loss_coefficient_wp(
            eddy_diffusivities[tid],
            particle_radii[tid],
            particle_densities[tid],
            temperatures[tid],
            pressures[tid],
            chamber_lengths[tid],
            chamber_widths[tid],
            chamber_heights[tid],
            boltzmann_constant,
            gas_constant,
            molecular_weight_air,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )

    @wp.kernel
    def _coulomb_self_potential_ratio_kernel(
        particle_radii: Any,
        particle_charges: Any,
        temperatures: Any,
        elementary_charge_value: wp.float64,
        electric_permittivity: wp.float64,
        boltzmann_constant: wp.float64,
        result: Any,
    ) -> None:
        """Calculate one Coulomb self-potential ratio per lane."""
        tid = wp.tid()
        result[tid] = _coulomb_self_potential_ratio_wp(
            particle_radii[tid],
            particle_charges[tid],
            temperatures[tid],
            elementary_charge_value,
            electric_permittivity,
            boltzmann_constant,
        )

    @wp.kernel
    def _image_charge_enhancement_kernel(
        particle_radii: Any,
        particle_charges: Any,
        temperatures: Any,
        elementary_charge_value: wp.float64,
        electric_permittivity: wp.float64,
        boltzmann_constant: wp.float64,
        result: Any,
    ) -> None:
        """Calculate one image-charge enhancement factor per lane."""
        tid = wp.tid()
        result[tid] = _image_charge_enhancement_wp(
            particle_radii[tid],
            particle_charges[tid],
            temperatures[tid],
            elementary_charge_value,
            electric_permittivity,
            boltzmann_constant,
        )

    @wp.kernel
    def _resolved_electric_field_kernel(
        geometry_modes: Any,
        chamber_radii: Any,
        chamber_lengths: Any,
        chamber_widths: Any,
        chamber_heights: Any,
        field_x: Any,
        field_y: Any,
        field_z: Any,
        wall_potentials: Any,
        spherical_result: Any,
        rectangular_result: Any,
        geometry_result: Any,
    ) -> None:
        """Calculate geometry scales and both geometry-specific field forms."""
        tid = wp.tid()
        scale = _geometry_scale_wp(
            geometry_modes[tid],
            chamber_radii[tid],
            chamber_lengths[tid],
            chamber_widths[tid],
            chamber_heights[tid],
        )
        geometry_result[tid] = scale
        spherical_result[tid] = _resolve_spherical_electric_field_wp(
            field_x[tid], wall_potentials[tid], scale
        )
        rectangular_result[tid] = _resolve_rectangular_electric_field_wp(
            field_x[tid],
            field_y[tid],
            field_z[tid],
            wall_potentials[tid],
            scale,
        )

    @wp.kernel
    def _electric_field_drift_kernel(
        particle_radii: Any,
        particle_charges: Any,
        temperatures: Any,
        resolved_fields: Any,
        geometry_scales: Any,
        elementary_charge_value: wp.float64,
        ref_viscosity: wp.float64,
        ref_temperature: wp.float64,
        sutherland_constant: wp.float64,
        result: Any,
    ) -> None:
        """Calculate one electric-field drift coefficient per lane."""
        tid = wp.tid()
        result[tid] = _electric_field_drift_wp(
            particle_radii[tid],
            particle_charges[tid],
            temperatures[tid],
            resolved_fields[tid],
            geometry_scales[tid],
            elementary_charge_value,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )

    @wp.kernel
    def _combine_charged_coefficient_kernel(
        neutral_coefficients: Any,
        electrostatic_factors: Any,
        drift_terms: Any,
        result: Any,
    ) -> None:
        """Combine one charged coefficient per lane."""
        tid = wp.tid()
        result[tid] = _combine_charged_wall_loss_coefficient_wp(
            neutral_coefficients[tid],
            electrostatic_factors[tid],
            drift_terms[tid],
        )


@pytest.fixture(params=_available_warp_devices())
def device(request) -> str:
    """Provide each available Warp device."""
    return request.param


def _state_lanes() -> tuple[np.ndarray, ...]:
    """Return diffusion- and gravity-dominated fp64 state lanes."""
    return (
        np.array([1.0e-3, 2.5e-3, 8.0e-4], dtype=np.float64),
        np.array([5.0e-9, 1.0e-7, 3.0e-6], dtype=np.float64),
        np.array([950.0, 1200.0, 1800.0], dtype=np.float64),
        np.array([285.0, 298.15, 315.0], dtype=np.float64),
        np.array([95000.0, 101325.0, 85000.0], dtype=np.float64),
    )


def _constants() -> list[Any]:
    """Return exact CPU-default constants as Warp fp64 scalars."""
    return [
        wp.float64(BOLTZMANN_CONSTANT),
        wp.float64(GAS_CONSTANT),
        wp.float64(MOLECULAR_WEIGHT_AIR),
        wp.float64(REF_VISCOSITY_AIR_STP),
        wp.float64(REF_TEMPERATURE_STP),
        wp.float64(SUTHERLAND_CONSTANT),
    ]


def _warp_arrays(values: tuple[np.ndarray, ...], device: str) -> list[Any]:
    """Copy float64 lane inputs to one Warp device."""
    return [
        wp.array(value, dtype=wp.float64, device=device) for value in values
    ]


def _coulomb_self_potential_ratio_oracle(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Calculate independent NumPy Coulomb self-potential ratios."""
    raw_ratio = -(particle_charges**2 * ELEMENTARY_CHARGE_VALUE**2) / (
        4.0
        * np.pi
        * ELECTRIC_PERMITTIVITY
        * (particle_radii + particle_radii)
        * BOLTZMANN_CONSTANT
        * temperatures
    )
    return np.maximum(raw_ratio, -200.0)


def _image_charge_enhancement_oracle(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Calculate independent pairwise-diagonal image-charge factors."""
    pairwise_ratio = -(
        particle_charges[:, None]
        * particle_charges[None, :]
        * ELEMENTARY_CHARGE_VALUE**2
    ) / (
        4.0
        * np.pi
        * ELECTRIC_PERMITTIVITY
        * (particle_radii[:, None] + particle_radii[None, :])
        * BOLTZMANN_CONSTANT
        * temperatures[:, None]
    )
    self_ratio = np.maximum(np.diagonal(pairwise_ratio), -200.0)
    enhancement = np.exp(np.clip(np.abs(self_ratio), -50.0, 50.0))
    return np.where(particle_charges == 0.0, 1.0, enhancement)


def _launch_coulomb_self_potential_ratio(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
    device: str,
) -> np.ndarray:
    """Launch the scalar-ratio helper and return synchronized fp64 results."""
    result = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)
    wp.launch(
        _coulomb_self_potential_ratio_kernel,
        dim=len(particle_radii),
        inputs=[
            *_warp_arrays(
                (particle_radii, particle_charges, temperatures),
                device,
            ),
            wp.float64(ELEMENTARY_CHARGE_VALUE),
            wp.float64(ELECTRIC_PERMITTIVITY),
            wp.float64(BOLTZMANN_CONSTANT),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


def _launch_image_charge_enhancement(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
    device: str,
) -> np.ndarray:
    """Launch the enhancement helper and return synchronized fp64 results."""
    result = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)
    wp.launch(
        _image_charge_enhancement_kernel,
        dim=len(particle_radii),
        inputs=[
            *_warp_arrays(
                (particle_radii, particle_charges, temperatures),
                device,
            ),
            wp.float64(ELEMENTARY_CHARGE_VALUE),
            wp.float64(ELECTRIC_PERMITTIVITY),
            wp.float64(BOLTZMANN_CONSTANT),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


def _field_resolution_oracle(
    geometry_modes: np.ndarray,
    chamber_radii: np.ndarray,
    chamber_lengths: np.ndarray,
    chamber_widths: np.ndarray,
    chamber_heights: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_z: np.ndarray,
    wall_potentials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate independent geometry scales and resolved field forms."""
    scales = np.where(
        geometry_modes == 0,
        chamber_radii,
        np.minimum(
            chamber_lengths, np.minimum(chamber_widths, chamber_heights)
        ),
    )
    potential = np.zeros_like(scales)
    valid_potential = (wall_potentials != 0.0) & (scales > 0.0)
    potential[valid_potential] = (
        wall_potentials[valid_potential] / scales[valid_potential]
    )
    spherical = field_x + potential
    rectangular = np.hypot(np.hypot(field_x, field_y), field_z) + potential
    return scales, spherical, rectangular


def _launch_resolved_electric_fields(
    geometry_modes: np.ndarray,
    chamber_radii: np.ndarray,
    chamber_lengths: np.ndarray,
    chamber_widths: np.ndarray,
    chamber_heights: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_z: np.ndarray,
    wall_potentials: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch field-resolution helpers and return synchronized fp64 outputs."""
    count = len(geometry_modes)
    spherical = wp.zeros(count, dtype=wp.float64, device=device)
    rectangular = wp.zeros(count, dtype=wp.float64, device=device)
    scales = wp.zeros(count, dtype=wp.float64, device=device)
    wp.launch(
        _resolved_electric_field_kernel,
        dim=count,
        inputs=[
            wp.array(geometry_modes, dtype=wp.int32, device=device),
            *_warp_arrays(
                (
                    chamber_radii,
                    chamber_lengths,
                    chamber_widths,
                    chamber_heights,
                    field_x,
                    field_y,
                    field_z,
                    wall_potentials,
                ),
                device,
            ),
        ],
        outputs=[spherical, rectangular, scales],
        device=device,
    )
    wp.synchronize()
    return scales.numpy(), spherical.numpy(), rectangular.numpy()


def _electric_field_drift_oracle(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
    resolved_fields: np.ndarray,
    geometry_scales: np.ndarray,
) -> np.ndarray:
    """Calculate independent Sutherland-viscosity signed drift terms."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        viscosity = (
            REF_VISCOSITY_AIR_STP
            * (temperatures / REF_TEMPERATURE_STP) ** 1.5
            * (REF_TEMPERATURE_STP + SUTHERLAND_CONSTANT)
            / (temperatures + SUTHERLAND_CONSTANT)
        )
        diameter = 2.0 * np.maximum(particle_radii, 1.0e-30)
        mobility = (
            np.abs(particle_charges)
            * ELEMENTARY_CHARGE_VALUE
            / (3.0 * np.pi * viscosity * diameter)
        )
        drift = (
            mobility
            * resolved_fields
            * np.sign(particle_charges)
            / np.maximum(geometry_scales, 1.0e-30)
        )
    drift[(particle_charges == 0.0) | (resolved_fields == 0.0)] = 0.0
    return np.nan_to_num(drift, nan=0.0, posinf=np.inf, neginf=-np.inf)


def _launch_electric_field_drift(
    particle_radii: np.ndarray,
    particle_charges: np.ndarray,
    temperatures: np.ndarray,
    resolved_fields: np.ndarray,
    geometry_scales: np.ndarray,
    device: str,
) -> np.ndarray:
    """Launch the drift helper and return synchronized fp64 results."""
    result = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)
    wp.launch(
        _electric_field_drift_kernel,
        dim=len(particle_radii),
        inputs=[
            *_warp_arrays(
                (
                    particle_radii,
                    particle_charges,
                    temperatures,
                    resolved_fields,
                    geometry_scales,
                ),
                device,
            ),
            wp.float64(ELEMENTARY_CHARGE_VALUE),
            wp.float64(REF_VISCOSITY_AIR_STP),
            wp.float64(REF_TEMPERATURE_STP),
            wp.float64(SUTHERLAND_CONSTANT),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


def _combine_charged_coefficient_oracle(
    neutral_coefficients: np.ndarray,
    electrostatic_factors: np.ndarray,
    drift_terms: np.ndarray,
) -> np.ndarray:
    """Calculate independently sanitized charged coefficient sums."""
    with np.errstate(over="ignore", invalid="ignore"):
        combined = neutral_coefficients * electrostatic_factors + drift_terms
    return np.clip(
        np.nan_to_num(combined, nan=0.0),
        0.0,
        np.finfo(np.float64).max,
    )


def _launch_combined_charged_coefficients(
    neutral_coefficients: np.ndarray,
    electrostatic_factors: np.ndarray,
    drift_terms: np.ndarray,
    device: str,
) -> np.ndarray:
    """Launch coefficient composition and return synchronized fp64 outputs."""
    result = wp.zeros(
        len(neutral_coefficients), dtype=wp.float64, device=device
    )
    wp.launch(
        _combine_charged_coefficient_kernel,
        dim=len(neutral_coefficients),
        inputs=[
            *_warp_arrays(
                (neutral_coefficients, electrostatic_factors, drift_terms),
                device,
            ),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


@pytest.mark.gpu_parity
def test_spherical_wall_loss_matches_cpu_vector_states(device: str) -> None:
    """Ensure spherical helper matches CPU state equations per lane."""
    state = _state_lanes()
    chamber_radii = np.array([0.35, 0.75, 1.20], dtype=np.float64)
    expected = np.array(
        [
            get_spherical_wall_loss_coefficient_via_system_state(
                eddy, radius, density, temperature, pressure, chamber_radius
            )
            for eddy, radius, density, temperature, pressure, chamber_radius in zip(
                *state, chamber_radii, strict=True
            )
        ],
        dtype=np.float64,
    )
    result = wp.zeros(len(expected), dtype=wp.float64, device=device)
    wp.launch(
        _spherical_wall_loss_kernel,
        dim=len(expected),
        inputs=[*_warp_arrays((*state, chamber_radii), device), *_constants()],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    actual = result.numpy()
    assert actual.shape == (len(expected),)
    assert np.all(np.isfinite(actual))
    # CPU get_debye_function omits the zero-endpoint trapezoid interval. Its
    # measured geometry-factor discrepancy for these lanes is 0.001001995.
    npt.assert_allclose(actual, expected, rtol=1.002e-3, atol=1e-20)


@pytest.mark.gpu_parity
def test_rectangle_wall_loss_matches_cpu_vector_states(device: str) -> None:
    """Ensure rectangular helper matches CPU state equations per lane."""
    state = _state_lanes()
    lengths = np.array([1.1, 2.0, 0.8], dtype=np.float64)
    widths = np.array([0.7, 0.9, 1.4], dtype=np.float64)
    heights = np.array([0.5, 1.3, 0.6], dtype=np.float64)
    expected = np.array(
        [
            get_rectangle_wall_loss_coefficient_via_system_state(
                eddy,
                radius,
                density,
                temperature,
                pressure,
                (length, width, height),
            )
            for eddy, radius, density, temperature, pressure, length, width, height in zip(
                *state, lengths, widths, heights, strict=True
            )
        ],
        dtype=np.float64,
    )
    result = wp.zeros(len(expected), dtype=wp.float64, device=device)
    wp.launch(
        _rectangle_wall_loss_kernel,
        dim=len(expected),
        inputs=[
            *_warp_arrays((*state, lengths, widths, heights), device),
            *_constants(),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    actual = result.numpy()
    assert actual.shape == (len(expected),)
    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-20)


def test_rectangle_wall_loss_handles_extreme_finite_dimensions(
    device: str,
) -> None:
    """Avoid product overflow for extreme but finite rectangular dimensions."""
    state = tuple(value[:1] for value in _state_lanes())
    dimensions = tuple(np.array([1.0e200], dtype=np.float64) for _ in range(3))
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(
        _rectangle_wall_loss_kernel,
        dim=1,
        inputs=[
            *_warp_arrays((*state, *dimensions), device),
            *_constants(),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()

    actual = result.numpy()
    assert np.all(np.isfinite(actual))
    assert np.all(actual > 0.0)


@pytest.mark.gpu_parity
def test_rectangle_wall_loss_uses_settling_limit_at_zero_transport(
    device: str,
) -> None:
    """Return finite settling limits for zero and underflow transport scales."""
    particle_radii = np.array([3.0e-6, 3.0e-6], dtype=np.float64)
    particle_densities = np.array([1800.0, 1800.0], dtype=np.float64)
    temperatures = np.array([298.15, 298.15], dtype=np.float64)
    pressures = np.array([101325.0, 101325.0], dtype=np.float64)
    lengths = np.array([1.0, 1.0], dtype=np.float64)
    widths = np.array([0.8, 0.8], dtype=np.float64)
    heights = np.array([0.6, 0.6], dtype=np.float64)
    eddy_diffusivities = np.array(
        [0.0, np.nextafter(np.float64(0.0), np.float64(1.0))],
        dtype=np.float64,
    )
    settling_velocity = get_particle_settling_velocity_via_system_state(
        particle_radius=particle_radii,
        particle_density=particle_densities,
        temperature=298.15,
        pressure=101325.0,
    )
    expected = settling_velocity / heights
    result = wp.zeros(2, dtype=wp.float64, device=device)
    wp.launch(
        _rectangle_wall_loss_kernel,
        dim=2,
        inputs=[
            *_warp_arrays(
                (
                    eddy_diffusivities,
                    particle_radii,
                    particle_densities,
                    temperatures,
                    pressures,
                    lengths,
                    widths,
                    heights,
                ),
                device,
            ),
            *_constants(),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    actual = result.numpy()
    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


@pytest.mark.gpu_parity
@pytest.mark.parametrize("lane", [0, 2], ids=["diffusion", "gravity"])
def test_wall_loss_helpers_match_cpu_scalar_regimes(
    device: str,
    lane: int,
) -> None:
    """Ensure scalar diffusion and gravity regimes match CPU state equations."""
    state = tuple(value[lane : lane + 1] for value in _state_lanes())
    sphere_radius = np.array([0.35 if lane == 0 else 1.2], dtype=np.float64)
    dimensions = (
        np.array([1.1 if lane == 0 else 0.8], dtype=np.float64),
        np.array([0.7 if lane == 0 else 1.4], dtype=np.float64),
        np.array([0.5 if lane == 0 else 0.6], dtype=np.float64),
    )
    sphere_expected = get_spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=state[0][0],
        particle_radius=state[1][0],
        particle_density=state[2][0],
        temperature=state[3][0],
        pressure=state[4][0],
        chamber_radius=sphere_radius[0],
    )
    rectangle_expected = get_rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=state[0][0],
        particle_radius=state[1][0],
        particle_density=state[2][0],
        temperature=state[3][0],
        pressure=state[4][0],
        chamber_dimensions=tuple(value[0] for value in dimensions),
    )
    sphere_result = wp.zeros(1, dtype=wp.float64, device=device)
    rectangle_result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(
        _spherical_wall_loss_kernel,
        dim=1,
        inputs=[*_warp_arrays((*state, sphere_radius), device), *_constants()],
        outputs=[sphere_result],
        device=device,
    )
    wp.launch(
        _rectangle_wall_loss_kernel,
        dim=1,
        inputs=[*_warp_arrays((*state, *dimensions), device), *_constants()],
        outputs=[rectangle_result],
        device=device,
    )
    wp.synchronize()
    sphere_actual = sphere_result.numpy()
    rectangle_actual = rectangle_result.numpy()
    assert sphere_actual.shape == (1,)
    assert rectangle_actual.shape == (1,)
    assert np.all(np.isfinite(sphere_actual))
    assert np.all(np.isfinite(rectangle_actual))
    # CPU get_debye_function omits the zero-endpoint trapezoid interval. Its
    # measured geometry-factor discrepancy for these lanes is 0.001001995.
    npt.assert_allclose(sphere_actual[0], sphere_expected, rtol=1.002e-3)
    npt.assert_allclose(rectangle_actual[0], rectangle_expected, rtol=1e-10)


def test_wall_loss_helper_smoke_launches(device: str) -> None:
    """Ensure both supported helpers compile, launch, and return finite rates."""
    state = tuple(value[:1] for value in _state_lanes())
    sphere_result = wp.zeros(1, dtype=wp.float64, device=device)
    rectangle_result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(
        _spherical_wall_loss_kernel,
        dim=1,
        inputs=[
            *_warp_arrays((*state, np.array([0.5], dtype=np.float64)), device),
            *_constants(),
        ],
        outputs=[sphere_result],
        device=device,
    )
    wp.launch(
        _rectangle_wall_loss_kernel,
        dim=1,
        inputs=[
            *_warp_arrays(
                (
                    *state,
                    np.array([1.0], dtype=np.float64),
                    np.array([0.8], dtype=np.float64),
                    np.array([0.6], dtype=np.float64),
                ),
                device,
            ),
            *_constants(),
        ],
        outputs=[rectangle_result],
        device=device,
    )
    wp.synchronize()
    assert np.isfinite(sphere_result.numpy()[0])
    assert np.isfinite(rectangle_result.numpy()[0])


@pytest.mark.gpu_parity
def test_coulomb_self_potential_ratio_matches_independent_oracle(
    device: str,
) -> None:
    """Match ordinary and clipped Coulomb self-potential ratios."""
    particle_radii = np.array([1.0e-7, 1.0e-9, 1.0e-10], dtype=np.float64)
    particle_charges = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    temperatures = np.array([300.0, 300.0, 300.0], dtype=np.float64)

    expected = _coulomb_self_potential_ratio_oracle(
        particle_radii,
        particle_charges,
        temperatures,
    )
    actual = _launch_coulomb_self_potential_ratio(
        particle_radii,
        particle_charges,
        temperatures,
        device,
    )

    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert np.all(np.isfinite(actual))
    assert np.all(actual <= 0.0)
    assert actual[2] == -200.0
    assert -200.0 < actual[1] < -50.0


@pytest.mark.gpu_parity
def test_image_charge_enhancement_matches_independent_oracle(
    device: str,
) -> None:
    """Match signed, zero, and mixed-scale image-charge lane factors."""
    particle_radii = np.array(
        [1.0e-9, 1.0e-8, 1.0e-6, 1.0e-8, 1.0e-9],
        dtype=np.float64,
    )
    particle_charges = np.array(
        [-3.0, -1.0, 0.0, 1.0, 3.0],
        dtype=np.float64,
    )
    temperatures = np.array(
        [300.0, 280.0, 310.0, 280.0, 300.0],
        dtype=np.float64,
    )

    expected = _image_charge_enhancement_oracle(
        particle_radii,
        particle_charges,
        temperatures,
    )
    actual = _launch_image_charge_enhancement(
        particle_radii,
        particle_charges,
        temperatures,
        device,
    )

    assert actual.shape == (len(particle_charges),)
    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert actual[2] == 1.0
    assert actual[0] == actual[4]
    assert actual[1] == actual[3]


@pytest.mark.gpu_parity
def test_image_charge_enhancement_is_greater_than_unity_without_field_inputs(
    device: str,
) -> None:
    """Apply image-charge enhancement without wall-potential or field inputs."""
    particle_radii = np.array([1.0e-7], dtype=np.float64)
    particle_charges = np.array([1.0], dtype=np.float64)
    temperatures = np.array([298.15], dtype=np.float64)

    actual = _launch_image_charge_enhancement(
        particle_radii,
        particle_charges,
        temperatures,
        device,
    )

    assert actual[0] > 1.0


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("particle_radius", "particle_charge"),
    [(1.0e-9, 2.0), (1.0e-9, 3.0)],
    ids=["exponent_clip", "raw_ratio_and_exponent_clip"],
)
def test_image_charge_enhancement_clipping_matches_oracle(
    device: str,
    particle_radius: float,
    particle_charge: float,
) -> None:
    """Match valid image-charge clipping domains without warnings."""
    particle_radii = np.array([particle_radius], dtype=np.float64)
    particle_charges = np.array([particle_charge], dtype=np.float64)
    temperatures = np.array([300.0], dtype=np.float64)

    expected = _image_charge_enhancement_oracle(
        particle_radii,
        particle_charges,
        temperatures,
    )
    actual = _launch_image_charge_enhancement(
        particle_radii,
        particle_charges,
        temperatures,
        device,
    )

    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


@pytest.mark.gpu_parity
def test_charged_field_helpers_match_geometry_specific_cpu_semantics(
    device: str,
) -> None:
    """Match signed spherical and norm-based rectangular field resolution."""
    geometry_modes = np.array([0, 1, 1, 0, 0, 0], dtype=np.int32)
    chamber_radii = np.array(
        [0.5, 9.0, 9.0, 1.0e-32, 0.0, -0.5], dtype=np.float64
    )
    lengths = np.array([3.0, 2.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    widths = np.array([3.0, 0.5, 0.6, 2.0, 2.0, 2.0], dtype=np.float64)
    heights = np.array([3.0, 1.0, 0.8, 2.0, 2.0, 2.0], dtype=np.float64)
    field_x = np.array([-3.0, 3.0, 0.0, 0.0, -3.0, -3.0])
    field_y = np.array([0.0, 4.0, 0.0, 0.0, 4.0, 4.0])
    field_z = np.array([0.0, 12.0, 0.0, 0.0, 12.0, 12.0])
    potentials = np.array([1.0, -2.0, 1.5, 1.0e-32, 2.0, 2.0])

    expected = _field_resolution_oracle(
        geometry_modes,
        chamber_radii,
        lengths,
        widths,
        heights,
        field_x,
        field_y,
        field_z,
        potentials,
    )
    actual = _launch_resolved_electric_fields(
        geometry_modes,
        chamber_radii,
        lengths,
        widths,
        heights,
        field_x,
        field_y,
        field_z,
        potentials,
        device,
    )

    for observed, reference in zip(actual, expected, strict=True):
        assert observed.shape == reference.shape
        npt.assert_allclose(observed, reference, rtol=1e-12, atol=0.0)
    assert actual[2][1] < np.linalg.norm([3.0, 4.0, 12.0])
    assert actual[0][3] == chamber_radii[3]
    npt.assert_allclose(actual[1][4:], field_x[4:], rtol=0.0, atol=0.0)
    npt.assert_allclose(
        actual[2][4:],
        np.linalg.norm(
            np.array([field_x[4:], field_y[4:], field_z[4:]]), axis=0
        ),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu_parity
def test_rectangular_field_resolution_is_stable_for_large_components(
    device: str,
) -> None:
    """Resolve finite extreme field components without intermediate overflow."""
    actual = _launch_resolved_electric_fields(
        np.array([1, 1], dtype=np.int32),
        np.ones(2),
        np.ones(2),
        np.ones(2),
        np.ones(2),
        np.array([1.0e200, 1.0e200]),
        np.array([-1.0e200, 0.0]),
        np.array([1.0e200, 0.0]),
        np.array([-1.0e200, 0.0]),
        device,
    )

    expected = np.array([np.sqrt(3.0) * 1.0e200 - 1.0e200, 1.0e200])
    assert np.all(np.isfinite(actual[2]))
    npt.assert_allclose(actual[2], expected, rtol=1e-12, atol=0.0)


@pytest.mark.gpu_parity
def test_electric_field_drift_matches_independent_signed_fp64_oracle(
    device: str,
) -> None:
    """Match charge sign, magnitude, zero controls, and guard boundaries."""
    particle_radii = np.array(
        [
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-40,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
        ],
        dtype=np.float64,
    )
    particle_charges = np.array([1.0, -1.0, 2.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    temperatures = np.full(len(particle_radii), 298.15, dtype=np.float64)
    resolved_fields = np.array(
        [
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            0.0,
            -10.0,
            -10.0,
            np.nan,
        ]
    )
    geometry_scales = np.array(
        [0.5, 0.5, 0.5, 0.5, 1.0e-40, 0.5, 0.5, 0.5, 0.5]
    )

    expected = _electric_field_drift_oracle(
        particle_radii,
        particle_charges,
        temperatures,
        resolved_fields,
        geometry_scales,
    )
    actual = _launch_electric_field_drift(
        particle_radii,
        particle_charges,
        temperatures,
        resolved_fields,
        geometry_scales,
        device,
    )

    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert actual[0] == -actual[1]
    assert actual[2] == 2.0 * actual[0]
    assert actual[3] == 0.0
    assert actual[5] == 0.0
    assert actual[6] == -actual[7]
    assert actual[6] < 0.0
    assert actual[8] == 0.0


@pytest.mark.gpu_parity
def test_charged_coefficient_composition_sanitizes_defensive_lanes(
    device: str,
) -> None:
    """Map cancellation and nonfinite combined rates to bounded fp64 values."""
    neutral = np.array(
        [2.0, 2.0, np.nan, 1.0, np.finfo(np.float64).max],
        dtype=np.float64,
    )
    factor = np.array(
        [3.0, 1.0, 1.0, 1.0, 2.0],
        dtype=np.float64,
    )
    drift = np.array([1.0, -3.0, 0.0, -np.inf, 0.0], dtype=np.float64)

    expected = _combine_charged_coefficient_oracle(neutral, factor, drift)
    actual = _launch_combined_charged_coefficients(
        neutral,
        factor,
        drift,
        device,
    )

    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    npt.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual[1] == 0.0
    assert actual[2] == 0.0
    assert actual[3] == 0.0
    assert actual[4] == np.finfo(np.float64).max
