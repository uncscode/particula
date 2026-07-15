"""Bounded autodiff-readiness and forward-limit tests for condensation."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support
from particula.util import constants

pytestmark = pytest.mark.warp

_INTERIOR_INVENTORY_MARGIN = 0.5
_FINITE_DIFFERENCE_EPSILON = 1.0e-9
_DERIVATIVE_RTOL = 2.0e-5
_DERIVATIVE_ATOL = 5.0e-11


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime(request: pytest.FixtureRequest) -> None:
    """Load Warp only while executing a selected Warp-backed test."""
    if request.node.get_closest_marker("warp") is not None:
        support._load_warp_runtime()


def _require_autodiff_capabilities(runtime: Any, device: str) -> None:
    """Skip only when this device lacks the bounded probe prerequisites."""
    wp = runtime.wp
    if not callable(getattr(wp, "Tape", None)):
        pytest.skip(
            f"{device}: wp.Tape is unavailable for the bounded out-of-place "
            "raw-rate probe"
        )
    if hasattr(wp.config, "enable_backward") and not wp.config.enable_backward:
        pytest.skip(
            f"{device}: Warp backward autograd is disabled for the bounded "
            "out-of-place raw-rate probe"
        )
    if not hasattr(wp.config, "verify_autograd_array_access"):
        pytest.skip(
            f"{device}: verify_autograd_array_access is unavailable for the "
            "bounded out-of-place raw-rate probe"
        )


@contextmanager
def _verify_autograd_array_access(runtime: Any) -> Iterator[None]:
    """Enable Warp access verification and restore its exact prior setting."""
    config = runtime.wp.config
    previous_value = config.verify_autograd_array_access
    config.verify_autograd_array_access = True
    try:
        yield
    finally:
        config.verify_autograd_array_access = previous_value


def _make_smooth_rate_case(
    runtime: Any,
    device: str,
    gas_concentration: float,
    *,
    requires_grad: bool = False,
) -> dict[str, Any]:
    """Build a one-lane positive, interior raw-rate kernel fixture."""
    wp = runtime.wp

    def array(values: Any, **kwargs: Any) -> Any:
        """Allocate an active-device fp64 fixture array."""
        return wp.array(values, dtype=wp.float64, device=device, **kwargs)

    mass_transfer_kwargs: dict[str, Any] = {
        "dtype": wp.float64,
        "device": device,
    }

    return {
        "masses": array([[[1.0e-15]]]),
        "concentration": array([[1.0e2]]),
        "density": array([1000.0]),
        "gas_concentration": array(
            [[gas_concentration]], requires_grad=requires_grad
        ),
        "vapor_pressure": array([[0.01]]),
        "molar_mass": array([0.018]),
        "surface_tension": array([0.072]),
        "kappas": array([0.0]),
        "molar_mass_reference": array([0.018]),
        "effective_surface_tension": array([[0.072]]),
        "mass_accommodation": array([1.0]),
        "diffusion_coefficient_vapor": array([2.0e-5]),
        "latent_heat": array([0.0]),
        "dynamic_viscosity": array([1.8e-5]),
        "mean_free_path": array([6.6e-8]),
        "temperature": array([298.15]),
        "mass_transfer": wp.zeros(
            (1, 1, 1),
            **mass_transfer_kwargs,
            requires_grad=requires_grad,
        ),
    }


def _launch_raw_rate(runtime: Any, device: str, case: dict[str, Any]) -> float:
    """Launch the raw out-of-place rate kernel and return its scalar output."""
    wp = runtime.wp
    wp.launch(
        runtime.condensation_module.condensation_mass_transfer_kernel,
        dim=(1, 1),
        inputs=[
            case["masses"],
            case["concentration"],
            case["density"],
            case["gas_concentration"],
            case["vapor_pressure"],
            case["molar_mass"],
            case["surface_tension"],
            case["kappas"],
            case["molar_mass_reference"],
            case["effective_surface_tension"],
            wp.int32(0),
            wp.int32(0),
            wp.int32(0),
            wp.int32(0),
            case["mass_accommodation"],
            case["diffusion_coefficient_vapor"],
            case["latent_heat"],
            wp.int32(0),
            case["dynamic_viscosity"],
            case["mean_free_path"],
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            case["temperature"],
            wp.float64(0.1),
        ],
        outputs=[case["mass_transfer"]],
        device=device,
        record_tape=True,
    )
    wp.synchronize_device(device)
    return float(case["mass_transfer"].numpy()[0, 0, 0])


def _assert_interior_margins(
    raw_transfer: float,
    gas_concentration: float,
) -> None:
    """Assert the raw proposal stays away from P2 clamp boundaries."""
    particle_mass = 1.0e-15
    particle_concentration = 1.0e6
    assert np.isfinite(raw_transfer)
    assert raw_transfer > 0.0
    assert particle_mass + raw_transfer > particle_mass * 0.99
    assert (
        particle_concentration * raw_transfer
        < gas_concentration * _INTERIOR_INVENTORY_MARGIN
    )


def _assert_tape_matches_centered_difference(runtime: Any, device: str) -> None:
    """Check the bounded raw-rate adjoint against a centered fp64 reference."""
    _require_autodiff_capabilities(runtime, device)
    gas_concentration = 1.0e-2
    tape_case = _make_smooth_rate_case(
        runtime, device, gas_concentration, requires_grad=True
    )
    with _verify_autograd_array_access(runtime):
        with runtime.wp.Tape() as tape:
            forward = _launch_raw_rate(runtime, device, tape_case)
        tape.backward(loss=tape_case["mass_transfer"])
    derivative = float(
        tape.gradients[tape_case["gas_concentration"]].numpy()[0, 0]
    )
    base_case = _make_smooth_rate_case(runtime, device, gas_concentration)
    _assert_interior_margins(
        _launch_raw_rate(runtime, device, base_case), gas_concentration
    )
    plus_case = _make_smooth_rate_case(
        runtime, device, gas_concentration + _FINITE_DIFFERENCE_EPSILON
    )
    minus_case = _make_smooth_rate_case(
        runtime, device, gas_concentration - _FINITE_DIFFERENCE_EPSILON
    )
    forward_plus = _launch_raw_rate(runtime, device, plus_case)
    forward_minus = _launch_raw_rate(runtime, device, minus_case)
    _assert_interior_margins(
        forward_plus, gas_concentration + _FINITE_DIFFERENCE_EPSILON
    )
    _assert_interior_margins(
        forward_minus, gas_concentration - _FINITE_DIFFERENCE_EPSILON
    )
    centered_derivative = (forward_plus - forward_minus) / (
        2.0 * _FINITE_DIFFERENCE_EPSILON
    )
    assert np.isfinite(forward)
    assert np.isfinite(derivative)
    npt.assert_allclose(
        derivative,
        centered_derivative,
        rtol=_DERIVATIVE_RTOL,
        atol=_DERIVATIVE_ATOL,
    )


def _make_p2_state(
    runtime: Any,
    device: str,
    mass: float,
    concentration: float,
    gas_concentration: float,
) -> tuple[Any, Any]:
    """Build minimal valid particle and gas state for direct P2 semantics."""
    particles = dataclasses.replace(
        support._make_particle_data(1, 1, 1),
        masses=np.array([[[mass]]], dtype=np.float64),
        concentration=np.array([[concentration]], dtype=np.float64),
    )
    gas = dataclasses.replace(
        support._make_gas_data(1, 1),
        concentration=np.array([[gas_concentration]], dtype=np.float64),
    )
    return (
        runtime.to_warp_particle_data(particles, device=device),
        runtime.to_warp_gas_data(
            gas,
            device=device,
            vapor_pressure=np.zeros((1, 1), dtype=np.float64),
        ),
    )


def test_autograd_array_access_guard_restores_previous_value_after_exception() -> (
    None
):
    """Access verification restoration survives a sentinel failure."""
    runtime = support._load_warp_runtime()
    _require_autodiff_capabilities(runtime, "cpu")
    config = runtime.wp.config
    original_value = config.verify_autograd_array_access
    configured_value = not original_value
    config.verify_autograd_array_access = configured_value
    try:
        with pytest.raises(RuntimeError, match="sentinel"):
            with _verify_autograd_array_access(runtime):
                raise RuntimeError("sentinel")
        assert config.verify_autograd_array_access is configured_value
    finally:
        config.verify_autograd_array_access = original_value


def test_condensation_raw_rate_tape_matches_centered_difference_on_cpu() -> (
    None
):
    """Tape differentiates the bounded raw rate with respect to gas inventory."""
    _assert_tape_matches_centered_difference(
        support._load_warp_runtime(), "cpu"
    )


@pytest.mark.cuda
def test_condensation_raw_rate_tape_matches_centered_difference_on_cuda() -> (
    None
):
    """CUDA adds optional bounded raw-rate Tape evidence when available."""
    runtime = support._load_warp_runtime()
    if not runtime.cuda_available(runtime.wp):
        pytest.skip(runtime.CUDA_SKIP_REASON)
    _assert_tape_matches_centered_difference(runtime, "cuda")


def test_p2_evaporation_boundary_clamps_transfer_to_owned_mass() -> None:
    """P2 forward semantics clamp evaporation at owned particle mass."""
    runtime = support._load_warp_runtime()
    particles, gas = _make_p2_state(runtime, "cpu", 2.0, 1.0, 1.0)
    proposal = runtime.wp.array(
        [[[-3.0]]], dtype=runtime.wp.float64, device="cpu"
    )
    finalized = runtime._finalize_inventory_limited_mass_transfer(
        particles, gas, proposal
    )
    npt.assert_allclose(finalized.numpy(), [[[-2.0]]], rtol=0.0, atol=0.0)
    npt.assert_allclose(particles.masses.numpy(), [[[0.0]]], rtol=0.0, atol=0.0)
    assert np.all(particles.masses.numpy() >= 0.0)


def test_p2_uptake_inventory_boundary_limits_concentration_weighted_transfer() -> (
    None
):
    """P2 forward semantics scale uptake to available gas inventory."""
    runtime = support._load_warp_runtime()
    particles, gas = _make_p2_state(runtime, "cpu", 1.0, 2.0, 1.0)
    wp = runtime.wp
    scratch = runtime.CondensationScratchBuffers(
        positive_mass_transfer_demand=wp.zeros(
            (1, 1), dtype=wp.float64, device="cpu"
        ),
        negative_mass_transfer_release=wp.zeros(
            (1, 1), dtype=wp.float64, device="cpu"
        ),
        positive_mass_transfer_scale=wp.zeros(
            (1, 1), dtype=wp.float64, device="cpu"
        ),
    )
    finalized = runtime._finalize_inventory_limited_mass_transfer(
        particles,
        gas,
        wp.array([[[2.0]]], dtype=wp.float64, device="cpu"),
        scratch,
    )
    assert scratch.positive_mass_transfer_scale is not None
    scale = scratch.positive_mass_transfer_scale.numpy()[0, 0]
    assert 0.0 < scale < 1.0
    npt.assert_allclose(2.0 * finalized.numpy()[0, 0, 0], 1.0, rtol=1e-12)


def test_p2_in_place_mass_mutation_is_forward_semantics_non_claim() -> None:
    """Forward-only non-claim: P2 applies finalized transfer in place."""
    runtime = support._load_warp_runtime()
    particles, gas = _make_p2_state(runtime, "cpu", 1.0, 1.0, 10.0)
    finalized = runtime._finalize_inventory_limited_mass_transfer(
        particles,
        gas,
        runtime.wp.array([[[0.25]]], dtype=runtime.wp.float64, device="cpu"),
    )
    npt.assert_allclose(finalized.numpy(), [[[0.25]]], rtol=0.0, atol=0.0)
    npt.assert_allclose(
        particles.masses.numpy(), [[[1.25]]], rtol=0.0, atol=0.0
    )
