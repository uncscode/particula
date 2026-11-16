"""Native adaptive GPU accuracy, physical budgets and publication boundaries."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp
from scipy.integrate import quad

from particula.execution.condensation_ode import CondensationODE
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.tests.condensation_ode_test import case


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require Warp CPU and qualify CUDA only on a real available device."""
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def reference_time(radius_ratio=1.2):
    """Independent inverse-radius quadrature with Sutherland gas viscosity."""
    temperature, pressure, molar = 298.15, 101325.0, 0.018
    viscosity = (
        1.716e-5
        * (temperature / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature + 110.4)
    )
    mean_free_path = (
        2
        * viscosity
        / pressure
        / np.sqrt(8 * molar / (np.pi * 8.31446261815324 * temperature))
    )
    initial_mass = 4 * np.pi * 1000 * 1e-18 / 3

    def inverse(x):
        kn = mean_free_path / (1e-6 * x)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        gas = 1e-5 - 1e9 * initial_mass * (x**3 - 1)
        return 1000 * 1e-12 * x / (2e-5 * correction * gas)

    duration, uncertainty = quad(
        inverse, 1, radius_ratio, epsabs=1e-12, epsrel=1e-12
    )
    assert uncertainty < 1e-10
    return duration


def advance(owner, bound):
    """Advance only complete accepted device or CPU steps."""
    while owner.time < bound:
        owner.step(bound)
    return owner.snapshot()


def test_independent_growth_accuracy_and_species_inventory(device):
    """Mapped growth reaches the quadrature radius without hiding gas error."""
    initial, strategy = case()
    owner = GPUCondensationODE(
        initial,
        strategy,
        device=device,
        rtol=1e-6,
        particle_atol=1e-24,
        gas_atol=1e-14,
    )
    final = advance(owner, reference_time())
    npt.assert_allclose(final.particles.radii[0, 0] / 1e-6, 1.2, rtol=2e-6)
    total = (
        initial.particles.species_mass_density[:, 0]
        + initial.gas.concentration[:, 1]
    )
    npt.assert_allclose(
        final.particles.species_mass_density[:, 0]
        + final.gas.concentration[:, 1],
        total,
        rtol=3e-13,
    )
    npt.assert_array_equal(
        final.gas.concentration[:, 0], initial.gas.concentration[:, 0]
    )
    npt.assert_array_equal(final.particles.masses[:, 1], 0)
    npt.assert_array_equal(
        final.particles.concentration, initial.particles.concentration
    )
    npt.assert_allclose(initial.particles.radii[0, 0], 1e-6, rtol=1e-15)
    final.particles.masses[:] = 999
    assert owner.snapshot().particles.masses.max() < 1e-12
    assert owner.device_storage_bytes > owner._state.size * 8


def test_tolerance_refinement_meets_declared_observable_targets(device):
    """Looser/tighter local controls improve the independent global radius."""
    initial, strategy = case()
    errors = []
    for tolerance in (1e-3, 1e-5, 1e-7):
        owner = GPUCondensationODE(
            initial,
            strategy,
            device=device,
            rtol=tolerance,
            particle_atol=1e-27,
            gas_atol=1e-17,
            first_step=0.5,
            max_step=10,
        )
        final = advance(owner, reference_time())
        errors.append(abs(final.particles.radii[0, 0] / 1e-6 - 1.2))
    assert errors[1] < errors[0] / 5
    assert errors[2] < errors[1] / 5
    npt.assert_array_less(errors, [1e-3, 1e-5, 1e-7])


def test_shared_heterogeneous_boxes_match_independent_cpu_rates(device):
    """Different reservoirs and non-unit volumes retain per-box inventories."""
    initial, strategy = case(3)
    initial.gas.concentration[:, 1] *= [0.1, 1, 10]
    gpu = GPUCondensationODE(initial, strategy, device=device, rtol=1e-7)
    cpu = CondensationODE(initial, strategy, rtol=1e-8)
    result = advance(gpu, 0.2)
    reference = advance(cpu, 0.2)
    npt.assert_allclose(
        result.particles.masses, reference.particles.masses, rtol=2e-7
    )
    npt.assert_allclose(
        result.gas.concentration, reference.gas.concentration, rtol=2e-7
    )
    npt.assert_allclose(
        result.particles.species_mass_density[:, 0]
        + result.gas.concentration[:, 1],
        initial.particles.species_mass_density[:, 0]
        + initial.gas.concentration[:, 1],
        rtol=3e-13,
    )


def test_domain_rejection_and_trial_exhaustion_preserve_accepted_state(device):
    """A failed candidate cannot publish even when device trial buffers changed."""
    initial, strategy = case()
    owner = GPUCondensationODE(
        initial,
        strategy,
        device=device,
        first_step=100,
        max_step=100,
        max_trials=1,
    )
    before = owner.snapshot()
    with pytest.raises(RuntimeError, match="trial count"):
        owner.step(100)
    assert owner.time == 0
    assert owner.statistics.rejected == owner.statistics.domain_rejected == 1
    npt.assert_array_equal(
        owner.snapshot().particles.masses, before.particles.masses
    )
    npt.assert_array_equal(
        owner.snapshot().gas.concentration, before.gas.concentration
    )
    # Reuse the same buffers after rejection and subsequently accept valid work.
    owner.max_trials = 100
    advance(owner, 0.2)
    assert owner.statistics.accepted > 0


def test_normal_steps_download_only_control_scalars_and_never_cpu_rates(
    device, monkeypatch
):
    """The host handles control; physical arrays and derivatives stay on device."""
    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device=device)
    original_numpy = wp.array.numpy

    def scalar_only(array, *args, **kwargs):
        assert array.size == 1
        return original_numpy(array, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU physical rates must not run")

    monkeypatch.setattr(wp.array, "numpy", scalar_only)
    monkeypatch.setattr(type(strategy), "mass_transfer_rate", forbidden)
    owner.step(0.1)
    assert owner.control_readbacks == 9


def test_absolute_stage_forcing_agrees_with_cpu(device):
    """Nonautonomous temperatures are evaluated at absolute RK stage times."""
    initial, strategy = case()
    seen = []

    def forcing(time):
        seen.append(time)
        return 298.15 + 2 * time, 101325.0

    gpu = GPUCondensationODE(
        initial, strategy, device=device, time=2, forcing=forcing
    )
    cpu = CondensationODE(initial, strategy, time=2, forcing=forcing)
    actual, expected = advance(gpu, 2.1), advance(cpu, 2.1)
    assert min(seen) >= 2
    assert max(seen) <= 2.1
    npt.assert_allclose(actual.environment.temperature, [302.35], rtol=1e-15)
    npt.assert_allclose(
        actual.particles.masses, expected.particles.masses, rtol=1e-10
    )


def test_unsupported_methods_reject_before_upload(monkeypatch):
    """Stiff GPU work is not silently run on a CPU or through a fixed-step map."""
    initial, strategy = case()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "unsupported method must reject before device access"
        )

    monkeypatch.setattr(wp, "get_device", forbidden)
    with pytest.raises(ValueError, match="rk23 only"):
        GPUCondensationODE(initial, strategy, device="cuda:0", method="ros2")


def test_cuda_capture_rejects_before_launch():
    """Native capture cannot include host-controlled adaptive trial decisions."""
    if not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device="cuda:0")
    with wp.ScopedCapture(device=owner.device):
        with pytest.raises(RuntimeError, match="capture"):
            owner.step(0.1)
    assert owner.time == owner.statistics.accepted == 0


def test_discrete_population_uses_density_without_resolved_volume_factor(
    device,
):
    """The equivalent discrete and resolved populations have the same growth."""
    from particula.aerosol import Aerosol
    from particula.particles import ParticleData

    resolved, strategy = case()
    discrete = Aerosol(
        particles=ParticleData(
            masses=resolved.particles.masses.copy(),
            concentration=(
                resolved.particles.concentration
                / resolved.particles.volume[:, None]
            ),
            charge=resolved.particles.charge.copy(),
            density=resolved.particles.density.copy(),
            volume=np.ones(1),
            distribution_type="discrete",
        ),
        gas=resolved.gas.copy(),
        environment=resolved.environment.copy(),
    )
    left = advance(GPUCondensationODE(resolved, strategy, device=device), 0.2)
    right = advance(GPUCondensationODE(discrete, strategy, device=device), 0.2)
    npt.assert_allclose(
        left.particles.masses, right.particles.masses, rtol=1e-13
    )
    npt.assert_allclose(
        left.gas.concentration, right.gas.concentration, rtol=1e-13
    )


def test_mixed_material_kappa_and_buck_profile_matches_cpu(device):
    """The existing mapped water/activity profile retains dry and inert lanes."""
    from particula.dynamics import CondensationIsothermal
    from particula.execution.tests.prepared_restart_capture_test import scenario

    initial, process, _ = scenario("condensation")
    source = process.condensation_strategy
    strategy = CondensationIsothermal(
        **{
            name: getattr(source, name)
            for name in (
                "molar_mass",
                "diffusion_coefficient",
                "accommodation_coefficient",
                "update_gases",
                "skip_partitioning_indices",
                "activity_strategy",
                "surface_strategy",
                "vapor_pressure_strategy",
                "species_map",
            )
        }
    )
    actual = advance(
        GPUCondensationODE(initial, strategy, device=device, rtol=1e-8), 0.01
    )
    reference = advance(CondensationODE(initial, strategy, rtol=1e-9), 0.01)
    npt.assert_allclose(
        actual.particles.masses, reference.particles.masses, rtol=2e-8
    )
    npt.assert_allclose(
        actual.gas.concentration, reference.gas.concentration, rtol=2e-8
    )
    npt.assert_array_equal(
        actual.particles.masses[:, :, 1], initial.particles.masses[:, :, 1]
    )
    npt.assert_array_equal(
        actual.gas.concentration[:, :2], initial.gas.concentration[:, :2]
    )


def test_requested_unavailable_cuda_never_selects_cpu(monkeypatch):
    """An unavailable requested device fails instead of silently falling back."""
    initial, strategy = case()
    requests = []

    def unavailable(device):
        requests.append(device)
        raise RuntimeError("requested CUDA unavailable")

    monkeypatch.setattr(wp, "get_device", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        GPUCondensationODE(initial, strategy, device="cuda:9")
    assert requests == ["cuda:9"]


def test_near_depletion_meets_independent_inventory_target(device):
    """A nearly exhausted reservoir is accurate separately from total mass."""
    initial, strategy = case()
    target_radius = 1.49
    owner = GPUCondensationODE(
        initial,
        strategy,
        device=device,
        rtol=1e-5,
        particle_atol=1e-27,
        gas_atol=1e-17,
        max_step=20,
    )
    final = advance(owner, reference_time(target_radius))
    expected_gas = 1e-5 - 1e9 * (4 * np.pi * 1000 * 1e-18 / 3) * (
        target_radius**3 - 1
    )
    assert expected_gas < 0.04 * 1e-5
    assert abs(final.particles.radii[0, 0] / 1e-6 - target_radius) < 1e-5
    assert abs(final.gas.concentration[0, 1] - expected_gas) / 1e-5 < 1e-5
