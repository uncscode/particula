"""Independent pair limits and native material/representation contracts."""

import copy

import numpy as np
import pytest
from particula import dynamics as dyn
from particula.dynamics.coagulation._pair_potential import (
    dielectric_response,
    hamaker_energy,
)
from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    pair_enhancements,
)
from particula.dynamics.tests.native_runnable_test import state
from particula.particles import ParticleData


def test_distant_sphere_dipole_and_hamaker_tail():
    """At large separation dispersion decays as s^-6 and induction as s^-4."""
    x, f, eps = 0.001, 0.25, 5.9
    expected = -((4 * f * (1 - f)) ** 3) * x**6 / 36
    assert hamaker_energy(x, f) == pytest.approx(expected, rel=5e-6, abs=0)
    polarizability = (eps - 1) / (eps + 2)
    response = dielectric_response(x, f, eps, 16)
    assert response[0] == pytest.approx(
        -0.5 * polarizability * (1 - f) ** 3 * x**4, rel=2e-6, abs=0
    )
    np.testing.assert_allclose(
        response,
        dielectric_response(x, 1 - f, eps, 16)[::-1],
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("psi", [-0.4, 0, 2, 19])
def test_pure_coulomb_capture(psi):
    """Independent analytic diffusion resistance and Maxwell flux integral."""
    radius = 80e-9
    scale = COULOMB_LENGTH_T / (298 * 2 * radius)
    z = np.sqrt(abs(psi) / scale)
    ec, ef = pair_enhancements(radius, radius, z, np.sign(psi) * z, 298, 0, 1)
    assert ec == pytest.approx(psi / np.expm1(psi) if psi else 1.0)
    assert ef == pytest.approx(np.exp(-psi) if psi > 0 else 1 - psi)


def test_factory_hard_sphere_limit_and_pair_symmetry():
    """Compare the zero-force family against Particula's dimensionless route."""
    r = np.array([70e-9, 92e-9, 130e-9])
    m = 4 * np.pi / 3 * r**3 * 2160
    base = dyn.get_hard_sphere_kernel_via_system_state(
        r, m, np.zeros(3), 298, 101325
    )
    evaluate = dyn.get_van_der_waals_kernel_via_system_state
    for hamaker, tolerance in [(0, 3e-15), (1e-35, 4e-4)]:
        np.testing.assert_allclose(
            evaluate(r, m, [0, 0, 0], 298, 101325, hamaker, 5.9),
            base,
            rtol=tolerance,
            atol=0,
        )
    z = np.array([-1, 0, 1])
    value = evaluate(r, m, z, 298, 101325, 8.9e-20, 5.9)
    np.testing.assert_array_equal(value, value.T)
    np.testing.assert_allclose(
        value, evaluate(r, m, -z, 298, 101325, 8.9e-20, 5.9), rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        value,
        evaluate(r[::-1], m[::-1], z[::-1], 298, 101325, 8.9e-20, 5.9)[
            ::-1, ::-1
        ],
        rtol=1e-14,
        atol=0,
    )
    assert (
        value[1, 2] > evaluate(r, m, [0, 0, 0], 298, 101325, 8.9e-20, 5.9)[1, 2]
    )


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_native_density_rate_and_step(kind):
    """Compare loss to the independent per-bin or per-radius measure."""
    r = np.geomspace(50e-9, 250e-9, 8)
    number = np.linspace(1.0, 2.0, 8) * 1e6
    if kind == "continuous_pdf":
        number = number / r
    particles = ParticleData(
        masses=(4 * np.pi / 3 * r**3 * 1700)[None, :, None],
        concentration=number[None, :].copy(),
        density=np.array([1700.0]),
        charge=np.zeros((1, 8)),
        volume=np.ones(1),
        distribution_type=kind,
        radius_grid=r if kind == "continuous_pdf" else None,
    )
    s = dyn.CoagulationFactory().get_strategy(
        "van_der_waals",
        {
            "distribution_type": kind,
            "hamaker_constant": 6.4e-20,
            "relative_permittivity": 5.9,
        },
    )
    k = s.kernel(particles, 298, 101325)
    expected = number * (
        k @ number
        if kind == "discrete"
        else np.trapezoid(k * number, r, axis=1)
    )
    np.testing.assert_allclose(
        s.loss_rate(particles, k), expected, rtol=1e-13, atol=0
    )
    net = s.net_rate(particles, 298, 101325)
    assert s.step(particles, 298, 101325, 0.1) is particles
    np.testing.assert_allclose(
        particles.concentration[0], number + 0.1 * net, rtol=1e-13, atol=0
    )


def test_native_rejection_and_composition():
    """Invalid attraction and thermal double counting reject before mutation."""
    s = dyn.VanDerWaalsCoagulationStrategy("particle_resolved", 8.9e-20, 5.9)
    nested = dyn.CombineCoagulationStrategy([s])
    with pytest.raises(ValueError, match="already includes"):
        dyn.CombineCoagulationStrategy(
            [nested, dyn.BrownianCoagulationStrategy("particle_resolved")]
        )
    p = state().particles
    p.charge[:] = [[2, -2]]
    initial = copy.deepcopy(p)
    rng = copy.deepcopy(s.random_generator.bit_generator.state)
    with pytest.raises(ValueError, match="opposite-charge"):
        s.step(p, 298.0, 101325.0, 1.0)
    assert s.random_generator.bit_generator.state == rng
    np.testing.assert_array_equal(p.masses, initial.masses)
    p.masses[0, 1] = 0
    p.concentration[0, 1] = 0
    k = s.kernel(p, 298.0, 101325.0)
    assert k[0, 0] > 0
    np.testing.assert_array_equal(k[1, :], 0)


@pytest.mark.parametrize("combine", [False, True])
def test_resolved_domain_only_checks_distinct_slots(combine):
    """A singleton charged state may collide with a larger neutral sphere."""
    radii = np.array([5.0, 50.0]) * 1e-9
    density = 2000.0
    masses = 4 * np.pi / 3 * density * radii**3
    particles = ParticleData(
        masses=masses[None, :, None],
        density=np.array([density]),
        charge=np.array([[4.0, 0.0]]),
        concentration=np.ones((1, 2)),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    capture = dyn.VanDerWaalsCoagulationStrategy("particle_resolved", 0.0, 1.0)
    # With zero dispersion and permittivity one, this charged-neutral pair
    # is exactly hard-sphere. Only the hypothetical small self-pair is invalid.
    expected = dyn.get_hard_sphere_kernel_via_system_state(
        radii, masses, np.zeros(2), 298.0, 101325.0
    )[0, 1]
    strategy = capture
    if combine:
        settling = dyn.SedimentationCoagulationStrategy("particle_resolved")
        expected += settling.kernel(particles, 298.0, 101325.0)[0, 1]
        strategy = dyn.CombineCoagulationStrategy(
            [dyn.CombineCoagulationStrategy([capture]), settling]
        )
    strategy.random_generator = np.random.default_rng(42)
    strategy.step(particles, 298.0, 101325.0, 0.999999 / expected)
    assert np.count_nonzero(particles.concentration) == 1
    assert particles.masses.sum() == pytest.approx(masses.sum(), abs=0)
    assert particles.charge.sum() == 4


def test_real_pairs_between_repeated_charged_states_keep_domain_checks():
    """Identical material in distinct slots is still a real encounter."""
    mass = 4 * np.pi / 3 * 2000 * (5e-9) ** 3
    particles = ParticleData(
        masses=np.full((1, 2, 1), mass),
        density=np.array([2000.0]),
        charge=np.full((1, 2), 4.0),
        concentration=np.ones((1, 2)),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    capture = dyn.VanDerWaalsCoagulationStrategy("particle_resolved", 0.0, 1.0)
    rng = copy.deepcopy(capture.random_generator.bit_generator.state)
    with pytest.raises(ValueError, match="capture table domain"):
        capture.step(particles, 298.0, 101325.0, 1.0)
    assert capture.random_generator.bit_generator.state == rng
    np.testing.assert_array_equal(particles.concentration, [[1.0, 1.0]])


def test_resolved_identical_material_retains_cross_slot_rates():
    """A duplicate material keeps the physical rate on off-diagonal slots."""
    radii = np.array([10.0, 40.0, 10.0]) * 1e-9
    masses = 4 * np.pi / 3 * 2000 * radii**3
    inputs = radii, masses, [1.0, 0.0, 1.0], 298.0, 101325.0, 8.9e-20, 5.9
    evaluate = dyn.get_van_der_waals_kernel_via_system_state
    full = evaluate(*inputs)
    distinct = evaluate(*inputs, exclude_self_pairs=True)
    assert full[0, 0] == full[0, 2] > 0
    np.fill_diagonal(full, 0)
    np.testing.assert_array_equal(distinct, full)
