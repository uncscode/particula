"""Facade-free native coagulation laws, population measures and admission."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics import (
    BrownianCoagulationStrategy,
    ChargedCoagulationStrategy,
    CombineCoagulationStrategy,
    SedimentationCoagulationStrategy,
    TurbulentDNSCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
)
from particula.dynamics.coagulation import (
    charged_dimensional_kernel as charged_laws,
)
from particula.dynamics.coagulation import charged_kernel_strategy as charged
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    _get_concentration,
    _get_effective_density,
    _get_mass,
    _get_radius,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008 import (
    get_turbulent_dns_kernel_ao2008_via_system_state,
)
from particula.particles import ParticleData
from particula.util import constants as c

CHARGED = {
    "hard": (
        charged.HardSphereKernelStrategy,
        "get_hard_sphere_kernel_via_system_state",
    ),
    "dyachkov": (
        charged.CoulombDyachkov2007KernelStrategy,
        "get_coulomb_kernel_dyachkov2007_via_system_state",
    ),
    "gatti": (
        charged.CoulombGatti2008KernelStrategy,
        "get_coulomb_kernel_gatti2008_via_system_state",
    ),
    "gopal": (
        charged.CoulombGopalakrishnan2012KernelStrategy,
        "get_coulomb_kernel_gopalakrishnan2012_via_system_state",
    ),
    "chahl": (
        charged.CoulumbChahl2019KernelStrategy,
        "get_coulomb_kernel_chahl2019_via_system_state",
    ),
}
MECHANISMS = ["brownian", "sedimentation", "shear", "dns", *CHARGED, "combined"]
REPRESENTATIONS = [
    ("discrete", 1.0),
    ("continuous_pdf", 1.0),
    ("particle_resolved", 0.25),
    ("particle_resolved", 1.0),
    ("particle_resolved", 4.0),
]


def _particle(kind="discrete", volume=1.0):
    """Material composition sets physical size independently of PDF coordinate."""
    return ParticleData(
        masses=np.array([[[1.0, 2.0], [4.0, 3.0], [10.0, 9.0], [25.0, 30.0]]])
        * 1e-18,
        concentration=np.array([[0.5, 2.0, 4.0, 10.0]])
        * (1e13 if kind == "continuous_pdf" else 1.0),
        charge=np.array([[1.0, -1.0, 2.0, 0.0]]),
        density=np.array([1000.0, 1600.0]),
        volume=np.array([volume]),
        distribution_type=kind,
        radius_grid=np.array([1.0, 2.0, 4.0, 8.0]) * 1e-7
        if kind == "continuous_pdf"
        else None,
    )


def _strategy(name, kind):
    if name == "brownian":
        return BrownianCoagulationStrategy(kind)
    if name == "sedimentation":
        return SedimentationCoagulationStrategy(kind)
    if name == "shear":
        return TurbulentShearCoagulationStrategy(kind, 0.01, 1.225)
    if name == "dns":
        return TurbulentDNSCoagulationStrategy(kind, 0.01, 1.225, 23.0, 0.1)
    if name == "combined":
        return CombineCoagulationStrategy(
            [_strategy("brownian", kind), _strategy("shear", kind)]
        )
    return ChargedCoagulationStrategy(kind, CHARGED[name][0]())


def _physical(particle, temperature=298.15, pressure=101325.0):
    """Independently derive per-particle geometry and gas transport quantities."""
    masses = np.sum(particle.masses[0], axis=1)
    volumes = np.sum(particle.masses[0] / particle.density, axis=1)
    # Use a cube root, not a power with a rounded 1/3 exponent. DNS variance
    # cancellation can amplify its several-ulp radius bias across CPU builds.
    # Inputs remain independently derived from raw species masses/densities.
    radius = np.cbrt(3 * volumes / (4 * np.pi))
    # The retained CPU effective-density contract is mass-weighted arithmetic
    # density, distinct from the volume used to calculate physical radius.
    density = np.sum(particle.masses[0] * particle.density, axis=1) / masses
    mu = (
        c.REF_VISCOSITY_AIR_STP
        * (temperature / c.REF_TEMPERATURE_STP) ** 1.5
        * (c.REF_TEMPERATURE_STP + c.SUTHERLAND_CONSTANT)
        / (temperature + c.SUTHERLAND_CONSTANT)
    )
    path = (
        2
        * mu
        / pressure
        / np.sqrt(
            8 * c.MOLECULAR_WEIGHT_AIR / (np.pi * c.GAS_CONSTANT * temperature)
        )
    )
    kn = path / radius
    slip = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
    return masses, radius, density, mu, slip


def _reference(name, particle, temperature=298.15, pressure=101325.0):
    """Independent simple laws; unchanged charged/DNS laws receive derived inputs.

    Charged and DNS checks establish native input/normalization correctness,
    not an independent rederivation of those existing complex physical laws.
    Their lower-level scientific regression suites remain part of the gate.
    """
    mass, radius, density, mu, slip = _physical(particle, temperature, pressure)
    pair_radius = radius[:, None] + radius[None, :]
    if name == "brownian":
        diffusion = (
            c.BOLTZMANN_CONSTANT
            * temperature
            * slip
            / (6 * np.pi * mu * radius)
        )
        speed = np.sqrt(8 * c.BOLTZMANN_CONSTANT * temperature / (np.pi * mass))
        path = 8 * diffusion / (np.pi * speed)
        collection = (
            (2 * radius + path) ** 3 - (4 * radius**2 + path**2) ** 1.5
        ) / (6 * radius * path) - 2 * radius
        pair_diff = diffusion[:, None] + diffusion[None, :]
        pair_speed = np.sqrt(speed[:, None] ** 2 + speed[None, :] ** 2)
        pair_g = np.sqrt(collection[:, None] ** 2 + collection[None, :] ** 2)
        return (
            4
            * np.pi
            * pair_diff
            * pair_radius
            / (
                pair_radius / (pair_radius + pair_g)
                + 4 * pair_diff / (pair_radius * pair_speed)
            )
        )
    if name == "sedimentation":
        # Existing strategy uses arithmetic mean effective density, not a
        # concentration-weighted or material-density average.
        velocity = (
            2
            * radius**2
            * np.mean(density)
            * c.STANDARD_GRAVITY
            * slip
            / (9 * mu)
        )
        return (
            np.pi
            * pair_radius**2
            * np.abs(velocity[:, None] - velocity[None, :])
        )
    if name == "shear":
        return (
            np.sqrt(np.pi * 0.01 / (120 * (mu / 1.225)))
            * (2 * pair_radius) ** 3
        )
    if name == "combined":
        return _reference(
            "brownian", particle, temperature, pressure
        ) + _reference("shear", particle, temperature, pressure)
    if name == "dns":
        return get_turbulent_dns_kernel_ao2008_via_system_state(
            particle_radius=radius,
            particle_density=float(np.mean(density)),
            fluid_density=1.225,
            temperature=temperature,
            re_lambda=23.0,
            relative_velocity=0.1,
            turbulent_dissipation=0.01,
        )
    return getattr(charged_laws, CHARGED[name][1])(
        particle_radius=radius,
        particle_mass=mass,
        particle_charge=particle.charge[0],
        temperature=temperature,
        pressure=pressure,
    )


@pytest.mark.parametrize("name", MECHANISMS)
@pytest.mark.parametrize("kind,volume", REPRESENTATIONS)
def test_native_kernels_preserve_physical_laws_and_population_independence(
    name, kind, volume
):
    """Every supported mechanism reads physical kg/radius/charge, never N*m."""
    particle = _particle(kind, volume)
    strategy = _strategy(name, kind)
    before = [
        a.copy()
        for a in (
            particle.masses,
            particle.concentration,
            particle.density,
            particle.charge,
            particle.volume,
        )
    ]
    result = strategy.kernel(particle, 298.15, 101325.0)
    expected = _reference(name, particle)
    npt.assert_allclose(result, expected, rtol=2e-12, atol=1e-30)
    assert (
        result.shape == (4, 4)
        and np.all(np.isfinite(result))
        and np.all(result >= 0)
    )
    particle.concentration *= 7
    npt.assert_allclose(
        strategy.kernel(particle, 298.15, 101325.0), result, rtol=0, atol=0
    )
    particle.concentration /= 7
    for actual, saved in zip(
        (
            particle.masses,
            particle.concentration,
            particle.density,
            particle.charge,
            particle.volume,
        ),
        before,
        strict=True,
    ):
        npt.assert_array_equal(actual, saved)


@pytest.mark.parametrize("kind,volume", REPRESENTATIONS)
def test_native_helpers_separate_particle_mass_from_density(kind, volume):
    """Native helpers retain mass per particle and use the declared density."""
    particle = _particle(kind, volume)
    mass, radius, density, _, _ = _physical(particle)
    npt.assert_allclose(_get_mass(particle), mass, rtol=0, atol=0)
    npt.assert_allclose(_get_radius(particle), radius, rtol=1e-15, atol=0)
    npt.assert_allclose(
        _get_effective_density(particle), density, rtol=1e-15, atol=0
    )
    expected = (
        particle.concentration[0] / volume
        if kind == "particle_resolved"
        else particle.concentration[0]
    )
    npt.assert_array_equal(_get_concentration(particle), expected)
    assert not np.shares_memory(
        _get_concentration(particle), particle.concentration
    )


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_rate_measures_and_gain_quadrature_have_independent_expected_values(
    kind, monkeypatch
):
    """Constant PDF*kernel makes the spline result independently known."""
    particle = _particle(kind)
    strategy = BrownianCoagulationStrategy(kind)
    radius = (
        particle.radius_grid
        if kind == "continuous_pdf"
        else _physical(particle)[1]
    )
    widths = np.diff(radius, append=2 * radius[-1] - radius[-2])
    pdf = 3e12
    particle.concentration[0] = (
        pdf if kind == "continuous_pdf" else pdf * widths
    )
    kernel_value = 2e-15
    kernel = np.full((4, 4), kernel_value)
    before = particle.concentration.copy()
    if kind == "continuous_pdf":
        expected_loss = np.full(
            4, kernel_value * pdf**2 * (radius[-1] - radius[0])
        )
    else:
        expected_loss = (
            kernel_value
            * particle.concentration[0]
            * sum(particle.concentration[0])
        )
    expected_gain = []
    for target in radius:
        integration = np.linspace(0.0, target / 2 ** (1 / 3), 4)
        partner = (target**3 - integration**3) ** (1 / 3)
        values = kernel_value * pdf**2 / partner**2
        expected_gain.append(
            target**2
            * sum((values[:-1] + values[1:]) / 2 * np.diff(integration))
        )
    expected_gain = np.array(expected_gain)
    if kind == "discrete":
        expected_gain *= widths
    npt.assert_allclose(
        strategy.loss_rate(particle, kernel), expected_loss, rtol=1e-14, atol=0
    )
    npt.assert_allclose(
        strategy.gain_rate(particle, kernel), expected_gain, rtol=1e-14, atol=0
    )
    monkeypatch.setattr(strategy, "kernel", lambda **_: kernel)
    npt.assert_allclose(
        strategy.net_rate(particle, 298.15, 101325.0),
        expected_gain - expected_loss,
        rtol=1e-13,
        atol=0,
    )
    npt.assert_array_equal(particle.concentration, before)


@pytest.mark.parametrize("name", MECHANISMS)
@pytest.mark.parametrize(
    "fault",
    [
        "kind",
        "volume",
        "mass",
        "density",
        "charge",
        "count",
        "temperature",
        "pressure",
        "multibox",
        "schema",
        "empty_mass",
    ],
)
def test_native_kernel_rejection_is_read_only(name, fault):
    """Mutated native state/configuration rejects without changing arrays."""
    particle = _particle()
    strategy = _strategy(name, "discrete")
    temperature, pressure = 298.15, 101325.0
    if fault == "kind":
        strategy.distribution_type = "particle_resolved"
    elif fault == "temperature":
        temperature = True
    elif fault == "pressure":
        pressure = np.array([101325.0])
    elif fault == "multibox":
        particle = replace(
            particle,
            masses=np.repeat(particle.masses, 2, axis=0),
            concentration=np.repeat(particle.concentration, 2, axis=0),
            charge=np.repeat(particle.charge, 2, axis=0),
            volume=np.ones(2),
        )
    elif fault == "schema":
        particle.masses = particle.masses.astype(np.float32)
    elif fault == "empty_mass":
        particle.masses[0, 0] = 0
    else:
        array = getattr(
            particle,
            {"mass": "masses", "count": "concentration"}.get(fault, fault),
        )
        array.flat[0] = 4 if fault == "volume" else np.nan
    arrays = [
        particle.masses,
        particle.concentration,
        particle.density,
        particle.charge,
        particle.volume,
    ]
    before = [a.copy() for a in arrays]
    with pytest.raises(ValueError):
        strategy.kernel(particle, temperature, pressure)
    for a, old in zip(arrays, before, strict=True):
        npt.assert_array_equal(a, old)


@pytest.mark.parametrize("name", MECHANISMS)
@pytest.mark.parametrize("empty", [False, True])
def test_native_kernels_keep_zero_mass_slots_and_empty_layout(name, empty):
    """Empty slots retain zero interaction rows and their original indices."""
    particle = _particle("particle_resolved", 0.25)
    if empty:
        particle = replace(
            particle,
            masses=particle.masses[:, :0, :].copy(),
            concentration=particle.concentration[:, :0].copy(),
            charge=particle.charge[:, :0].copy(),
        )
    else:
        particle.masses[0, 1] = 0
        particle.concentration[0, 1] = 0
        particle.charge[0, 1] = 0
    kernel = _strategy(name, "particle_resolved").kernel(
        particle, 298.15, 101325.0
    )
    assert kernel.shape == (particle.n_particles, particle.n_particles)
    if not empty:
        npt.assert_array_equal(kernel[1], 0)
        npt.assert_array_equal(kernel[:, 1], 0)
        selected = replace(
            particle,
            masses=particle.masses[:, [0, 2, 3], :],
            concentration=particle.concentration[:, [0, 2, 3]],
            charge=particle.charge[:, [0, 2, 3]],
        )
        npt.assert_allclose(
            kernel[np.ix_([0, 2, 3], [0, 2, 3])],
            _reference(name, selected),
            rtol=2e-12,
            atol=1e-30,
        )


@pytest.mark.parametrize("method", ["gain_rate", "loss_rate"])
@pytest.mark.parametrize(
    "fault",
    [
        "negative",
        "nan",
        "infinite",
        "shape",
        "complex",
        "bool",
        "grid",
        "kind",
        "volume",
    ],
)
def test_native_rates_validate_kernel_and_current_population(method, fault):
    """Supplied rate matrices and mutable PDF metadata reject without writes."""
    particle = _particle("continuous_pdf")
    strategy = _strategy("brownian", "continuous_pdf")
    kernel = np.full((4, 4), 2e-15)
    if fault in ("negative", "nan", "infinite"):
        kernel[0, 0] = {"negative": -1.0, "nan": np.nan, "infinite": np.inf}[
            fault
        ]
    elif fault == "shape":
        kernel = kernel[:, :1]
    elif fault == "complex":
        kernel = kernel.astype(complex)
    elif fault == "bool":
        kernel = kernel.astype(bool)
    elif fault == "grid":
        particle.radius_grid[0] = particle.radius_grid[1]
    elif fault == "kind":
        strategy.distribution_type = "discrete"
    else:
        particle.volume[0] = 4.0
    before = particle.masses.copy(), particle.concentration.copy()
    with pytest.raises(ValueError):
        getattr(strategy, method)(particle, kernel)
    npt.assert_array_equal(particle.masses, before[0])
    npt.assert_array_equal(particle.concentration, before[1])


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("nodes", [2, 3])
def test_native_gain_preserves_existing_cubic_grid_requirement(kind, nodes):
    """Shared metadata does not infer a new low-order interpolation algorithm."""
    particle = _particle(kind)
    particle = replace(
        particle,
        masses=particle.masses[:, :nodes].copy(),
        concentration=particle.concentration[:, :nodes].copy(),
        charge=particle.charge[:, :nodes].copy(),
        radius_grid=particle.radius_grid[:nodes].copy()
        if kind == "continuous_pdf"
        else None,
    )
    strategy = _strategy("brownian", kind)
    kernel = np.full((nodes, nodes), 2e-15)
    with pytest.raises(ValueError, match="at least four"):
        strategy.gain_rate(particle, kernel)
    assert np.all(strategy.loss_rate(particle, kernel) > 0)
    particle.concentration.fill(0.0)
    npt.assert_array_equal(
        strategy.gain_rate(particle, kernel), np.zeros(nodes)
    )
    npt.assert_array_equal(
        strategy.loss_rate(particle, kernel), np.zeros(nodes)
    )


@pytest.mark.parametrize("method", ["gain_rate", "loss_rate", "net_rate"])
def test_resolved_rate_api_remains_unsupported(method):
    """Resolved densities do not create deterministic gain/loss support."""
    particle = _particle("particle_resolved", 0.25)
    strategy = _strategy("brownian", "particle_resolved")
    with pytest.raises(ValueError, match="Invalid distribution type"):
        if method == "net_rate":
            strategy.net_rate(particle, 298.15, 101325.0)
        else:
            getattr(strategy, method)(particle, np.full((4, 4), 2e-15))


@pytest.mark.parametrize(
    "name,field",
    [
        ("shear", "fluid_density"),
        ("shear", "turbulent_dissipation"),
        ("dns", "fluid_density"),
        ("dns", "turbulent_dissipation"),
        ("dns", "reynolds_lambda"),
        ("dns", "relative_velocity"),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, True, np.array([0.1])])
def test_native_mutated_mechanism_controls_reject(name, field, value):
    """Validation is performed again after users mutate process configuration."""
    particle = _particle()
    strategy = _strategy(name, "discrete")
    setattr(strategy, field, value)
    before = particle.masses.copy(), particle.concentration.copy()
    with pytest.raises(ValueError):
        strategy.kernel(particle, 298.15, 101325.0)
    npt.assert_array_equal(particle.masses, before[0])
    npt.assert_array_equal(particle.concentration, before[1])


@pytest.mark.parametrize(
    "fault", ["empty", "kind", "cycle", "child", "control"]
)
def test_native_combination_rechecks_children(fault):
    """A mutable composition must remain finite, compatible and acyclic."""
    particle = _particle()
    strategy = _strategy("combined", "discrete")
    if fault == "empty":
        strategy.strategies = []
    elif fault == "kind":
        strategy.strategies[0].distribution_type = "continuous_pdf"
    elif fault == "cycle":
        strategy.strategies.append(strategy)
    elif fault == "child":
        strategy.strategies.append(None)
    else:
        strategy.strategies[1].turbulent_dissipation = np.nan
    before = particle.masses.copy()
    with pytest.raises(ValueError):
        strategy.kernel(particle, 298.15, 101325.0)
    npt.assert_array_equal(particle.masses, before)


def test_native_zero_shear_remains_zero():
    """The existing shear formula admits zero dissipation as no collisions."""
    particle = _particle()
    strategy = _strategy("shear", "discrete")
    strategy.turbulent_dissipation = 0.0
    npt.assert_array_equal(
        strategy.kernel(particle, 298.15, 101325.0), np.zeros((4, 4))
    )


@pytest.mark.parametrize(
    "name", ["hard", "dyachkov", "gatti", "gopal", "chahl"]
)
def test_native_charged_model_requires_compatible_strategy(name):
    """Invalid model replacement rejects even before material calculation."""
    strategy = _strategy(name, "discrete")
    strategy.kernel_strategy = None
    with pytest.raises(ValueError, match="charged kernel strategy"):
        strategy.kernel(_particle(), 298.15, 101325.0)
