"""Compare batched dynamics with independent same-ID single-box executions."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from particula import _cpu_batch
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
    Dilution,
    DilutionStrategy,
    MassCondensation,
    SphericalWallLossStrategy,
    WallLoss,
)
from particula.dynamics.tests.native_nucleation_test import fixture, runnable
from particula.dynamics.tests.native_runnable_test import condensation, state
from particula.execution import CPUExecutionContext


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize(
    "family", ["condensation", "coagulation", "wall", "dilution"]
)
def test_binned_boxes_match_direct_single_box_physics(kind, family):
    """PMF and PDF retain their measures through the runnable boundary."""
    single = state(1.0, kind)
    p = single.particles
    p.masses = np.concatenate([p.masses, p.masses * 4], axis=1)
    p.concentration = np.ones((1, 4))
    p.charge = np.zeros((1, 4))
    if kind == "continuous_pdf":
        p.radius_grid = np.array([1e-8, 2e-8, 4e-8, 8e-8])
    aerosol = batch(single)
    aerosol.particles.volume[:] = 1.0
    strategy, runner = {
        "condensation": (condensation(single.gas), MassCondensation),
        "coagulation": (BrownianCoagulationStrategy(kind), Coagulation),
        "wall": (SphericalWallLossStrategy(0.001, 0.5, kind), WallLoss),
        "dilution": (DilutionStrategy(0.1), Dilution),
    }[family]
    process = runner(strategy)
    reference = _cpu_batch.detached(aerosol)
    for i in range(2):
        box = _cpu_batch.box_view(reference, i)
        if family == "dilution":
            strategy.step(box, 1e-4)
        else:
            kwargs = dict(
                particle=box.particles,
                temperature=float(box.environment.temperature[0]),
                pressure=float(box.environment.pressure[0]),
                time_step=1e-4,
            )
            if family == "condensation":
                kwargs["gas_species"] = box.gas
            strategy.step(**kwargs)
    assert process.rate(aerosol).shape[0] == 2
    process.execute(aerosol, 1e-4)
    for name in process._writer_fields(aerosol):
        np.testing.assert_allclose(
            _cpu_batch.arrays(aerosol)[name],
            _cpu_batch.arrays(reference)[name],
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize(
    "strategy_type", [CondensationIsothermalStaggered, CondensationLatentHeat]
)
def test_condensation_variants_conserve_each_box(strategy_type):
    """Randomized and latent variants conserve mapped inventories per box."""
    aerosol = batch(state())
    base = condensation(aerosol.gas)
    strategy = strategy_type(
        molar_mass=aerosol.gas.molar_mass.copy(),
        activity_strategy=base.activity_strategy,
        surface_strategy=base.surface_strategy,
        vapor_pressure_strategy=base.vapor_pressure_strategy,
        species_map=base.species_map,
        **(
            {"theta_mode": "random", "num_batches": 2}
            if strategy_type is CondensationIsothermalStaggered
            else {"latent_heat": 2e6}
        ),
    )
    process = MassCondensation(strategy)
    masses_before = aerosol.particles.masses.copy()
    particle_before = aerosol.particles.species_mass_inventory.copy()
    before = (
        particle_before
        + aerosol.gas.concentration[:, [2, 1]]
        * aerosol.particles.volume[:, None]
    )
    process.execute(aerosol, 0.01, context=CPUExecutionContext(7, ("a", "b")))
    after = (
        aerosol.particles.species_mass_inventory
        + aerosol.gas.concentration[:, [2, 1]]
        * aerosol.particles.volume[:, None]
    )
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-30)
    if strategy_type is CondensationLatentHeat:
        assert strategy.last_latent_heat_energy is None
        np.testing.assert_allclose(
            process.last_latent_heat_energy,
            np.einsum(
                "bns,bn->b",
                aerosol.particles.masses - masses_before,
                aerosol.particles.concentration,
            )
            * 2e6,
            rtol=1e-10,
            atol=1e-30,
        )


def batch(single):
    """Expand independent state, with distinct environment and gas per box."""
    result = _cpu_batch.detached(single)
    for container, names in (
        (result.particles, ("masses", "concentration", "charge", "volume")),
        (result.gas, ("concentration",)),
        (result.environment, ("temperature", "pressure", "saturation_ratio")),
    ):
        for name in names:
            values = getattr(container, name)
            setattr(container, name, np.concatenate([values, values.copy()]))
    result.particles.volume[1] = 4.0
    result.environment.temperature[1] = 310.0
    result.gas.concentration[1] *= 0.5
    return result


def test_failed_later_wall_box_does_not_advance_data_or_streams():
    """A late physical failure rolls back staged draws as well as storage."""
    aerosol = batch(state())
    reference = _cpu_batch.detached(aerosol)
    process = WallLoss(
        SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
    )
    context = CPUExecutionContext(91, ("a", "b"))
    before = {
        name: value.copy() for name, value in _cpu_batch.arrays(aerosol).items()
    }
    aerosol.particles.masses[1, 0] = 0
    invalid = aerosol.particles.masses.copy()
    with pytest.raises(ValueError, match="positive mass"):
        process.execute(aerosol, 10.0, context=context)
    np.testing.assert_array_equal(aerosol.particles.masses, invalid)
    for name in process._writer_fields(aerosol):
        if name != "particles.masses":
            np.testing.assert_array_equal(
                _cpu_batch.arrays(aerosol)[name], before[name]
            )
    assert context._streams == {}
    aerosol.particles.masses[:] = before["particles.masses"]
    process.execute(aerosol, 10.0, context=context)
    process.execute(
        reference, 10.0, context=CPUExecutionContext(91, ("a", "b"))
    )
    for name in process._writer_fields(aerosol):
        np.testing.assert_array_equal(
            _cpu_batch.arrays(aerosol)[name], _cpu_batch.arrays(reference)[name]
        )


def test_wall_streams_follow_box_identity_across_reordering():
    """Changing lane order preserves each box's state and RNG continuation."""
    original = batch(state())
    reordered = _cpu_batch.detached(original)
    for name, value in _cpu_batch.arrays(reordered).items():
        if name not in (
            "particles.density",
            "gas.molar_mass",
            "gas.partitioning",
        ):
            value[:] = value[::-1].copy()
    process = WallLoss(
        SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
    )
    first = CPUExecutionContext(13, ("a", "b"))
    second = CPUExecutionContext(13, ("b", "a"))
    for _ in range(3):
        process.execute(original, 0.01, context=first)
        process.execute(reordered, 0.01, context=second)
    for name in process._writer_fields(original):
        np.testing.assert_array_equal(
            _cpu_batch.arrays(original)[name],
            _cpu_batch.arrays(reordered)[name][::-1],
        )


@pytest.mark.parametrize(
    "family", ["condensation", "coagulation", "wall", "dilution", "nucleation"]
)
def test_batch_matches_single_boxes_and_preserves_identity(family):
    """All five runners preserve per-box numerical and stream independence."""
    if family == "nucleation":
        single, config = fixture()
        process = runnable(single, config)
        process.commit_config = replace(
            process.commit_config,
            **{
                name: np.repeat(getattr(process.commit_config, name), 2)
                for name in (
                    "requested_scale",
                    "minimum_scale",
                    "minimum_volume",
                )
            },
        )
    else:
        single = state()
        process = {
            "condensation": lambda: MassCondensation(condensation(single.gas)),
            "coagulation": lambda: Coagulation(
                BrownianCoagulationStrategy("particle_resolved")
            ),
            "wall": lambda: WallLoss(
                SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
            ),
            "dilution": lambda: Dilution(DilutionStrategy(0.1)),
        }[family]()
    aerosol = batch(single)
    originals = _cpu_batch.arrays(aerosol)
    references = []
    for box in range(2):
        reference = _cpu_batch.detached(_cpu_batch.box_view(aerosol, box))
        worker = deepcopy(process)
        if family == "nucleation":
            worker.commit_config = replace(
                process.commit_config,
                **{
                    name: getattr(process.commit_config, name)[box : box + 1]
                    for name in (
                        "requested_scale",
                        "minimum_scale",
                        "minimum_volume",
                    )
                },
            )
        worker.execute(
            reference, 0.01, context=CPUExecutionContext(17, (str(box),))
        )
        references.append(reference)
    if family == "coagulation":
        # Resolved coagulation has no continuous number-rate API.
        with pytest.raises(ValueError, match="distribution type"):
            process.rate(aerosol)
    else:
        assert process.rate(aerosol).shape[0] == 2
    assert (
        process.execute(
            aerosol, 0.01, context=CPUExecutionContext(17, ("0", "1"))
        )
        is aerosol
    )
    for name, values in _cpu_batch.arrays(aerosol).items():
        assert values is originals[name]
        if name in process._writer_fields(aerosol):
            expected = np.concatenate(
                [_cpu_batch.arrays(ref)[name] for ref in references]
            )
            np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
@pytest.mark.parametrize(
    "mechanism",
    [
        "brownian",
        "sedimentation",
        "shear",
        "dns",
        "hard",
        "dyachkov",
        "gatti",
        "gopal",
        "chahl",
        "combined",
    ],
)
def test_all_coagulation_mechanisms_match_independent_boxes(kind, mechanism):
    """Every existing material kernel keeps per-box state and environment."""
    from particula.dynamics.coagulation.coagulation_strategy.tests.native_rates_test import (
        _particle,
        _strategy,
    )

    single = state(1.0, kind)
    single.particles = _particle(kind)
    if kind == "particle_resolved":
        single.particles.concentration[:] = 2.0
    aerosol = batch(single)
    if kind != "particle_resolved":
        aerosol.particles.volume[:] = 1
    aerosol.environment.pressure[1] = 90000
    reference = _cpu_batch.detached(aerosol)
    context = CPUExecutionContext(123, ("a", "b"))
    for i in range(2):
        box = _cpu_batch.box_view(reference, i)
        strategy = _strategy(mechanism, kind)
        strategy.random_generator = context._begin(2).generator(
            "coagulation", i
        )
        strategy.step(
            box.particles,
            float(box.environment.temperature[0]),
            float(box.environment.pressure[0]),
            1e-4,
        )
    process = Coagulation(_strategy(mechanism, kind))
    process.execute(aerosol, 1e-4, context=context)
    for name in process._writer_fields(aerosol):
        np.testing.assert_allclose(
            _cpu_batch.arrays(aerosol)[name],
            _cpu_batch.arrays(reference)[name],
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
@pytest.mark.parametrize(
    "geometry", ["rectangular", "charged_spherical", "charged_rectangular"]
)
def test_wall_geometries_match_independent_boxes(kind, geometry):
    """Charged and rectangular variants use each box's own environment."""
    from particula.dynamics import (
        ChargedWallLossStrategy,
        RectangularWallLossStrategy,
    )

    aerosol = batch(state(1.0, kind))
    if kind != "particle_resolved":
        aerosol.particles.volume[:] = 1
    aerosol.environment.pressure[1] = 90000
    if geometry == "rectangular":
        strategy = RectangularWallLossStrategy(0.001, (1.0, 1.0, 1.0), kind)
    else:
        strategy = ChargedWallLossStrategy(
            0.001,
            chamber_geometry=geometry.removeprefix("charged_"),
            chamber_radius=0.5,
            chamber_dimensions=(1.0, 1.0, 1.0),
            wall_potential=0.05,
            distribution_type=kind,
        )
    process = WallLoss(strategy)
    reference = _cpu_batch.detached(aerosol)
    context = CPUExecutionContext(42, ("a", "b"))
    for i in range(2):
        box = _cpu_batch.box_view(reference, i)
        strategy = deepcopy(process.wall_loss_strategy)
        strategy.random_generator = context._begin(2).generator("wall_loss", i)
        strategy.step(
            box.particles,
            float(box.environment.temperature[0]),
            float(box.environment.pressure[0]),
            0.01,
        )
    process.execute(aerosol, 0.01, context=context)
    for name in process._writer_fields(aerosol):
        np.testing.assert_array_equal(
            _cpu_batch.arrays(aerosol)[name],
            _cpu_batch.arrays(reference)[name],
        )


@pytest.mark.parametrize("family", ["coagulation", "wall"])
@pytest.mark.parametrize("zero_duration", [True, False])
def test_zero_work_preserves_random_continuation(family, zero_duration):
    """Empty populations and zero-duration calls do not consume randomness."""
    aerosol = batch(state())
    process = (
        Coagulation(BrownianCoagulationStrategy("particle_resolved"))
        if family == "coagulation"
        else WallLoss(
            SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
        )
    )
    context = CPUExecutionContext(17, ("a", "b"))
    reference = CPUExecutionContext(17, ("a", "b"))
    if not zero_duration:
        aerosol.particles.masses[:] = 0
        aerosol.particles.concentration[:] = 0
        aerosol.particles.charge[:] = 0
    process.execute(aerosol, 0 if zero_duration else 1, context=context)
    namespace = "coagulation" if family == "coagulation" else "wall_loss"
    for i in range(2):
        np.testing.assert_array_equal(
            context._begin(2).generator(namespace, i).random(10),
            reference._begin(2).generator(namespace, i).random(10),
        )


def test_late_nucleation_capacity_failure_preserves_every_box():
    """A full later box cannot commit earlier source injection or gas loss."""
    from particula.particles.exhaustion import ExhaustionControls

    single, config = fixture()
    process = runnable(single, config)
    process.commit_config = replace(
        process.commit_config,
        exhaustion_controls=ExhaustionControls(False, False),
        **{
            name: np.repeat(getattr(process.commit_config, name), 2)
            for name in ("requested_scale", "minimum_scale", "minimum_volume")
        },
    )
    aerosol = batch(single)
    aerosol.particles.masses[1, :] = aerosol.particles.masses[1, 0]
    aerosol.particles.concentration[1, :] = 2
    before = {
        name: value.copy() for name, value in _cpu_batch.arrays(aerosol).items()
    }
    with pytest.raises(ValueError):
        process.execute(aerosol, 1)
    for name, value in _cpu_batch.arrays(aerosol).items():
        np.testing.assert_array_equal(value, before[name])
