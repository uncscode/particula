# %% [markdown]
# # Follow a source marker through collisions and removal
#
# Can a tracer still tell you where particle material came from after particles
# merge or storage is resampled? This CPU example registers an additive per-particle
# marker and checks its represented inventory, Σ nᵢ qᵢ. The marker is dimensionless
# and extensive per physical particle; it is not a material species and does not
# change aerosol physics.
#
# The first case combines actual Brownian mergers with dilution. It uses 256
# particles with radii 20–80 nm, number density 10¹⁵ m⁻³, and an alternating 0/1
# tag. The high abundance makes mergers observable in a short 15 s calculation.
# Merging adds partner tags; dilution at 0.03 s⁻¹ changes counts and records the
# removed inventory while preserving each survivor's per-particle tag.
#
# The second, independent case forces fixed-capacity resampling during nucleation
# and assigns newborn tags. Read [represented counts](../../tutorials/building-an-aerosol/particles/distribution_tutorial.py)
# and [bounded sources](../particle-processes/sources-and-dilution/cpu_nucleation.py) first. These cases check
# CPU bookkeeping; they make no CPU/GPU stochastic-trajectory claim.

# %%
"""Registered particle-field inventory and redistribution examples."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import Aerosol, FieldSpec, plotting
from particula.dynamics import (
    ActivationNucleationBuilder,
    BrownianCoagulationStrategy,
    Coagulation,
    Dilution,
    DilutionStrategy,
    Nucleation,
    NucleationCommitConfig,
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.execution import CPUExecutionContext
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from particula.particles.exhaustion import ExhaustionControls
from particula.util.constants import AVOGADRO_NUMBER

TAG = "demo.tag"


def state(masses, counts, volume, gas_concentration):
    """Construct fixed-capacity resolved state with precursor and inert gas."""
    return Aerosol(
        particles=ParticleData(
            masses=np.asarray(masses, dtype=float).reshape(1, -1, 1),
            concentration=np.asarray(counts, dtype=float).reshape(1, -1),
            charge=np.zeros((1, len(counts))),
            density=np.array([1000.0]),
            volume=np.array([volume]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            name=["precursor", "inert"],
            molar_mass=np.array([0.2, 0.029]),
            concentration=np.array([[gas_concentration, 1e-6]]),
            partitioning=np.array([True, False]),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.ones((1, 2)),
        ),
    )


def tag_inventory(aerosol):
    """Independently compute represented tag equivalents in one box."""
    return float(np.sum(aerosol.particles.concentration * aerosol.fields[TAG]))


def coagulation_case():
    """Follow nonzero mergers and count removal without losing tag accounting."""
    slots = 256
    radius = np.geomspace(20e-9, 80e-9, slots)
    masses = 4 * np.pi / 3 * 1000 * radius**3
    aerosol = state(masses, np.ones(slots), slots / 1e15, 0.0)
    tags = (np.arange(slots) % 2 == 0).astype(float)[None, :]
    aerosol.fields.add(
        FieldSpec(
            TAG, ("box", "particle"), "1", particle_policy="additive", minimum=0
        ),
        tags,
    )
    initial_inventory = tag_inventory(aerosol)
    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    process = process | Dilution(DilutionStrategy(0.03))
    context = CPUExecutionContext(root_seed=47, box_ids=("chamber",))
    times = np.arange(31) * 0.5
    inventories, removals, active = [initial_inventory], [0.0], [slots]
    for time in times[:-1]:
        process.execute(aerosol, 0.5, time=float(time), context=context)
        inventory = tag_inventory(aerosol)
        ledger = aerosol.fields.inventory_changes(TAG)
        removed = float(ledger["removal"][0])
        np.testing.assert_allclose(
            inventory + removed, initial_inventory, rtol=2e-13
        )
        np.testing.assert_array_equal(ledger["source"], 0)
        inventories.append(inventory)
        removals.append(removed)
        active.append(int(np.count_nonzero(aerosol.particles.concentration)))
    assert active[-1] < slots, "the example must contain actual mergers"
    np.testing.assert_allclose(
        inventories, initial_inventory * np.exp(-0.03 * times), rtol=2e-13
    )
    mask = aerosol.particles.concentration[0] > 0
    np.testing.assert_array_equal(aerosol.fields[TAG][0, ~mask], 0)
    # Save compact weighted histograms, not full particle trajectories.
    edges = np.arange(0, int(aerosol.fields[TAG].max()) + 2) - 0.5
    histogram, _ = np.histogram(
        aerosol.fields[TAG][0, mask],
        bins=edges,
        weights=aerosol.particles.concentration[0, mask],
    )
    initial_histogram, _ = np.histogram(tags[0], bins=edges)
    return dict(
        time=times.tolist(),
        inventory=inventories,
        removal=removals,
        active=active,
        initial_inventory=initial_inventory,
        edges=edges.tolist(),
        initial_histogram=initial_histogram.tolist(),
        final_histogram=histogram.tolist(),
    )


# %% [markdown]
# ## Resampling and births have different meanings
#
# Three unit-count particles carry tags 10, 20 and 40. Only one slot is free,
# but the source needs two unit-count newborn slots. Resampling represents the
# three old particles with two count-1.5 survivors. Equal particle masses make
# the native sort order deterministic here. Its exact count-interval overlaps
# give (10 + 0.5×20)/1.5 and (0.5×20 + 40)/1.5 tag units per survivor.
# Each newborn receives 7 tag units from the field's birth default.
#
# The field ledger therefore records 14 added tag units, while native gas and
# particle mass conserve the mapped material independently. These arbitrary tag
# units are not molecular mass. Resampling preserves represented tag inventory,
# but can change the joint size/tag distribution. Equal-count GPU coagulation
# must not be applied to the resulting unequal survivor/newborn counts without
# satisfying its separate admission contract; these are separate examples.


# %%
def source_case():
    """Check exact redistribution, birth defaults and native material balance."""
    aerosol = state(
        [1e-20, 1e-20, 1e-20, 0], [1, 1, 1, 0], 1.0, 8 * 0.2 / AVOGADRO_NUMBER
    )
    aerosol.fields.add(
        FieldSpec(
            TAG,
            ("box", "particle"),
            "1",
            particle_policy="additive",
            initial_value=7,
        ),
        np.array([[10.0, 20.0, 40.0, 0.0]]),
    )
    strategy = (
        ActivationNucleationBuilder()
        .set_parameters(
            {
                "coefficient": 0.25,
                "coefficient_units": "s^-1",
                "coefficient_provenance": "illustrative fixed-capacity source",
                "precursor_number_concentration_lower": 0.0,
                "precursor_number_concentration_lower_units": "1/m^3",
                "precursor_number_concentration_upper": 1e30,
                "precursor_number_concentration_upper_units": "1/m^3",
                "temperature_lower": 200.0,
                "temperature_lower_units": "K",
                "temperature_upper": 400.0,
                "temperature_upper_units": "K",
                "injection_composition": [1, 0],
                "formation_diameter": 1.0,
                "formation_diameter_units": "nm",
            }
        )
        .build()
    )
    process = Nucleation(
        NucleationSourceConfig(
            strategy,
            0,
            NucleationSpeciesMap(tuple(aerosol.gas.name), ((0, 0),)),
        ),
        NucleationCommitConfig(
            maximum_slot_weight=1.0,
            source_charge=0.0,
            exhaustion_controls=ExhaustionControls(True, False),
            requested_scale=np.ones(1),
            minimum_scale=np.ones(1),
            minimum_volume=np.array([0.1]),
            radius_cubed_relative_error=0.01,
        ),
    )

    def material_inventory():
        p = aerosol.particles
        return float(
            np.sum(p.masses[..., 0] * p.concentration)
            + aerosol.gas.concentration[0, 0] * p.volume[0]
        )

    before = material_inventory()
    initial_tags = aerosol.fields[TAG].copy()
    process.execute(aerosol, 1.0)
    np.testing.assert_allclose(
        aerosol.particles.concentration, [[1.5, 1.5, 1, 1]], rtol=1e-14
    )
    np.testing.assert_allclose(
        aerosol.fields[TAG], [[40 / 3, 100 / 3, 7, 7]], rtol=1e-14
    )
    np.testing.assert_allclose(tag_inventory(aerosol), 84, rtol=1e-14)
    ledger = aerosol.fields.inventory_changes(TAG)
    np.testing.assert_allclose(ledger["source"], [14], rtol=1e-14)
    np.testing.assert_array_equal(ledger["removal"], 0)
    np.testing.assert_allclose(
        material_inventory(), before, rtol=1e-12, atol=1e-30
    )
    return dict(
        before=initial_tags[0].tolist(),
        after=aerosol.fields[TAG][0].tolist(),
        counts=aerosol.particles.concentration[0].tolist(),
        inventory=tag_inventory(aerosol),
        added=float(ledger["source"][0]),
    )


def passive_rejection():
    """Undefined label merging rejects without changing caller state."""
    aerosol = state([1e-20, 2e-20], [1, 1], 1e-12, 0.0)
    aerosol.fields.add(
        FieldSpec("demo.label", ("box", "particle"), "1", dtype="int32"),
        np.array([[1, 2]], dtype=np.int32),
    )
    before = aerosol.particles.masses.copy()
    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    try:
        process.execute(aerosol, 1.0)
    except ValueError as error:
        assert "additive" in str(error)
    else:
        raise AssertionError("passive labels unexpectedly merged")
    np.testing.assert_array_equal(aerosol.particles.masses, before)
    np.testing.assert_array_equal(aerosol.fields["demo.label"], [[1, 2]])


def simulate():
    """Run separate collision/removal and resampling/source accounting cases."""
    passive_rejection()
    return dict(coagulation=coagulation_case(), source=source_case())


# %% [markdown]
# ## Read three different accounting views
#
# The first panel tests inventory closure under removal. The second shows how
# mergers change tag amount per particle; its vertical axis is represented count,
# not a normalized probability. The third shows per-particle tag values in storage
# slots after resampling and births. Multiply by the displayed case's counts
# (1.5, 1.5, 1, 1) before summing that panel into an inventory.
#
# `passive_rejection` separately verifies that a label with no additive policy
# is rejected without changing particle masses or labels. The following renderer
# is called by `run_example` after these scientific checks.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Show inventory closure, changed distributions and explicit source tags."""
    coag, source = report["coagulation"], report["source"]
    fig, axes = plotting.subplots(1, 3)
    axes[0].plot(coag["time"], coag["inventory"], label="In-box inventory")
    axes[0].plot(coag["time"], coag["removal"], label="Removed inventory")
    axes[0].plot(
        coag["time"],
        np.asarray(coag["inventory"]) + coag["removal"],
        "k--",
        label="Sum (initial inventory)",
    )
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Represented tag equivalents",
        title="Coagulation + dilution",
    )
    axes[1].stairs(coag["initial_histogram"], coag["edges"], label="Initial")
    axes[1].stairs(coag["final_histogram"], coag["edges"], label="Final")
    axes[1].set(
        xlabel="Tag units per particle",
        ylabel="Represented count",
        title="Distribution after mergers",
    )
    slots = np.arange(4)
    axes[2].bar(slots - 0.18, source["before"], 0.36, label="Before source")
    axes[2].bar(slots + 0.18, source["after"], 0.36, label="After source")
    axes[2].set(
        xlabel="Storage slot",
        ylabel="Tag units per particle",
        title="Resampling and two births",
        xticks=slots,
    )
    for ax in axes:
        plotting.legend(ax)
        ax.grid(alpha=0.2)
    plotting.fit(fig)
    plt.show()


def run_example():
    """Execute all scientific checks and render their compact saved outputs."""
    report = simulate()
    print("Tracer inventory, exact resampling and passive rejection passed.")
    print(
        "Active slots after coagulation:", report["coagulation"]["active"][-1]
    )
    print(
        "Source inventory: 70 +",
        report["source"]["added"],
        "=",
        report["source"]["inventory"],
    )
    plot_results(report)
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# ## Preserve an inventory without overinterpreting a label
#
# The first case checks that retained plus removed tag inventory equals the
# initial 128 units, and that retained inventory follows the analytic dilution
# factor. The second gives 70 old tag units plus 14 newborn units, totaling 84.
# Those are separate experiments; their inventories must not be combined.
#
# This establishes additive-field accounting and a controlled resampling example,
# not preservation of every size–tag correlation. To adapt the tracer to source
# mixing, tag two initial sources and compare tag fraction conditional on size
# before and after mergers. Vary resampling pressure and stochastic realization
# to determine whether an inferred mixing trend survives changes in representation.
# A categorical source name needs an explicit merging rule; the passive-label
# rejection shows why arbitrary labels cannot be silently added.
