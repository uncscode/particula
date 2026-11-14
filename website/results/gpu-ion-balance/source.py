# %% [markdown]
# # Explicit atmospheric ion balance on CUDA
#
# Before asking whether charging helps a rare seed survive, check the ion
# population itself. Here every ion is a unit-count particle of diameter
# **0.5 nm**. Positive and negative ions carry +1 and −1 elementary charges
# and masses of 150 u and 90 u. Their different material densities give the
# same initial diameter without conflating mass with size.
#
# We compare no production (decay) with 2 and 10 ion pairs cm⁻³ s⁻¹. A Poisson
# source creates pairs; opposite ions recombine with coefficient
# α = 1.7 × 10⁻¹² m³ s⁻¹. Neutral products retain both material masses until
# cleanup exports them. There are no aerosols, walls, vapor or bulk ion sinks.
# Thus this example tests source and recombination physics, not attachment.

# %%
"""Validate explicit ion production, recombination and export on real CUDA."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from particula import Aerosol
from particula.dynamics.ions.cleanup import IonNeutralCleanup
from particula.dynamics.ions.coagulation import (
    NEGATIVE_ION_MASS,
    POSITIVE_ION_MASS,
    AtmosphericCoagulation,
)
from particula.dynamics.ions.source import IonPairSource
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

ALPHA = 1.7e-12
VOLUME = 2e-8  # m³ per independent box


def initial_state(boxes=64, capacity=256, initial_pairs=0):
    """Create independent physical boxes with two reserved ion materials."""
    masses = np.zeros((boxes, capacity, 2))
    counts = np.zeros((boxes, capacity))
    charges = np.zeros((boxes, capacity))
    for start, mass, sign, material in (
        (0, POSITIVE_ION_MASS, 1, 0),
        (initial_pairs, NEGATIVE_ION_MASS, -1, 1),
    ):
        stop = start + initial_pairs
        masses[:, start:stop, material] = mass
        counts[:, start:stop] = 1
        charges[:, start:stop] = sign
    ion_volume = np.pi / 6 * (0.5e-9) ** 3
    return Aerosol(
        particles=ParticleData(
            masses,
            counts,
            charges,
            np.array([POSITIVE_ION_MASS, NEGATIVE_ION_MASS]) / ion_volume,
            np.full(boxes, VOLUME),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert air"],
            np.array([0.0289644]),
            np.zeros((boxes, 1)),
            np.array([False]),
        ),
        environment=EnvironmentData(
            np.full(boxes, 298.15),
            np.full(boxes, 101325.0),
            np.zeros((boxes, 1)),
        ),
    )


# %% [markdown]
# ## An independent finite-population reference
#
# In one box there are always n positive and n negative free ions. The exact
# continuous-time birth–death process has transitions
# n → n+1 at rate qV and n → n−1 at rate αn²/V.
# We solve its master equation independently with a sparse matrix exponential.
# The upper state is a numerical truncation; its probability must remain
# below 10⁻¹⁰. This is not the GPU particle algorithm.
#
# The familiar continuum equation dn/dt = q − αn² gives equilibrium
# √(q/α). Finite boxes fluctuate, so their ensemble mean need not exactly equal
# that continuum prediction. Both references are shown. Shading is a pointwise
# 95% normal interval for the mean of independent boxes, not an interval for
# an individual ion trajectory or a simultaneous confidence band.


# %%
def finite_population_reference(q, initial_pairs, times, maximum_pairs=128):
    """Return pair-count moments at uniform times starting at zero.

    The sparse exponential API returns an evenly spaced time grid. Reject
    incompatible inputs rather than labeling its results with different times.
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or times[0] != 0
        or times[-1] <= 0
        or not np.allclose(
            np.diff(times), times[-1] / (times.size - 1), rtol=1e-12, atol=0
        )
    ):
        raise ValueError("times must be uniform, increasing and start at zero")
    if type(maximum_pairs) is not int or maximum_pairs < 1:
        raise ValueError("maximum_pairs must be a positive integer")
    if (
        type(initial_pairs) is not int
        or not 0 <= initial_pairs <= maximum_pairs
    ):
        raise ValueError("initial_pairs must be an integer within the chain")
    if not np.isfinite(q) or q < 0:
        raise ValueError("q must be finite and nonnegative")
    states = np.arange(maximum_pairs + 1, dtype=float)
    birth = np.full(maximum_pairs + 1, q * VOLUME)
    birth[-1] = 0
    death = ALPHA * states**2 / VOLUME
    generator = diags(
        [birth[:-1], -(birth + death), death[1:]],
        [-1, 0, 1],
        format="csc",
    )
    initial = np.zeros(maximum_pairs + 1)
    initial[initial_pairs] = 1
    probabilities = expm_multiply(
        generator, initial, start=0, stop=float(times[-1]), num=len(times)
    )
    assert probabilities[:, -2:].sum(axis=1).max() < 1e-10
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)
    mean = probabilities @ states
    variance = np.maximum(probabilities @ states**2 - mean**2, 0)
    return mean, variance


def run_case(q_cm3, *, boxes=64, capacity=256, time_step=1.0):
    """Run 2000 s, observing only box reductions and cumulative budgets."""
    q = q_cm3 * 1e6  # pairs cm⁻³ s⁻¹ → pairs m⁻³ s⁻¹
    initial_pairs = 20 if q == 0 else 0
    initial = initial_state(boxes, capacity, initial_pairs)
    process = (
        IonPairSource(q, 0, 1, POSITIVE_ION_MASS, NEGATIVE_ION_MASS)
        | AtmosphericCoagulation(0, 1, recombination=ALPHA)
        | IonNeutralCleanup(0, 1)
    )
    times = np.linspace(0, 2000, 41)
    segment_steps = round(50 / time_step)
    assert segment_steps * time_step == 50
    mean, sem = [], []
    compute_seconds = 0.0
    maximum_occupancy = 0
    maximum_budget_error = 0.0
    setup_start = perf_counter()
    with process.prepare(
        initial, device="cuda:0", time_step=time_step, root_seed=821
    ) as simulation:
        simulation.capture()
        simulation.synchronize()
        setup_seconds = perf_counter() - setup_start
        for index in range(len(times)):
            if index:
                start = perf_counter()
                simulation.run(segment_steps)
                simulation.synchronize()
                compute_seconds += perf_counter() - start
            data = simulation.diagnostics()
            np.testing.assert_array_equal(data.charge_concentration, 0)
            # Cleanup leaves only free ions, with exactly equal polarities.
            pairs = data.number_concentration * VOLUME / 2
            mean.append(pairs.mean())
            sem.append(pairs.std(ddof=1) / np.sqrt(boxes))
            maximum_occupancy = max(maximum_occupancy, data.active_slots.max())
            source = simulation.ion_source_diagnostics()
            cleanup = simulation.ion_cleanup_diagnostics()
            before = initial.particles.masses.sum(axis=1)
            supplied = before + source["injected_mass"]
            accounted = (
                data.particle_mass_concentration * VOLUME
                + cleanup["exported_mass"]
            )
            np.testing.assert_allclose(accounted, supplied, rtol=2e-12, atol=0)
            nonzero = supplied > 0
            if np.any(nonzero):
                maximum_budget_error = max(
                    maximum_budget_error,
                    np.max(np.abs(accounted[nonzero] / supplied[nonzero] - 1)),
                )
    reference, variance = finite_population_reference(q, initial_pairs, times)
    mean, sem = np.asarray(mean), np.asarray(sem)
    uncertainty = np.sqrt(variance[1:] / boxes)
    # Fixed acceptance criterion for the stochastic implementation at dt=1 s.
    standardized_error = np.max(np.abs(mean[1:] - reference[1:]) / uncertainty)
    assert standardized_error < 6
    print(
        f"q={q_cm3:g} pairs cm^-3 s^-1: {boxes} boxes, capacity={capacity}, "
        f"dt={time_step:g} s; max mean error={standardized_error:.2f} SE; "
        f"observed peak slots={maximum_occupancy}/{capacity}; "
        f"mass residual={maximum_budget_error:.2e}; "
        f"setup/capture={setup_seconds:.3f} s, "
        f"synchronized stepping={compute_seconds:.3f} s"
    )
    return times, mean, sem, reference


# %% [markdown]
# ## Run the three source conditions
#
# The workload contains 64 independent boxes per condition and 256 allocated
# slots per box. Cleanup runs after coagulation every second. No slot resizing
# or resampling occurs. Exhaustion raises an error instead of dropping ions.
# Timings separate preparation/capture from synchronized stepping and exclude
# the reference solver and analysis. Device exclusivity is not controlled;
# elapsed stepping can include contention from other GPU jobs.
# This small validation workload does not
# establish a CPU speedup or a maximum GPU particle count.


# %%
def run_example():
    """Compare CUDA ensembles with independent references and inspect budgets."""
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires CUDA; no CPU fallback")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    conversion = 1 / VOLUME / 1e6  # pairs per box → ions per polarity per cm³
    for axis, q in zip(axes, [0, 2, 10], strict=True):
        times, mean, sem, reference = run_case(q)
        axis.plot(times, mean * conversion, label="CUDA ensemble")
        axis.fill_between(
            times,
            (mean - 1.96 * sem) * conversion,
            (mean + 1.96 * sem) * conversion,
            alpha=0.25,
        )
        axis.plot(times, reference * conversion, "--", label="Finite-box exact")
        if q:
            continuum = (
                np.sqrt(q * 1e6 / ALPHA)
                * np.tanh(np.sqrt(q * 1e6 * ALPHA) * times)
                / 1e6
            )
        else:
            n0 = 20 / VOLUME
            continuum = n0 / (1 + ALPHA * n0 * times) / 1e6
        axis.plot(times, continuum, ":", label="Continuum")
        axis.set(
            title=f"q = {q} pairs cm⁻³ s⁻¹",
            xlabel="Time [s]",
            ylabel="Free ions per polarity [cm⁻³]",
        )
    axes[0].legend(fontsize=8)
    plt.show()
    return figure


# %% [markdown]
# ## Interpretation and next step
#
# Charge remains zero because production and recombination both act on pairs.
# Mass is conserved after including externally supplied ion mass and exported
# neutral products. The two polarities therefore overlap exactly in this
# closed ion-only example. In an aerosol population their free concentrations
# can differ because attachment removes a particular sign from the free pool.
#
# The source range is an illustrative sensitivity study, not a site-specific
# ionization measurement. The recombination coefficient is fixed at reference
# conditions; atmospheric composition and temperature dependence require a
# broader model. Before a survival claim, add attachment and condensation,
# then verify timestep, capacity, cleanup and ensemble convergence separately.
#
# Primary sources:
# - [Yu (2010), ion-mediated nucleation and ionization sensitivity](https://doi.org/10.1029/2009JD012630).
# - [Rose et al. (2013), cluster ion sources and sinks](https://doi.org/10.5194/acp-13-11573-2013).
# - [Zauner-Wieczorek et al. (2022), ion–ion recombination review](https://doi.org/10.5194/acp-22-12443-2022).
# - [López-Yglesias and Flagan (2013), attachment coefficients](https://doi.org/10.1080/02786826.2013.783684),
#   used by `AtmosphericCoagulation` when aerosols are present; not exercised here.

# %%
if __name__ == "__main__":
    run_example()
