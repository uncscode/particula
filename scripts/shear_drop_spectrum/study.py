"""Stochastic shear coalescence and the independent discrete mass balance."""

from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from particula import Aerosol
from particula.dynamics import Coagulation, TurbulentShearCoagulationStrategy
from particula.gas import EnvironmentData, GasData, get_dynamic_viscosity
from particula.particles import ParticleData
from scipy.integrate import solve_ivp
from scipy.stats import binomtest

MARKERS = (
    Path(__file__).resolve().parents[2]
    / "docs/data/shear-drop-spectrum/figure2.csv"
)
MASS = 1.5e-3 / 6e8  # kg per elementary drop
RADIUS = (3 * MASS / (4 * np.pi * 1000)) ** (1 / 3)
DENSITY = 6e8  # initial number per cubic meter
EPSILON = 0.01  # m2/s3
NU = 1.7e-5  # m2/s
TEMPERATURE = 283.15
AIR = float(get_dynamic_viscosity(TEMPERATURE)) / NU
CLOCK = np.sqrt(8 * np.pi / 15 * EPSILON / NU) * RADIUS**3 * DENSITY
TIMES = np.array([0.03, 0.06, 0.09])


def constant_abundance(z, mass_class):
    """Return the monodisperse constant-kernel solution (equation 3)."""
    return z ** (mass_class - 1) / (1 + z) ** (mass_class + 1)


def reference(n, times=(*TIMES, 0.1044)):
    """Integrate normalized class abundances against dimensionless time K."""
    mass = np.arange(1, n + 1)
    kernel = (np.cbrt(mass[:, None]) + np.cbrt(mass[None, :])) ** 3
    destination = (mass[:, None] + mass[None, :]).ravel()

    def rate(_, state):
        pairs = kernel * (state[:, None] * state[None, :])
        births = np.bincount(
            destination, weights=pairs.ravel(), minlength=2 * n + 1
        )
        return 0.5 * births[1 : n + 1] - pairs.sum(axis=1)

    initial = np.zeros(n)
    initial[0] = 1
    solution = solve_ivp(
        rate,
        (0, max(times)),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=2e-11,
        atol=1e-15,
        max_step=0.0005,
    )
    if not (
        solution.success
        and np.isfinite(solution.y).all()
        and solution.y.min() >= 0
    ):
        raise ValueError("Invalid population-balance solution")
    rows = []
    for k, state in zip(solution.t, solution.y.T, strict=True):
        total = state.sum()
        for j in range(1, 9):
            rows.append(
                dict(
                    cutoff=n,
                    K=k,
                    mass_class=j,
                    shear=state[j - 1],
                    constant=constant_abundance(4 * k, j),
                    matched=constant_abundance(1 / total - 1, j),
                    shear_ge=state[j - 1 :].sum(),
                    total=total,
                    mass_sum=mass @ state,
                    time_s=k / CLOCK,
                )
            )
    return rows


# %% Native particle-resolved simulation. Only snapshots cross back to the CPU.
def run_case(name, slots, boxes, seed, steps, physical=False):
    """Run one independent multi-box ensemble with native CUDA shear."""
    shape = (boxes, slots)
    aerosol = Aerosol(
        particles=ParticleData(
            masses=np.full((*shape, 1), MASS),
            concentration=np.ones(shape),
            charge=np.zeros(shape),
            density=np.array([1000.0]),
            volume=np.full(boxes, slots / DENSITY),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["air"], np.array([0.029]), np.zeros((boxes, 1)), np.array([False])
        ),
        environment=EnvironmentData(
            np.full(boxes, TEMPERATURE),
            np.full(boxes, 101325.0),
            np.zeros((boxes, 1)),
        ),
    )
    process = Coagulation(
        TurbulentShearCoagulationStrategy("particle_resolved", EPSILON, AIR)
    )
    dt = 0.03 / (steps * CLOCK)
    observations = [(k, steps) for k in TIMES]
    if physical:
        dt = 60 / steps
        observations = [(960 * CLOCK, 16 * steps), (5400 * CLOCK, 74 * steps)]
    rows = []
    drift = 0.0
    with process.prepare(
        aerosol, device="cuda:0", root_seed=seed, time_step=dt
    ) as sim:
        sim.capture()
        previous = np.full(boxes, slots)
        for k, interval_steps in observations:
            sim.run(steps=interval_steps)
            state = sim.snapshot().particles
            counts = state.concentration
            if not (
                np.isfinite(state.masses).all() and np.all(state.masses >= 0)
            ):
                raise ValueError("Negative or nonfinite droplet mass")
            if not np.all((counts == 1) | (counts == 0)):
                raise ValueError("Nonunit active droplet count")
            np.testing.assert_array_equal(state.charge, 0)
            units = counts * state.masses[:, :, 0] / MASS
            np.testing.assert_allclose(
                units, np.rint(units), atol=1e-10, rtol=0
            )
            drift = max(
                drift, float(np.max(abs(units.sum(axis=1) / slots - 1)))
            )
            if drift >= 1e-12:
                raise ValueError("Box water mass changed during coalescence")
            current = counts.sum(axis=1)
            if np.any(current > previous):
                raise ValueError("Droplet number increased during coalescence")
            previous = current
            for box, values in enumerate(np.rint(units).astype(int)):
                bins = np.bincount(values, minlength=9)
                rows.append(
                    dict(
                        K=k,
                        box=box,
                        total=int(current[box]),
                        ge8=int(sum(bins[8:])),
                        **{f"n{j}": int(bins[j]) for j in range(1, 9)},
                    )
                )
    frame = pd.DataFrame(rows)
    frame["case"] = name
    frame["slots"] = slots
    return frame, dict(
        name=name,
        slots=slots,
        boxes=boxes,
        seed=seed,
        dt_s=dt,
        steps=[n for _, n in observations],
        max_mass_drift=drift,
    )


def summarize(raw, fine, rng):
    """Resample whole boxes jointly for total number and rare-class counts."""
    rows = []
    for (name, k), group in pd.concat([raw, fine]).groupby(["case", "K"]):
        n = len(group)
        slots = int(group.slots.iloc[0])
        values = group[["total", "n8"]].to_numpy() / slots
        avg_tot, avg8 = values.mean(axis=0)
        counts = int((group.n8 > 0).sum())
        # Joint box bootstrap propagates the fitted clock into the tail ratio.
        bs = values[rng.integers(n, size=(4000, n))].mean(axis=1)

        def matched(total):
            return total**2 * (1 - total) ** 7

        ci8 = np.quantile(bs[:, 1], [0.025, 0.975])
        cir = np.quantile(bs[:, 1] / matched(bs[:, 0]), [0.025, 0.975])
        interval = binomtest(counts, n).proportion_ci(method="wilson")
        rows.append(
            dict(
                case=name,
                K=k,
                slots=slots,
                boxes=n,
                total=avg_tot,
                n8=avg8,
                n8_count=group.n8.sum(),
                n8_low=ci8[0],
                n8_high=ci8[1],
                matched_constant=matched(avg_tot),
                ratio=avg8 / matched(avg_tot),
                ratio_low=cir[0],
                ratio_high=cir[1],
                ge8=group.ge8.mean() / slots,
                with8=counts,
                detection=counts / n,
                p_low=interval.low,
                p_high=interval.high,
            )
        )
    return rows


def threshold_counts(physical, physical_ref, rng):
    """Count all drops reaching the specified minimum mass, per litre."""
    physical_rows = []
    for (k, group), threshold, minutes in zip(
        physical.groupby("K"), [4, 8], [16, 90], strict=True
    ):
        reported = physical_ref[
            (physical_ref.mass_class == threshold)
            & np.isclose(physical_ref.K, k)
        ].iloc[0]
        count = (
            group.ge8
            if threshold == 8
            else group.total - group.n1 - group.n2 - group.n3
        )
        n = len(count)
        per_litre = DENSITY / 1000
        samples = count.to_numpy() * per_litre / group.slots.to_numpy()
        means = samples[rng.integers(n, size=(4000, n))].mean(axis=1)
        constant_ge = (4 * k / (1 + 4 * k)) ** (threshold - 1) / (1 + 4 * k)
        physical_rows.append(
            dict(
                minutes=minutes,
                mass_at_least=threshold,
                boxes=n,
                clusters=count.sum(),
                boxes_with_cluster=(count > 0).sum(),
                cuda_per_L=samples.mean(),
                low_per_L=np.quantile(means, 0.025),
                high_per_L=np.quantile(means, 0.975),
                shear_theory_per_L=per_litre * reported.shear_ge,
                constant_per_L=per_litre * constant_ge,
                paper_approx_per_L=100,
            )
        )
    return physical_rows


def simulate():
    """Run the published case matrix and return compact numerical evidence."""
    cases = [
        ("pilot", 256, 32, 17001, 30),
        ("small", 256, 1024, 17002, 60),
        ("large_coarse", 1024, 384, 17003, 30),
        ("large_fine_a", 1024, 384, 17004, 60),
        ("large_fine_b", 1024, 384, 17005, 60),
        ("large_finer", 1024, 384, 17006, 120),
    ]
    tables, case_info = [], []
    for case in cases:
        table, info = run_case(*case)
        tables.append(table)
        case_info.append(info)
    physical, info = run_case("physical", 1024, 768, 17007, 3, physical=True)
    case_info.append(info)
    raw = pd.concat(tables, ignore_index=True)
    fine = raw[raw.case.isin(["large_fine_a", "large_fine_b"])].copy()
    fine["case"] = "large_fine_pooled"
    rng = np.random.default_rng(20261121)
    summary = summarize(raw, fine, rng)
    reference_rows = pd.DataFrame(
        [row for n in (32, 64, 128) for row in reference(n)]
    )
    narrow = reference_rows.query("cutoff == 64")
    wide = reference_rows.query("cutoff == 128")
    np.testing.assert_allclose(narrow.shear, wide.shear, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(wide.mass_sum, 1, atol=1e-12, rtol=0)
    physical_ref = pd.DataFrame(reference(128, [960 * CLOCK, 5400 * CLOCK]))
    physical_rows = threshold_counts(physical, physical_ref, rng)
    markers = pd.read_csv(MARKERS)
    markers["abundance"] = 10 ** ((290 - markers.y_px) * 5 / 563)
    last = fine.query("K == 0.09")
    spectrum = (
        last[[f"n{i}" for i in range(1, 9)]] / last.slots.to_numpy()[:, None]
    )
    return dict(
        device=wp.get_device("cuda:0").name,
        cases=case_info,
        reference=reference_rows.to_dict("records"),
        summary=pd.DataFrame(summary).to_dict("records"),
        physical=pd.DataFrame(physical_rows).to_dict("records"),
        markers=markers.to_dict("records"),
        mean=spectrum.mean().tolist(),
        halfwidth=(1.96 * spectrum.std() / np.sqrt(len(last))).tolist(),
    )
