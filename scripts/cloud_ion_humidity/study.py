"""Native CUDA finite-volume CLOUD ion balance."""

import csv
from pathlib import Path

import numpy as np
import particula as par
import warp as wp
from particula.dynamics.ions.cleanup import IonNeutralCleanup
from particula.dynamics.ions.coagulation import (
    NEGATIVE_ION_MASS,
    POSITIVE_ION_MASS,
    AtmosphericCoagulation,
)
from particula.dynamics.ions.source import IonPairSource
from scipy.optimize import brentq

DATA = Path(__file__).resolve().parents[2] / "docs/data/cloud-ion-humidity"
ALPHA = {"default": 1.7, "0": 9.9, "7": 8.1, "40": 2.3, "70": 2.0}
BETA = 0.0083
MASSES = np.array([POSITIVE_ION_MASS, NEGATIVE_ION_MASS])
DENSITY = MASSES / (np.pi / 6 * (0.5e-9) ** 3)
RADIUS = (3 * 26.1 / (4 * np.pi)) ** (1 / 3)
T = 293.15
P = 101325.0


def mean_field(q, a, b=BETA):
    """Calculate the positive steady root in ions per cubic centimetre."""
    return 2 * q / (b + np.sqrt(b * b + 4 * a * 1e-6 * q))


def wall(eddy):
    """Construct the effective spherical first-order loss process."""
    return par.dynamics.SphericalWallLossStrategy(
        eddy, RADIUS, "particle_resolved"
    )


def coefficients(eddy):
    """Evaluate native loss frequencies for the two model ion masses."""
    return wall(eddy).loss_coefficient_for_particles(
        np.full(2, 0.25e-9), DENSITY, T, P
    )


def state(boxes, slots, pairs, vol):
    """Initialize unit-count ions in each physical sampling volume."""
    mass = np.zeros((boxes, slots, 2))
    mass[:, :pairs, 0] = MASSES[0]
    mass[:, pairs : 2 * pairs, 1] = MASSES[1]
    charge = (mass[:, :, 0] > 0).astype(float) - (mass[:, :, 1] > 0)
    return par.Aerosol(
        particles=par.particles.ParticleData(
            mass,
            abs(charge),
            charge,
            DENSITY,
            np.full(boxes, vol * 1e-6),
            distribution_type="particle_resolved",
        ),
        gas=par.gas.GasData(
            ["air"],
            np.array([0.0289644]),
            np.zeros((boxes, 1)),
            np.array([False]),
        ),
        environment=par.gas.EnvironmentData(
            np.full(boxes, T), np.full(boxes, P), np.zeros((boxes, 1))
        ),
    )


def run(label, q, vol, dt, boxes, seed, eddy):
    """Return concentration and inventory checks for one CUDA ensemble."""
    ref = mean_field(q, ALPHA[label])
    pairs = round(vol * ref)
    slots = max(48, int(2 * pairs + 12 * np.sqrt(max(1, 2 * pairs)) + 20))
    initial = state(boxes, slots, pairs, vol)
    seq = (
        IonPairSource(q * 1e6, 0, 1, *MASSES)
        | AtmosphericCoagulation(
            0,
            1,
            recombination=ALPHA[label] * 1e-12,
            attachment_model="mobility",
        )
        | IonNeutralCleanup(0, 1)
        | par.dynamics.WallLoss(wall(eddy))
    )
    maxima = 0
    means = []
    with seq.prepare(
        initial, device="cuda:0", time_step=dt, root_seed=seed
    ) as sim:
        sim.capture()
        burn, window, stride = 400, 600, 20
        for t in range(stride, burn + window + 1, stride):
            sim.run(round(stride / dt))
            d = sim.diagnostics()
            maxima = max(maxima, int(d.active_slots.max()))
            # Native wall retirement may preserve raw count in empty slots.
            free = (
                np.asarray(d.particle_mass_concentration)
                * (vol * 1e-6)
                / MASSES
            )
            np.testing.assert_allclose(free, np.rint(free), atol=1e-7, rtol=0)
            np.testing.assert_allclose(
                free.sum(axis=1), d.active_slots, atol=1e-7, rtol=0
            )
            np.testing.assert_allclose(
                free[:, 0] - free[:, 1],
                d.charge_concentration * vol * 1e-6,
                atol=1e-7,
                rtol=0,
            )
            if not (np.isfinite(free).all() and (free >= 0).all()):
                raise ValueError("Ion counts must be finite and nonnegative")
            if t > burn:
                means.append(free.mean(axis=1) / vol)
        source = sim.ion_source_diagnostics()["injected_pairs"]
        exported = sim.ion_cleanup_diagnostics()["exported_mass"] / MASSES
        np.testing.assert_allclose(
            exported[:, 0], exported[:, 1], atol=1e-7, rtol=0
        )
        wall_inferred = pairs + source[:, None] - exported - free
        if not (wall_inferred.min() >= -1e-07):
            raise ValueError("Inferred wall losses must be nonnegative")
        np.testing.assert_allclose(
            wall_inferred, np.rint(wall_inferred), atol=1e-7, rtol=0
        )
    box_means = np.mean(means, axis=0)
    sem = box_means.std(ddof=1) / np.sqrt(boxes)
    return dict(
        label=label,
        q=q,
        volume_cm3=vol,
        dt_s=dt,
        boxes=boxes,
        seed=seed,
        initial_pairs=pairs,
        capacity=slots,
        peak_active=maxima,
        reference_cm3=ref,
        mean_cm3=box_means.mean(),
        sem_cm3=sem,
        first_sample_cm3=np.mean(means[0]),
        last_sample_cm3=np.mean(means[-1]),
        injected_pairs=int(source.sum()),
        recombined_pairs=round(exported[:, 0].sum()),
        inferred_wall_ions=round(wall_inferred.sum()),
    )


def recovery(times, alpha, beta, extra, offset):
    """Bracket detected signals across residual ions and first-order lag."""
    # A 0.5 s lag grid resolves the continuous 30 s extrema to <0.01 ion/cm³.
    tau = np.linspace(0.5, 60.0, 120)[:, None]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    rate = np.sqrt(beta**2 + 4 * alpha * (1.84 + extra))
    equilibrium = 2 * (1.84 + extra) / (beta + rate)

    def signal(t, residual):
        delta = residual - equilibrium
        return (
            offset
            + equilibrium
            + delta
            * np.exp(-rate * t)
            / (1 - alpha * delta / rate * np.expm1(-rate * t))
        )

    def bound(t, residual, reduce):
        # Convolve the physical recovery with the detector response. Include
        # the instantaneous detector explicitly at the zero-lag boundary.
        initial = signal(0, residual)
        end_hold = initial + (1407.0 - initial) * np.exp(-120 / tau)
        limit = np.minimum(t / tau, 50)
        u = limit * (nodes + 1) / 2
        integral = (
            limit[:, 0]
            / 2
            * ((np.exp(-u) * signal(t - tau * u, residual)) @ weights)
        )
        readings = end_hold[:, 0] * np.exp(-t / tau[:, 0]) + integral
        return float(reduce(np.append(readings, signal(t, residual))))

    return {
        "low": [bound(t, 0, np.min) for t in times],
        "high": [bound(t, 100, np.max) for t in times],
    }


def simulate():
    """Reproduce the matched chamber comparison and conditional design."""
    if not wp.is_cuda_available():
        raise RuntimeError("The particle-resolved ensembles require CUDA.")
    eddy = brentq(lambda k: coefficients(k).mean() - BETA, 1e-5, 1e5)
    np.testing.assert_allclose(coefficients(eddy), BETA, rtol=1e-4, atol=0)
    cases = [
        ("7", 68, 0.02, 0.25, 96, 71831),
        ("70", 68, 0.02, 0.25, 96, 71832),
        ("70", 2.64, 0.04, 0.5, 128, 71833),
    ]
    runs = [run(*case, eddy) for case in cases]
    with (DATA / "figure5d.csv").open() as stream:
        points = [
            {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)
        ]
    for case in runs:
        case["observation"] = min(
            (p for p in points if p["rh"] == float(case["label"])),
            key=lambda p: abs(p["q_cm3_s"] - case["q"]),
        )

    # Substitution of y=n+d into the ion balance gives identical ordinary
    # shutter traces for this extra source and weaker first-order loss.
    alpha = 1.938024269774123e-6
    beta = 0.0067
    offset = 1064.9460072225502
    times = list(range(121))
    offset_band = recovery(times, alpha, beta, 0, offset)
    source_band = recovery(
        times,
        alpha,
        beta - 2 * alpha * offset,
        beta * offset - alpha * offset**2,
        0,
    )
    with (DATA / "two-inlet-design.csv").open() as stream:
        design = [
            {k: float(row[k]) for k in ("noise", "gain", "contrast", "lower95")}
            for row in csv.DictReader(stream)
        ]
    return {
        "device": wp.get_device("cuda:0").name,
        "cases": runs,
        "observed_difference": runs[1]["observation"]["ions_cm3"]
        - runs[0]["observation"]["ions_cm3"],
        "simulated_difference": runs[1]["mean_cm3"] - runs[0]["mean_cm3"],
        "difference_95": 1.96
        * np.hypot(runs[0]["sem_cm3"], runs[1]["sem_cm3"]),
        "ceiling": runs[2]["q"] / beta,
        "times": times,
        "offset": offset_band,
        "source": source_band,
        "design": design,
    }
