"""Bounded AD refits of particle chemistry and stochastic chamber physics."""

import json
import time

import numpy as np
from gradient import conditional_chemistry, gaussian_ad
from scoring import Targets
from simulate import ARCHIVE, HERE, Chamber

RESULTS = HERE / "results"


def write_json(name, record):
    """Save reproducible parameter and optimization records."""
    (RESULTS / name).write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def parameter_bounds():
    """Physical limits used for fitting and perturbation replays."""
    caps = np.array([x["upper"] for x in ARCHIVE["reactions"]])
    lows = np.r_[
        [1 / 12] * 3,
        [0] * 6,
        2.818,
        0,
        0,
        0,
        0,
        0,
        0,
        np.zeros(70),
        np.full(15, 0.85),
        1.20,
        0,
        0,
        0,
        0,
    ]
    highs = np.r_[
        [4] * 3,
        [2] * 6,
        16.5515,
        1,
        300,
        100,
        1,
        0.1,
        5,
        caps,
        np.full(15, 1.15),
        2.05,
        1,
        0.35,
        0.30,
        0.30,
    ]
    return lows, highs


def parameter_steps(center, caps):
    """Absolute perturbation scales in the fitted parameter coordinates."""
    rate_scales = np.minimum(caps * 0.1, np.maximum(center[16:86] * 0.3, 0.08))
    return np.r_[
        np.maximum(0.08, center[:3] * 0.25),
        np.full(6, 0.15),
        1.8,
        0,
        0,
        12,
        0,
        0,
        0,
        rate_scales,
        np.full(15, 0.05),
        0.14,
        0.23,
        0.10,
        0.065,
        0.065,
    ]


def run():
    """Fit kinetics and score stochastic particle replays."""
    RESULTS.mkdir(exist_ok=True)
    c = Chamber()
    data = Targets(c)

    def measure(points, seed=17, n=256, dt=30, snapshot=False, details=False):
        result = c.run(
            points[:, :-2], c.ids, n=n, dt=dt, rng=seed, snapshot=snapshot
        )
        score, d = data.evaluate(result, points, c.ids, details)
        return score, result, d

    base = np.r_[c.baseline(), 0.046, 0.0]
    joint = np.r_[c.baseline("joint"), 0.046, 0.0]
    lows, highs = parameter_bounds()
    caps = highs[16:86]
    frozen = np.array([10, 11, 13, 14, 15])
    seed = 19
    start = time.monotonic()
    initial, result, _ = measure(
        np.array([base]), seed=seed, n=1024, snapshot=True
    )
    shadow, info = conditional_chemistry(
        c, result[0][0], c.gas_history, base, caps
    )
    first = np.stack([base, joint, shadow])
    initial_small, *_ = measure(first)
    center = first[initial_small.loss.argmin()].copy()
    scales = parameter_steps(center, caps)
    dims = np.flatnonzero(scales > 0)
    rng = np.random.default_rng(321)
    # Exact baselines stay candidates. Interior center allows two-sided
    # proposals.
    saved = center.copy()
    center[-4:] = [0.12, 0.06, 0.05, 0.045]

    def bounded(raw):
        v = np.clip(raw, lows, highs)
        v[..., frozen] = base[frozen]
        return v

    for iteration in range(8):
        # Antithetic perturbations and common particle random numbers within
        # round.
        z = rng.normal(size=(9, len(dims)))
        z = np.vstack((z, -z))
        sigma = 0.75 * 0.9**iteration
        trial = np.tile(center, (len(z), 1))
        trial[:, dims] += z * sigma * scales[dims]
        trial = bounded(trial)
        checkpoints = np.vstack((saved, center, trial))
        scored, *_ = measure(checkpoints, seed=40 + iteration, n=192)
        grad = gaussian_ad(z * sigma, scored.loss.to_numpy()[2:], sigma)
        direction = grad / max(1e-12, np.sqrt(np.mean(grad**2)))
        proposals = np.tile(center, (3, 1))
        for v, step in zip(proposals, [0.1, 0.35, 0.7], strict=True):
            v[dims] -= step * scales[dims] * np.clip(direction, -3, 3)
        proposals = bounded(proposals)
        pool = np.vstack(
            (saved, center, checkpoints[int(scored.loss.argmin())], proposals)
        )
        check, *_ = measure(pool, seed=212, n=384)
        winner = int(check.loss.argmin())
        saved = pool[winner].copy()
        center = saved.copy()
        print(
            "round",
            iteration + 1,
            "score",
            check.loss.min(),
            "picked",
            winner,
            "elapsed",
            round(time.monotonic() - start),
            flush=True,
        )
        write_json(
            "fit.json",
            dict(
                ids=c.ids,
                initial=base.tolist(),
                previous_joint=joint.tolist(),
                local_AD=shadow.tolist(),
                selected=saved.tolist(),
                conditional_AD=info,
            ),
        )
    return c, data, base, saved


if __name__ == "__main__":
    run()
