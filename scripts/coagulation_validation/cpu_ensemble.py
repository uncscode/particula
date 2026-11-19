"""Check resolved CPU encounter statistics and timestep convergence."""

import argparse
import json
from pathlib import Path

import numpy as np
from docs.examples.gpu.prepared.van_der_waals import initial_state, inventory
from particula.dynamics import VanDerWaalsCoagulationStrategy


def run(trials=1024):
    """Return independent two-particle box counts at a fixed physical time."""
    results = []
    strategy = VanDerWaalsCoagulationStrategy("particle_resolved", 8.9e-20, 5.9)
    state = initial_state(1)
    start = state.particles
    _, mass, charge = inventory(state)
    rate = strategy.kernel(start, 298.0, 101325.0)[0, 1]
    for steps in (8, 32):
        dt = start.volume[0] / (rate * steps)
        hits = 0
        for seed in range(trials):
            strategy.random_generator = np.random.default_rng(
                seed + steps * 10000
            )
            state = initial_state(1)
            for _ in range(steps):
                strategy.step(state.particles, 298.0, 101325.0, dt)
                if np.count_nonzero(state.particles.concentration) == 1:
                    break
            merged, m, q = inventory(state)
            np.testing.assert_allclose(m, mass, rtol=1e-13, atol=0)
            np.testing.assert_array_equal(q, charge)
            hits += int(merged[0])
        p = 1 - (1 - 1 / steps) ** steps
        if abs(hits / trials - p) > 4 * np.sqrt(p * (1 - p) / trials):
            raise AssertionError(
                "CPU ensemble outside binomial sampling tolerance"
            )
        results.append(
            dict(
                steps=steps,
                trials=trials,
                hits=hits,
                probability=hits / trials,
                continuous_probability=float(-np.expm1(-1.0)),
                finite_step_probability=p,
                dt_s=float(dt),
                seed_start=steps * 10000,
            )
        )
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + "\n")
