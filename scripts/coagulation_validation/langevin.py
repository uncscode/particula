"""Independent CUDA first-passage experiment for a finite spherical bath.

Lengths are in contact radii, time in R/sqrt(kT/mu), speed in sqrt(kT/mu).
The bath emits Maxwell flux through its outer sphere. Particles are followed
until they hit the contact sphere or leave the bath. A continuum resistance
outside the bath joins the boundary flux to the density at infinity.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.integrate import quad
from scripts.coagulation_validation.profiles import potential_profile
from scripts.coagulation_validation.reference import reference_capture


@wp.func
def _force(
    position: wp.vec3d,
    slope: wp.array(dtype=wp.float64),
    geometry: wp.float64,
    attraction: wp.float64,
):
    one = wp.float64(1.0)
    radius = wp.length(position)
    x = wp.min(one - wp.float64(1e-8), one / radius)
    address = x * wp.float64(slope.shape[0] - 1)
    index = wp.min(wp.int32(address), slope.shape[0] - 2)
    weight = address - wp.float64(index)
    derivative = slope[index] * (one - weight) + slope[index + 1] * weight
    denom = (one - x) * (one + x)
    u = geometry * x * x / denom
    # Analytic derivative of Hamaker's expression through u. Its sign
    # gives an inward acceleration in the radial equation of motion.
    derivative -= (
        attraction
        / wp.float64(6.0)
        * (u / (one + u))
        * (u / (one + u))
        * geometry
        * x
        / (denom * denom)
    )
    return position * (x * x / radius * derivative)


@wp.kernel
def _trajectories(
    seed: int,
    count: int,
    outer: wp.float64,
    tau: wp.float64,
    step_scale: wp.float64,
    geometry: wp.float64,
    attraction: wp.float64,
    slope: wp.array(dtype=wp.float64),
    total: wp.array(dtype=wp.int32),
):
    idx = wp.tid()
    rng = wp.rand_init(seed, idx)
    position = wp.vec3d(outer, wp.float64(0), wp.float64(0))
    normal_speed = wp.sqrt(
        -wp.float64(2)
        * wp.log(wp.max(wp.float64(wp.randf(rng)), wp.float64(1e-15)))
    )
    velocity = wp.vec3d(
        -normal_speed, wp.float64(wp.randn(rng)), wp.float64(wp.randn(rng))
    )
    radius = outer
    for _ in range(count):
        force = _force(position, slope, geometry, attraction)
        dt = step_scale * wp.min(
            tau, radius / wp.max(wp.length(velocity), wp.float64(1))
        )
        # Resolve steep attractive trajectories just outside contact.
        dt = wp.min(
            dt,
            wp.sqrt(
                wp.float64(0.1)
                * wp.max(radius - wp.float64(1), wp.float64(1e-5))
                / wp.max(wp.length(force), wp.float64(1e-12))
            ),
        )
        half = dt * wp.float64(0.5)
        velocity += half * force
        position += half * velocity
        if wp.length(position) <= wp.float64(1):
            wp.atomic_add(total, 0, 1)
            return
        decay = wp.exp(-dt / tau)
        noise = wp.sqrt(wp.float64(1) - decay * decay)
        velocity = decay * velocity + noise * wp.vec3d(
            wp.float64(wp.randn(rng)),
            wp.float64(wp.randn(rng)),
            wp.float64(wp.randn(rng)),
        )
        position += half * velocity
        radius = wp.length(position)
        if radius <= wp.float64(1):
            wp.atomic_add(total, 0, 1)
            return
        if radius >= outer:
            return
        velocity += half * _force(position, slope, geometry, attraction)
    wp.atomic_add(total, 1, 1)


def measure(
    fraction,
    charges,
    attraction,
    coupling,
    epsilon,
    tau,
    outer,
    scale,
    shots,
    seed,
):
    """Measure the rate and sampling uncertainty at fixed outer radius."""
    interp, potential = potential_profile(
        fraction, charges, attraction, coupling, epsilon
    )
    ec, ef = reference_capture(potential, resolution=72)
    from particula.dynamics.coagulation.charged_dimensionless_kernel import (
        get_hard_sphere_kernel,
    )

    predicted = float(
        get_hard_sphere_kernel(tau * ec / ef) * (ef * ef / ec) / tau
    )
    derivative = interp.derivative()(np.linspace(0, 1, 4097))
    slopes = wp.array(derivative, dtype=wp.float64, device="cuda")
    totals = wp.zeros(2, dtype=wp.int32, device="cuda")
    started = time.monotonic()
    wp.launch(
        _trajectories,
        shots,
        inputs=[
            seed,
            1000000,
            outer,
            tau,
            scale,
            4 * fraction * (1 - fraction),
            attraction,
            slopes,
            totals,
        ],
        device="cuda",
    )
    hits, censored = totals.numpy()
    if censored:
        raise RuntimeError(f"{censored} trajectories reached the time cutoff")
    # Inner flux per unit boundary number density. The chemical-potential
    # matching condition contributes exp(U_outer) to the inverse inner rate.
    factor = np.pi * outer**2 * np.sqrt(8 / np.pi)
    inner = factor * hits / shots
    exterior = quad(lambda x: np.exp(potential(x)), 0, 1 / outer)[0] / (
        4 * np.pi * tau
    )
    weight = np.exp(potential(1 / outer))
    rate = 1 / (weight / inner + exterior)
    se = (
        rate**2
        * weight
        / inner**2
        * factor
        * np.sqrt(hits / shots * (1 - hits / shots) / shots)
    )
    return dict(
        fraction=fraction,
        charges=charges,
        attraction=attraction,
        coupling=coupling,
        epsilon=epsilon,
        diffusive_knudsen=tau,
        outer_radius=outer,
        step_scale=scale,
        trials=shots,
        hits=int(hits),
        censored=int(censored),
        seed=seed,
        rate=float(rate),
        ci95_half=float(1.96 * se),
        predicted=predicted,
        relative_residual=float(predicted / rate - 1),
        seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--shots", type=int, default=524288)
    args = parser.parse_args()
    records = []
    for case, (f, z, a, c, q) in enumerate(
        [
            (0.5, (0, 0), 0.0, 0.0, 1.0),
            (0.25, (1, 0), 12.0, 1.5, 1.0),
            (0.5, (1, 1), 12.0, 2.5, 1.0),
            (0.2, (1, -1), 12.0, 1.5, 1.0),
            (0.5, (1, -1), 12.0, 1.5, 0.5),
            (0.25, (1, 1), 12.0, 2.5, 2.0),
            (0.5, (1, 0), 12.0, 1.5, 1.0),
            (0.2, (1, -1), 12.0, 0.5, 1.0),
            (0.5, (1, -1), 12.0, 0.5, 1.0),
            (0.2, (1, -1), 12.0, 0.5, 0.5),
            (0.2, (1, -1), 12.0, 0.5, 2.0),
            (0.5, (0, 0), 12.0, 0.0, 1.0),
            (0.2, (0, 0), 12.0, 0.0, 1.0),
        ]
    ):
        for outer, dt in [(5.0, 0.12), (8.0, 0.12), (8.0, 0.06)]:
            result = measure(
                f,
                z,
                a,
                c,
                5.9,
                q,
                outer,
                dt,
                args.shots,
                1759 + case * 3 + len(records),
            )
            records.append(result)
            print(
                case,
                outer,
                dt,
                result["rate"],
                result["relative_residual"],
                flush=True,
            )
            args.output.write_text(
                json.dumps(
                    dict(device=wp.get_device("cuda").name, results=records),
                    indent=2,
                )
                + "\n"
            )
    # The strong unlike controls measure where the closure breaks down.
    # Use each finest-step trajectory estimate to check the admitted cases.
    for row in records:
        if row["step_scale"] != 0.06:
            continue
        bare = row["coupling"] * np.prod(row["charges"])
        if bare >= -0.5 and (
            abs(row["relative_residual"]) > 0.1
            or 2 * row["ci95_half"] / row["rate"] > 0.1
        ):
            raise SystemExit(f"trajectory comparison exceeds tolerance: {row}")
