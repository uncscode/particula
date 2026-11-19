"""Differentiable fixed-step growth in closed single-material boxes.

This concrete research path assumes zero equilibrium vapor, unit accommodation,
constant diffusivity and positive particle masses. Each occupied slot represents
a monodisperse component. Independent boxes may share initial guesses.
"""

from contextlib import nullcontext

import numpy as np
import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
)
from particula.gpu.dynamics.condensation_funcs import (
    first_order_mass_transport_k_wp,
)
from particula.gpu.properties.particle_properties import (
    vapor_transition_correction_wp,
)


@wp.kernel
def _mass_from_radius(
    radius: FloatMatrix,
    factor: wp.float64,
    mass: FloatMatrix,
):
    b, j = wp.tid()  # type: ignore[misc]
    r = radius[b, j] / wp.float64(1e6)
    mass[b, j] = factor * r * r * r


@wp.kernel
def _reservoir(
    initial: FloatMatrix,
    current: FloatMatrix,
    counts: FloatMatrix,
    initial_gas: FloatVector,
    gas: FloatVector,
):
    b = wp.tid()
    used = wp.float64(0.0)
    for j in range(current.shape[1]):
        used += counts[b, j] * (current[b, j] - initial[b, j])
    gas[b] = initial_gas[b] - used


@wp.kernel
def _advance(
    current: FloatMatrix,
    gas: FloatVector,
    factor: wp.float64,
    mean_path: wp.float64,
    diffusion: wp.float64,
    dt: wp.float64,
    result: FloatMatrix,
):
    b, j = wp.tid()  # type: ignore[misc]
    radius = wp.pow(current[b, j] / factor, (wp.float64(1.0) / wp.float64(3.0)))
    transition = vapor_transition_correction_wp(
        mean_path / radius, wp.float64(1.0)
    )
    rate = (
        first_order_mass_transport_k_wp(radius, transition, diffusion) * gas[b]
    )
    result[b, j] = current[b, j] + dt * rate


@wp.kernel
def _observe(
    mass: FloatMatrix,
    factor: wp.float64,
    result: FloatMatrix,
):
    b, j = wp.tid()  # type: ignore[misc]
    result[b, j] = wp.float64(1e6) * wp.pow(
        mass[b, j] / factor, (wp.float64(1.0) / wp.float64(3.0))
    )


def _growth_inputs(
    initial_um,
    number_density,
    vapor,
    density,
    mean_path,
    diffusivity,
    duration,
    steps,
    segment_steps,
    adjoint,
):
    """Validate the interior growth domain before allocating device state."""
    radii = np.asarray(initial_um, dtype=float)
    numbers = np.asarray(number_density, dtype=float)
    vapor = np.asarray(vapor, dtype=float)
    if (
        radii.ndim != 2
        or not radii.size
        or numbers.shape != radii.shape
        or vapor.shape != (len(radii),)
    ):
        raise ValueError(
            "expected box-by-component radii and weights and per-box vapor"
        )
    if not all(
        np.all(np.isfinite(a)) and np.all(a > 0)
        for a in (
            radii,
            numbers,
            vapor,
            density,
            mean_path,
            diffusivity,
            duration,
        )
    ):
        raise ValueError("growth inputs must be positive and finite")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if segment_steps is not None and (
        type(segment_steps) is not int or not 1 <= segment_steps <= steps
    ):
        raise ValueError(
            "segment_steps must be an integer between one and steps"
        )
    if adjoint is not None and (
        np.shape(adjoint) != radii.shape or not np.all(np.isfinite(adjoint))
    ):
        raise ValueError("invalid terminal adjoint")
    # Each component cannot gain more mass than the entire initial vapor.
    # This radius bound gives an upper bound on removal in one Euler update.
    factor = 4 * np.pi * density / 3
    max_radius = np.cbrt(
        (radii * 1e-6) ** 3 + vapor[:, None] / (factor * numbers)
    )
    if np.any(
        duration
        / steps
        * 4
        * np.pi
        * diffusivity
        * np.sum(numbers * max_radius, axis=1)
        >= 1
    ):
        raise ValueError("refine the timestep to guarantee interior vapor")

    return radii, numbers, vapor, factor


def terminal_radius(
    initial_um,
    number_density,
    vapor,
    *,
    density,
    mean_path,
    diffusivity,
    duration,
    steps,
    device,
    adjoint=None,
    segment_steps=None,
):
    """Evaluate wet radii (µm) and optional vector-Jacobian product.

    initial_um and number_density (m⁻³) are positive box-by-component arrays.
    vapor (kg/m³) has one entry per box. Density is kg/m³, molecular mean
    free path is m, diffusivity is m²/s and duration is seconds. An optional
    adjoint gives one terminal observation weight per component.
    """
    radii, numbers, vapor, factor = _growth_inputs(
        initial_um,
        number_density,
        vapor,
        density,
        mean_path,
        diffusivity,
        duration,
        steps,
        segment_steps,
        adjoint,
    )
    grad = adjoint is not None
    options = dict(dtype=wp.float64, device=device)

    def empty(shape):
        return wp.empty(shape, requires_grad=grad, **options)

    def launch(kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=device)

    x = wp.array(radii, requires_grad=grad, **options)
    weights = wp.array(numbers, **options)
    g0 = wp.array(vapor, **options)
    m0 = empty(radii.shape)
    r = empty(radii.shape)

    def segment(start, length):
        mass = start
        for _ in range(length):
            gas = empty(len(radii))
            launch(_reservoir, len(radii), [m0, mass, weights, g0, gas])
            following = empty(radii.shape)
            launch(
                _advance,
                radii.shape,
                [
                    mass,
                    gas,
                    factor,
                    mean_path,
                    diffusivity,
                    duration / steps,
                    following,
                ],
            )
            mass = following
        return mass

    tape = wp.Tape() if grad else None
    if segment_steps is None or not grad:
        with tape if tape is not None else nullcontext():
            launch(_mass_from_radius, radii.shape, [x, factor, m0])
            final = segment(m0, steps)
            launch(_observe, radii.shape, [final, factor, r])
        if tape is not None:
            tape.backward(
                grads={r: wp.array(np.asarray(adjoint, dtype=float), **options)}
            )
    else:
        # The same initial inventory participates in every step. Its adjoint
        # accumulates across segments as well as through the first mass state.
        with wp.Tape() as initial_tape:
            launch(_mass_from_radius, radii.shape, [x, factor, m0])
        checkpoints = []
        mass = m0
        for start in range(0, steps, segment_steps):
            length = min(steps - start, segment_steps)
            checkpoints.append((mass, length))
            mass = segment(mass, length)
        with wp.Tape() as observation_tape:
            launch(_observe, radii.shape, [mass, factor, r])
        observation_tape.backward(
            grads={r: wp.array(np.asarray(adjoint, dtype=float), **options)}
        )
        seed = mass.grad
        for checkpoint, length in reversed(checkpoints):
            with wp.Tape() as reverse:
                replayed = segment(checkpoint, length)
            reverse.backward(grads={replayed: seed})
            seed = checkpoint.grad
        initial_tape.backward()
    return r.numpy(), x.grad.numpy() if grad else None
