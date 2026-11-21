"""Bulk optical losses for the smooth closed-condensation inverse model."""

from contextlib import nullcontext

import numpy as np
import warp as wp

from particula.execution._condensation_ode_kernels import FloatMatrix
from particula.execution.inverse_condensation import (
    _growth_inputs,
    _mass_from_radius,
    _mass_step,
)
from particula.optics.gpu.inverse import BulkOpticsAdjoint


@wp.kernel
def _optical_radii(
    mass: FloatMatrix,
    coefficient: wp.float64,
    waves: int,
    output: FloatMatrix,
):
    group, slot = wp.tid()  # type: ignore[misc]
    output[group, slot] = wp.pow(
        mass[group // waves, slot] / coefficient,
        wp.float64(1.0) / wp.float64(3.0),
    )


def _observation_inputs(
    observation_steps, wavelength, index, steps, shape, adjoint
):
    samples = tuple(observation_steps)
    if (
        not samples
        or any(type(s) is not int for s in samples)
        or samples[0] < 0
        or samples[-1] > steps
        or any(a >= b for a, b in zip(samples[:-1], samples[1:], strict=True))
    ):
        raise ValueError(
            "Observation steps must be ordered within the trajectory"
        )
    if np.iscomplexobj(wavelength):
        raise ValueError("Require real wavelengths")
    wave = np.atleast_1d(np.asarray(wavelength, dtype=float))
    if (
        wave.ndim != 1
        or not wave.size
        or not np.all(np.isfinite(wave) & (wave > 0))
    ):
        raise ValueError("Require positive wavelengths")
    indices = np.asarray(index, dtype=complex)
    if (
        np.any(~np.isfinite(indices))
        or np.any(indices.real <= 0)
        or np.any(indices.imag < 0)
    ):
        raise ValueError("Optical indices require n>0 and k>=0")
    indices = np.broadcast_to(indices, (shape[0], wave.size, shape[1]))
    target_shape = (len(samples), shape[0], wave.size, 3)
    if adjoint is not None:
        if np.iscomplexobj(adjoint):
            raise ValueError("Optical adjoint must be real")
        adjoint = np.asarray(adjoint, dtype=float)
        if adjoint.shape != target_shape or not np.all(np.isfinite(adjoint)):
            raise ValueError(
                "Optical adjoint must match the finite observation grid"
            )
    return samples, wave, indices, target_shape, adjoint


class _OpticalGrowth:
    def __init__(
        self, radii, numbers, vapor, wave, indices, transport, grad, device
    ):
        self.device = device
        self.grad = grad
        self.transport = transport
        self.waves = wave.size
        self.output_shape = (radii.shape[0], wave.size, 3)
        self.radius = self.upload(radii, grad=grad)
        self.initial = self.empty(radii.shape)
        self.water = self.upload(vapor)
        self.count = self.upload(numbers)
        self.grid = (len(radii) * wave.size, radii.shape[1])
        self.index = wp.array(
            np.ascontiguousarray(
                np.stack((indices.real, indices.imag), axis=-1).reshape(
                    (*self.grid, 2)
                )
            ),
            dtype=wp.vec2d,
            device=device,
        )
        self.density = self.upload(np.repeat(numbers, wave.size, axis=0))
        self.mie = BulkOpticsAdjoint(
            self.grid, np.tile(wave, len(radii)), device=device
        )

    def upload(self, data, *, grad=False):
        return wp.array(
            data, dtype=wp.float64, device=self.device, requires_grad=grad
        )

    def empty(self, shape):
        return wp.empty(
            shape,
            dtype=wp.float64,
            device=self.device,
            requires_grad=self.grad,
        )

    def launch(self, kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def initialize(self):
        self.launch(
            _mass_from_radius,
            self.initial.shape,
            [self.radius, self.transport[0], self.initial],
        )

    def observe(self, mass, tape=None):
        radii = self.empty(self.grid)
        self.launch(
            _optical_radii,
            self.grid,
            [mass, self.transport[0], self.waves, radii],
        )
        return self.mie.evaluate(radii, self.index, self.density, tape=tape)

    def segment(self, mass, start, end, visit):
        for step in range(start + 1, end + 1):
            mass = _mass_step(
                self.initial,
                mass,
                self.count,
                self.water,
                self.transport,
                self.empty,
                self.launch,
            )
            visit(step, mass)
        return mass


def condensation_optics(
    initial_um,
    number_density,
    vapor,
    *,
    wavelength,
    refractive_index,
    observation_steps,
    density,
    mean_path,
    diffusivity,
    duration,
    steps,
    device="cuda:0",
    adjoint=None,
    segment_steps=None,
):
    """Evaluate a closed-box bulk optical trajectory and optional initial VJP.

    Physics inputs and fixed-step restrictions match terminal_radius().
    Wavelengths [m] are shared across boxes. Absolute refractive indices
    broadcast to (box, wavelength, slot) and stay prescribed. Outputs
    are in m⁻¹ on (observation, box, wavelength, channel), with channels
    extinction, scattering and absorption in that order. Supply optical
    adjoint weights in the same layout for a gradient per initial radius
    measured in micrometres. The reverse pass replays checkpoint segments
    and uses bounded Mie scratch at each optical observation.
    """
    radii, number_density, vapor, coefficient = _growth_inputs(
        initial_um,
        number_density,
        vapor,
        density,
        mean_path,
        diffusivity,
        duration,
        steps,
        segment_steps,
        None,
    )
    samples, wave, indices, shape, adjoint = _observation_inputs(
        observation_steps,
        wavelength,
        refractive_index,
        steps,
        radii.shape,
        adjoint,
    )
    model = _OpticalGrowth(
        radii,
        number_density,
        vapor,
        wave,
        indices,
        (coefficient, mean_path, diffusivity, duration / steps),
        adjoint is not None,
        device,
    )
    selection = {step: i for i, step in enumerate(samples)}
    values = np.empty(shape)
    initial_tape = wp.Tape() if model.grad else None
    first = None
    with initial_tape if initial_tape is not None else nullcontext():
        model.initialize()
        if 0 in selection:
            first = model.observe(model.initial, initial_tape)
            values[0] = first.numpy().reshape(shape[1:])

    def sample(step, mass):
        if step in selection:
            values[selection[step]] = (
                model.observe(mass).numpy().reshape(shape[1:])
            )

    checkpoints = []
    current = model.initial
    stride = segment_steps or steps
    for start in range(0, samples[-1], stride):
        end = min(start + stride, samples[-1])
        if model.grad:
            checkpoints.append((start, end, current))
        current = model.segment(current, start, end, sample)
    if not model.grad:
        return values, None

    _reverse(model, adjoint, initial_tape, first, checkpoints, selection)
    return values, model.radius.grad.numpy()


def _reverse(model, adjoint, initial_tape, first, checkpoints, selection):
    cross_step = None
    for start, end, checkpoint in reversed(checkpoints):
        reverse = wp.Tape()
        observations = {}

        def accumulate(step, mass, reverse=reverse, observations=observations):
            if step in selection:
                output = model.observe(mass, reverse)
                observations[output] = model.upload(
                    adjoint[selection[step]].reshape(output.shape)
                )

        with reverse:
            replayed = model.segment(checkpoint, start, end, accumulate)
        if cross_step is not None:
            observations[replayed] = cross_step
        reverse.backward(grads=observations)
        cross_step = checkpoint.grad
    initial_seed = (
        None
        if first is None
        else {first: model.upload(adjoint[0].reshape(first.shape))}
    )
    initial_tape.backward(grads=initial_seed)
