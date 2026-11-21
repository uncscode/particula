# Warp DSL vector operations and annotations are compiled by Warp.
# mypy: disable-error-code="index,operator,valid-type"
"""Bounded-memory CUDA bulk optical predictions and Warp tape adjoints."""

from math import pi

import numpy as np
import warp as wp

from particula.optics.gpu.derivatives import mie_cross_section_sums_jacobian
from particula.optics.gpu.mie import (
    mie_coefficients_kernel,
    mie_efficiencies_kernel,
)
from particula.optics.gpu.workspace import _array, _capacities

_PI = wp.constant(wp.float64(pi))


@wp.kernel
def _gather(
    radius: wp.array2d(dtype=wp.float64),
    index: wp.array2d(dtype=wp.vec2d),
    weight: wp.array2d(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    medium: wp.float64,
    start: int,
    n_cases: int,
    gathered_index: wp.array(dtype=wp.vec2d),
    size: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    flat = start + i
    zero = wp.float64(0.0)
    gathered_index[i] = wp.vec2d(wp.float64(1.0), zero)
    size[i] = zero
    if flat < n_cases:
        group = flat // radius.shape[1]
        slot = flat % radius.shape[1]
        r = radius[group, slot]
        w = weight[group, slot]
        size[i] = r * scale[group]
        gathered_index[i] = index[group, slot] / medium
        if not wp.isfinite(w) or w < zero:
            size[i] = wp.float64(-1.0)


@wp.kernel
def _reduce(
    radius: wp.array2d(dtype=wp.float64),
    weight: wp.array2d(dtype=wp.float64),
    q: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    start: int,
    n_cases: int,
    bulk: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    flat = start + i
    if flat >= n_cases:
        return
    if status[i] != 0:
        wp.atomic_max(failure, 0, status[i])
        return
    group = flat // radius.shape[1]
    slot = flat % radius.shape[1]
    r = radius[group, slot]
    amount = _PI * r * r * weight[group, slot]
    for channel in range(3):
        value = amount * q[i, channel]
        if not wp.isfinite(value):
            wp.atomic_max(failure, 0, 3)
        else:
            wp.atomic_add(bulk, group, channel, value)


@wp.kernel
def _propagate(
    radius: wp.array2d(dtype=wp.float64),
    weight: wp.array2d(dtype=wp.float64),
    q: wp.array2d(dtype=wp.float64),
    ext: wp.array(dtype=wp.vec3d),
    sca: wp.array(dtype=wp.vec3d),
    status: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float64),
    medium: wp.float64,
    start: int,
    n_cases: int,
    adj_bulk: wp.array2d(dtype=wp.float64),
    adj_radius: wp.array2d(dtype=wp.float64),
    adj_index: wp.array2d(dtype=wp.vec2d),
    adj_weight: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    flat = start + i
    if flat >= n_cases:
        return
    if status[i] != 0:
        wp.atomic_max(failure, 0, status[i])
        return
    group = flat // radius.shape[1]
    slot = flat % radius.shape[1]
    r = radius[group, slot]
    ext_seed = adj_bulk[group, 0] + adj_bulk[group, 2]
    sca_seed = adj_bulk[group, 1] - adj_bulk[group, 2]
    inverse_scale = wp.float64(1.0) / scale[group]
    factor = (
        wp.float64(2.0)
        * _PI
        * inverse_scale
        * inverse_scale
        * weight[group, slot]
    )
    adj = factor * (ext_seed * ext[i] + sca_seed * sca[i])
    if adj_radius:
        adj_radius[group, slot] += adj[2] * scale[group]
    if adj_index:
        adj_index[group, slot] += wp.vec2d(adj[0], adj[1]) / medium
    if adj_weight:
        adj_weight[group, slot] += (
            _PI
            * r
            * r
            * (
                adj_bulk[group, 0] * q[i, 0]
                + adj_bulk[group, 1] * q[i, 1]
                + adj_bulk[group, 2] * q[i, 2]
            )
        )


class BulkOpticsAdjoint:
    """Bulk extinction, scattering and absorption with an analytic Warp VJP.

    Construct for fixed (group, particle) dimensions and one vacuum
    wavelength per group, allowing duplicates. Groups can represent
    box/wavelength pairs or measurement windows. One workspace is used
    on its device's current stream at a time.

    Each evaluate() returns independent reduced arrays. With a Tape,
    the reverse pass recomputes the local coefficients one tile at a time.
    Inputs must stay alive and immutable through backward. Treat particle
    count and coefficient series order as discrete. Only derivatives of the
    smooth spherical optical model are supported.
    """

    def __init__(
        self,
        shape,
        wavelength,
        *,
        device="cuda:0",
        medium_index=1.0,
        particle_tile=256,
        coefficient_capacity=128,
    ):
        """Allocate fixed-capacity particle and Mie series scratch."""
        groups, particles = shape
        if np.iscomplexobj(wavelength) or np.iscomplexobj(medium_index):
            raise ValueError("Wavelengths and medium index must be real")
        wave = np.broadcast_to(
            np.asarray(wavelength, dtype=np.float64), (groups,)
        )
        if (
            min(shape) < 1
            or np.any(~np.isfinite(wave))
            or np.any(wave <= 0)
            or not np.isfinite(medium_index)
            or medium_index <= 0
        ):
            raise ValueError(
                "Require positive dimensions, wavelengths and medium"
            )
        _capacities(coefficient_capacity, particle_tile, 1)
        self.device = wp.get_device(device)
        self.shape = shape
        self._count = groups * particles
        self._tile = min(self._count, particle_tile)
        self._scale = wp.array(
            2 * pi * medium_index / wave, dtype=wp.float64, device=self.device
        )
        self._medium = wp.float64(medium_index)

        def empty(shape, dtype):
            return wp.empty(shape, dtype=dtype, device=self.device)

        tile = self._tile
        self._index = empty(tile, wp.vec2d)
        self._size = empty(tile, wp.float64)
        self._series = (
            empty((tile, coefficient_capacity + 1), wp.vec2d),
            empty((tile, coefficient_capacity), wp.vec2d),
            empty((tile, coefficient_capacity), wp.vec2d),
            empty(tile, wp.int32),
            empty(tile, wp.int32),
        )
        self._q = empty((tile, 7), wp.float64)
        self._ext = empty(tile, wp.vec3d)
        self._sca = empty(tile, wp.vec3d)

    def _launch(self, kernel, *args):
        wp.launch(
            kernel,
            self._tile,
            inputs=args,
            device=self.device,
            record_tape=False,
        )

    def _evaluate_tile(self, radius, index, weight, start):
        self._launch(
            _gather,
            radius,
            index,
            weight,
            self._scale,
            self._medium,
            start,
            self._count,
            self._index,
            self._size,
        )
        self._launch(
            mie_coefficients_kernel, self._index, self._size, *self._series
        )
        self._launch(
            mie_efficiencies_kernel,
            self._index,
            self._size,
            *self._series[1:],
            self._q,
        )

    def evaluate(self, radius, index, weight, *, tape=None):
        """Return device bulk coefficients (group, 3) [m⁻¹].

        Radius [m] and number-density weights [m⁻³] are float64. The
        absolute refractive index is a vec2d of (n,k). All arrays are
        contiguous with the prepared (group, particle) dimensions. To
        record this operation, pass the same tape as the surrounding
        Warp model. This call and the backward hook check device status.
        """
        _array(radius, wp.float64, self.shape, self.device, "radius")
        _array(index, wp.vec2d, self.shape, self.device, "index")
        _array(weight, wp.float64, self.shape, self.device, "weight")
        stream = wp.get_stream(self.device) if self.device.is_cuda else None
        bulk = wp.zeros(
            (self.shape[0], 3),
            dtype=wp.float64,
            device=self.device,
            requires_grad=tape is not None,
        )
        failure = wp.zeros(1, dtype=wp.int32, device=self.device)
        status = self._series[-1]
        for start in range(0, self._count, self._tile):
            self._evaluate_tile(radius, index, weight, start)
            self._launch(
                _reduce,
                radius,
                weight,
                self._q,
                status,
                start,
                self._count,
                bulk,
                failure,
            )
        self._check(bulk, failure)
        if tape is not None:
            tape.record_func(
                lambda: self._backward(
                    radius, index, weight, bulk, failure, stream
                ),
                [a for a in (radius, index, weight, bulk) if a.requires_grad],
            )
        return bulk

    def _check(self, bulk, failure):
        code = failure.numpy()[0]
        if code:
            raise ValueError(
                f"Bulk optical model failed with Mie status {code}"
            )
        if bulk is not None and not np.all(np.isfinite(bulk.numpy())):
            raise ValueError("Bulk optical model overflow")

    def _backward(self, radius, index, weight, bulk, failure, stream):
        with wp.ScopedStream(stream):
            for start in range(0, self._count, self._tile):
                self._evaluate_tile(radius, index, weight, start)
                self._launch(
                    mie_cross_section_sums_jacobian,
                    self._index,
                    self._size,
                    *self._series,
                    self._ext,
                    self._sca,
                )
                self._launch(
                    _propagate,
                    radius,
                    weight,
                    self._q,
                    self._ext,
                    self._sca,
                    self._series[-1],
                    self._scale,
                    self._medium,
                    start,
                    self._count,
                    bulk.grad,
                    radius.grad,
                    index.grad,
                    weight.grad,
                    failure,
                )
            self._check(None, failure)
