# mypy: disable-error-code="valid-type,index,operator"
"""Resident equilibrium CCN sampling, using the prepared owner's stream."""

import numpy as np
import warp as wp

from particula.gpu.properties.kappa_kohler import (
    kappa_kohler_critical_dry_diameter,
)
from particula.gpu.warp_types import WarpParticleData

_ZERO = wp.constant(wp.float64(0.0))
_VOLUME_FACTOR = wp.constant(wp.float64(6.0 / np.pi))
_TIE = wp.constant(wp.float64(1.0 - 32 * np.finfo(float).eps))


@wp.kernel
def _thresholds(
    common: wp.array(dtype=wp.float64),
    ratios: wp.array(dtype=wp.float64),
    length: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    product: wp.float64,
    scales: wp.array2d(dtype=wp.float64),
):
    box, setting = wp.tid()  # type: ignore[misc]
    a = length[box]
    if product > _ZERO:
        a = product / temperature[box]
    scales[box, setting] = kappa_kohler_critical_dry_diameter(
        common[box], ratios[setting], a
    )


@wp.kernel
def _begin(
    row: int,
    thresholds: wp.array2d(dtype=wp.float64),
    common: int,
    counts: wp.array3d(dtype=wp.float64),
    diameters: wp.array3d(dtype=wp.float64),
    status: wp.array2d(dtype=int),
):
    box, setting = wp.tid()  # type: ignore[misc]
    counts[row, box, setting] = _ZERO
    if common:
        cut = thresholds[box, setting]
        diameters[row, box, setting] = cut
        if cut <= _ZERO:
            wp.atomic_max(status, row, box, 1)


@wp.func
def _dry_volume(p: WarpParticleData, box: int, slot: int, water: int):
    dry = wp.float64(0.0)
    for lane in range(p.masses.shape[2]):
        if lane != water:
            v = p.masses[box, slot, lane] / p.density[lane]
            if not wp.isfinite(v) or v < _ZERO:
                return wp.float64(-1.0)
            dry += v
    return dry


@wp.func
def _volume_weighted_kappa(
    p: WarpParticleData,
    box: int,
    slot: int,
    water: int,
    dry: wp.float64,
    species: wp.array(dtype=wp.float64),
):
    k = wp.float64(0.0)
    for lane in range(p.masses.shape[2]):
        if lane != water:
            k += (p.masses[box, slot, lane] / p.density[lane] / dry) * species[
                lane
            ]
    return k


@wp.kernel
def _count(
    p: WarpParticleData,
    temperature: wp.array(dtype=wp.float64),
    ratios: wp.array(dtype=wp.float64),
    species: wp.array(dtype=wp.float64),
    length: wp.array(dtype=wp.float64),
    product: wp.float64,
    water: int,
    mixed: int,
    thresholds: wp.array2d(dtype=wp.float64),
    row: int,
    counts: wp.array3d(dtype=wp.float64),
    status: wp.array2d(dtype=int),
):
    box, slot, setting = wp.tid()  # type: ignore[misc]
    weight = p.population[box, slot]
    if (
        not wp.isfinite(weight)
        or weight < _ZERO
        or not wp.isfinite(p.volume[box])
        or p.volume[box] <= _ZERO
    ):
        wp.atomic_max(status, row, box, 2)
        return
    if weight == _ZERO:
        return
    dry = _dry_volume(p, box, slot, water)
    if dry < _ZERO or not wp.isfinite(dry):
        wp.atomic_max(status, row, box, 2)
        return
    if dry == _ZERO:
        return
    cut = wp.float64(0.0)
    if mixed:
        k = _volume_weighted_kappa(p, box, slot, water, dry, species)
        a = length[box]
        if product > _ZERO:
            a = product / temperature[box]
        cut = kappa_kohler_critical_dry_diameter(k, ratios[setting], a)
    else:
        cut = thresholds[box, setting]
    if cut <= _ZERO:
        wp.atomic_max(status, row, box, 1)
        return
    if wp.cbrt(dry * _VOLUME_FACTOR) >= cut * _TIE:
        wp.atomic_add(counts, row, box, setting, weight)


@wp.kernel
def _normalize(
    row: int,
    volume: wp.array(dtype=wp.float64),
    counts: wp.array3d(dtype=wp.float64),
):
    box, setting = wp.tid()  # type: ignore[misc]
    counts[row, box, setting] = counts[row, box, setting] / volume[box]


class DeviceCCNBatch:
    """Prepare constant inputs and reusable compact output buffers."""

    def __init__(self, owner, config, capacity):
        self.owner = owner
        p = owner._session.particles
        boxes, slots, materials = p.masses.shape
        distribution = {1: "discrete", 3: "particle_resolved"}.get(
            p.distribution_kind
        )
        common, length = config.bind(boxes, materials, distribution)
        self.groups = (boxes, len(config.saturation_ratios))
        self.shape = (boxes, slots, self.groups[1])
        device = owner.device.native

        def upload(values):
            return wp.array(np.array(values), dtype=wp.float64, device=device)

        self.ratios = upload(config.saturation_ratios)
        self.mixed = common is None
        self.common = upload(np.zeros(boxes) if self.mixed else common)
        self.species = upload(config.material_kappa or (0.0,))
        self.length = upload(np.ones(boxes) if length is None else length)
        self.product = wp.float64(
            config.kelvin_temperature_product if length is None else 0
        )
        self.dynamic = length is None
        self.water = -1 if config.water_index is None else config.water_index
        self.thresholds = wp.empty(self.groups, dtype=wp.float64, device=device)
        self.counts = wp.empty(
            (capacity, *self.groups), dtype=wp.float64, device=device
        )
        self.diameters = wp.empty(
            (capacity if not self.mixed else 1, *self.groups),
            dtype=wp.float64,
            device=device,
        )
        self.status = wp.zeros((capacity, boxes), dtype=int, device=device)
        if not self.dynamic and not self.mixed:
            self.refresh()

    def _launch(self, kernel, shape, args):
        wp.launch(
            kernel,
            shape,
            inputs=args,
            device=self.owner.device.native,
            stream=self.owner._stream,
        )

    def refresh(self):
        """Compute common thresholds once per setting and box."""
        self._launch(
            _thresholds,
            self.groups,
            [
                self.common,
                self.ratios,
                self.length,
                self.owner._session.environment.temperature,
                self.product,
                self.thresholds,
            ],
        )

    def enqueue(self, row):
        """Sample the current accepted device state."""
        session = self.owner._session
        if self.dynamic and not self.mixed:
            self.refresh()
        self._launch(
            _begin,
            self.groups,
            [
                row,
                self.thresholds,
                int(not self.mixed),
                self.counts,
                self.diameters,
                self.status,
            ],
        )
        if self.shape[1]:
            self._launch(
                _count,
                self.shape,
                [
                    session.particles,
                    session.environment.temperature,
                    self.ratios,
                    self.species,
                    self.length,
                    self.product,
                    self.water,
                    int(self.mixed),
                    self.thresholds,
                    row,
                    self.counts,
                    self.status,
                ],
            )
        self._launch(
            _normalize,
            self.groups,
            [row, session.particles.volume, self.counts],
        )

    def download(self, count):
        """Return validated compact rows on explicit flush."""
        self.owner.synchronize()
        with self.owner._scope():
            if np.any(self.status[:count].numpy()):
                raise FloatingPointError(
                    "Invalid resident CCN state or critical diameter"
                )
            # Warp CPU exposes its buffer by reference. Detach completed
            # rows before the next batch reuses those slots.
            number = self.counts[:count].numpy().copy()
            result = {"equilibrium_ccn_number_concentration": number}
            if not self.mixed:
                result["critical_dry_diameter"] = (
                    self.diameters[:count].numpy().copy()
                )
            if not all(np.all(np.isfinite(value)) for value in result.values()):
                raise FloatingPointError("CCN history outside float64 range")
            return result
