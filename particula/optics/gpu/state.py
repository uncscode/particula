# Warp DSL annotations describe native device buffers.
# mypy: disable-error-code="index,valid-type"
"""Read live native particle arrays into the prepared optical workspace."""

from math import pi

import numpy as np
import warp as wp

from particula.gpu.kernels.thermodynamics import _buck_vapor_pressure
from particula.gpu.warp_types import WarpParticleData
from particula.optics.gpu._complex import divide
from particula.optics.gpu.bulk import prepare_population_mie
from particula.optics.state import lorentz_lorenz
from particula.util.constants import GAS_CONSTANT

_RADIUS_FACTOR = wp.constant(wp.float64(3.0 / (4.0 * pi)))
_CUBE_ROOT = wp.constant(wp.float64(1.0 / 3.0))
_TWO_PI = wp.constant(wp.float64(2.0 * pi))
_GAS = wp.constant(wp.float64(GAS_CONSTANT))


@wp.kernel
def _fill_cases(
    particles: WarpParticleData,
    wavelengths: wp.array(dtype=wp.float64),
    configured_index: wp.array2d(dtype=wp.vec2d),
    mixed: int,
    medium: wp.float64,
    index: wp.array2d(dtype=wp.vec2d),
    size: wp.array2d(dtype=wp.float64),
    radius: wp.array2d(dtype=wp.float64),
    weight: wp.array2d(dtype=wp.float64),
):
    group, slot = wp.tid()  # type: ignore[misc]
    wave = group % wavelengths.shape[0]
    box = group // wavelengths.shape[0]
    count = particles.concentration[box, slot]
    if particles.distribution_kind == 3:
        count = count / particles.volume[box]
    weight[group, slot] = count
    r = wp.float64(0.0)
    absolute_index = configured_index[wave, 0]
    if count > wp.float64(0.0):
        total = wp.float64(0.0)
        polar = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        for material in range(particles.masses.shape[2]):
            volume = (
                particles.masses[box, slot, material]
                / particles.density[material]
            )
            total = total + volume
            if mixed != 0:
                polar = polar + volume * configured_index[wave, material]
        r = wp.pow(total * _RADIUS_FACTOR, _CUBE_ROOT)
        if mixed != 0 and total > wp.float64(0.0):  # type: ignore[operator]
            polar = polar / total
            one = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
            dielectric = divide(one + wp.float64(2.0) * polar, one - polar)
            magnitude = wp.length(dielectric)  # type: ignore[call-overload]
            if dielectric[0] >= wp.float64(0.0):
                real = wp.sqrt((magnitude + dielectric[0]) / wp.float64(2.0))
                imag = dielectric[1] / (wp.float64(2.0) * real)
                absolute_index = wp.vec2d(real, imag)
            else:
                imag = wp.sqrt((magnitude - dielectric[0]) / wp.float64(2.0))
                real = dielectric[1] / (wp.float64(2.0) * imag)
                absolute_index = wp.vec2d(real, imag)
    if mixed != 0 and r == wp.float64(0.0):
        absolute_index = wp.vec2d(medium, wp.float64(0.0))
    radius[group, slot] = r
    size[group, slot] = _TWO_PI * (r / wavelengths[wave]) * medium
    index[group, slot] = absolute_index / medium


@wp.kernel
def _store_row(
    result: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    row: int,
    rows: wp.array3d(dtype=wp.float64),
    statuses: wp.array2d(dtype=wp.int32),
):
    group = wp.tid()
    for channel in range(9):
        rows[row, group, channel] = result[group, channel]
    statuses[row, group] = status[group]


@wp.kernel
def _relative_humidity_row(
    vapor: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    species: int,
    row: int,
    humidity: wp.array2d(dtype=wp.float64),
):
    box = wp.tid()
    t = temperature[box]
    partial_pressure = vapor[box, species] * _GAS * t / molar_mass[species]
    humidity[row, box] = partial_pressure / _buck_vapor_pressure(t)


class ParticleOpticalCases:
    """Prepare the optical inputs shared by resident observers and processes."""

    def __init__(self, particles, config, *, device, stream):
        """Allocate input buffers and retain a refresh command."""
        boxes, slots, materials = particles.masses.shape
        if particles.distribution_kind not in (1, 3) or slots < 1:
            raise ValueError(
                "Resident optics require nonempty discrete or resolved slots"
            )
        config.validate_materials(materials)
        wave = np.asarray(config.wavelength)
        groups = boxes * wave.size
        mixed = config.species_index is not None
        indices = (
            lorentz_lorenz(config.species_index)
            if mixed
            else np.asarray(config.particle_index)[:, None]
        )
        wavelengths = wp.array(wave, dtype=wp.float64, device=device)
        indices = wp.array(
            np.stack((indices.real, indices.imag), axis=-1),
            dtype=wp.vec2d,
            device=device,
        )
        self.index = wp.empty((groups, slots), dtype=wp.vec2d, device=device)
        self.size, self.radius, self.weight = [
            wp.empty((groups, slots), dtype=wp.float64, device=device)
            for _ in range(3)
        ]
        self._inputs = (particles, wavelengths, indices)
        self._stream = stream
        self._fill = wp.launch(
            _fill_cases,
            (groups, slots),
            inputs=[
                particles,
                wavelengths,
                indices,
                int(mixed),
                wp.float64(config.medium_index),
                self.index,
                self.size,
                self.radius,
                self.weight,
            ],
            device=device,
            stream=stream,
            record_cmd=True,
        )

    def enqueue(self):
        """Fill optical coordinates from the live particle state."""
        self._fill.launch(stream=self._stream)


class DeviceOpticalBatch:
    """Reduce on the owner stream and retain a fixed number of optical rows."""

    def __init__(
        self,
        owner,
        config,
        batch_size,
        particle_tile,
        coefficient_capacity,
        *,
        water_lane=None,
    ):
        """Bind device workspaces to the live particle arrays."""
        particles = owner._session.particles
        device = owner.device.native
        self._cases = ParticleOpticalCases(
            particles,
            config,
            device=device,
            stream=owner._stream,
        )
        boxes = particles.masses.shape[0]
        wave_count = len(config.wavelength)
        groups = boxes * wave_count
        self._owner = owner
        self._water_lane = water_lane
        self._humidity = (
            wp.empty((batch_size, boxes), dtype=wp.float64, device=device)
            if water_lane is not None
            else None
        )
        self._shape = (boxes, wave_count, 9)
        self._rows = wp.empty(
            (batch_size, groups, 9), dtype=wp.float64, device=device
        )
        self._statuses = wp.empty(
            (batch_size, groups), dtype=wp.int32, device=device
        )
        self._workspace = prepare_population_mie(
            self._cases.index,
            self._cases.size,
            self._cases.radius,
            self._cases.weight,
            particle_tile=particle_tile,
            coefficient_capacity=coefficient_capacity,
            stream=owner._stream,
        )

    def enqueue(self, row):
        """Append device results and status to a batch slot."""
        owner = self._owner
        self._cases.enqueue()
        self._workspace.enqueue()
        wp.launch(
            _store_row,
            self._workspace.status.size,
            inputs=[
                self._workspace.optics,
                self._workspace.status,
                row,
                self._rows,
                self._statuses,
            ],
            device=owner.device.native,
            stream=owner._stream,
        )

        if self._humidity is not None:
            session = owner._session
            wp.launch(
                _relative_humidity_row,
                self._shape[0],
                inputs=[
                    session.gas.concentration,
                    session.gas.molar_mass,
                    session.environment.temperature,
                    self._water_lane,
                    row,
                    self._humidity,
                ],
                device=owner.device.native,
                stream=owner._stream,
            )

    def download(self, count):
        """Check physics and optics before exposing any row of the batch."""
        owner = self._owner
        owner.synchronize()
        with owner._scope():
            status = self._statuses[:count].numpy()
            if np.any(status):
                codes = np.unique(status[status != 0])
                raise RuntimeError(f"Resident optics failed with codes {codes}")
            optics = self._rows[:count].numpy().reshape((count, *self._shape))
            rh = (
                self._humidity[:count].numpy()
                if self._humidity is not None
                else None
            )
            return optics, rh
