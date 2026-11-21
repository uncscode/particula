"""Bounded optical histories independent of aerosol snapshots."""

import json
import sqlite3
from contextlib import closing
from operator import index
from pathlib import Path

import numpy as np

from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.optics.population import PopulationOptics
from particula.optics.state import OpticalConfig, population_optics

_CHANNELS = PopulationOptics._fields
_DEFAULT = (
    "extinction",
    "scattering",
    "absorption",
    "single_scattering_albedo",
    "asymmetry",
    "extinction_valid",
    "scattering_valid",
)


def _channels(variables):
    selected = tuple(variables)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in _CHANNELS for name in selected)
    ):
        raise ValueError("Choose unique bulk optical channel names")
    needed = {
        "single_scattering_albedo": "extinction_valid",
        "asymmetry": "scattering_valid",
    }
    return selected + tuple(
        mask
        for name, mask in needed.items()
        if name in selected and mask not in selected
    )


class OpticalHistory:
    """Append and read atomic optical chunks in a SQLite file.

    Every chunk has time [s] and bulk arrays with axes (time, box, wavelength).
    Coefficients are in inverse metres. Albedo and asymmetry are dimensionless
    and accompanied by validity masks. One writer owns the observation buffer.
    """

    def __init__(
        self,
        path,
        config,
        *,
        variables=_DEFAULT,
        provenance=None,
        water_vapor=None,
    ):
        """Configure output; a named water gas enables box RH observations."""
        if water_vapor is not None and (
            not isinstance(water_vapor, str) or not water_vapor.strip()
        ):
            raise ValueError(
                "Select water vapor by its native gas species name"
            )
        self.water_vapor = water_vapor
        self.path = Path(path)
        self.config = config
        self.variables = _channels(variables)
        self._provenance = json.dumps(
            provenance or {}, sort_keys=True, allow_nan=False
        )
        self._last_time = None

    @property
    def last_time(self):
        """Time of the last committed row, or None for an empty history."""
        return self._last_time

    def _water_lane(self, gas_names):
        if self.water_vapor is None:
            return None
        if self.water_vapor not in gas_names:
            raise ValueError(
                "Optical history water vapor is absent from the gas"
            )
        return gas_names.index(self.water_vapor)

    def _bind(self, boxes, distribution):
        humidity = {}
        if self.water_vapor is not None:
            humidity["relative_humidity"] = {
                "gas_species": self.water_vapor,
                "saturation_model": "Buck water/ice",
                "temperature": "live gas temperature",
                "axes": ["time", "box"],
                "unit": "1",
            }
        metadata = json.dumps(
            {
                **self.config.metadata(),
                "version": 1,
                "axes": ["time", "box", "wavelength"],
                "boxes": boxes,
                "distribution": distribution,
                "variables": list(self.variables),
                "units": {
                    "time": "s",
                    **{
                        name: "m^-1" if _CHANNELS.index(name) < 5 else "1"
                        for name in self.variables
                    },
                },
                "provenance": json.loads(self._provenance),
                **humidity,
            },
            sort_keys=True,
        )
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS optics_metadata (description TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS optics_chunks "
                "(first_time REAL PRIMARY KEY, last_time REAL, "
                "n_rows INTEGER, data BLOB)"
            )
            previous = db.execute(
                "SELECT description FROM optics_metadata"
            ).fetchone()
            if previous is None:
                db.execute(
                    "INSERT INTO optics_metadata VALUES (?)", (metadata,)
                )
            elif previous[0] != metadata:
                raise ValueError(
                    "Optical history configuration or layout has changed"
                )
            last = db.execute(
                "SELECT last_time FROM optics_chunks "
                "ORDER BY first_time DESC LIMIT 1"
            ).fetchone()
        self._last_time = None if last is None else last[0]

    def _write(self, times, values, humidity):
        humidity_bytes = b""
        if humidity is not None:
            if np.any(~np.isfinite(humidity) | (humidity < 0)):
                raise ValueError("Optical history relative humidity is invalid")
            humidity_bytes = np.asarray(humidity, dtype="<f8").tobytes()
        times = np.asarray(times, dtype="<f8")
        selected = np.asarray(
            values[..., [_CHANNELS.index(name) for name in self.variables]],
            dtype="<f8",
        )
        if (
            times.size == 0
            or np.any(~np.isfinite(times))
            or times[0] < 0
            or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(selected))
        ):
            raise ValueError("Invalid optical history chunk")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute(
                "SELECT last_time FROM optics_chunks "
                "ORDER BY first_time DESC LIMIT 1"
            ).fetchone()
            if last is not None and times[0] <= last[0]:
                raise ValueError(
                    "Optical history writer overlaps an existing chunk"
                )
            db.execute(
                "INSERT INTO optics_chunks VALUES (?, ?, ?, ?)",
                (
                    times[0].item(),
                    times[-1].item(),
                    times.size,
                    times.tobytes() + selected.tobytes() + humidity_bytes,
                ),
            )
        self._last_time = times[-1].item()


def read_optical_history(path):
    """Yield metadata and arrays for each committed chunk."""
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        metadata = json.loads(
            db.execute("SELECT description FROM optics_metadata").fetchone()[0]
        )
        if metadata["version"] != 1:
            raise ValueError("Unsupported optical history version")
        variables = metadata["variables"]
        shape = (
            metadata["boxes"],
            len(metadata["wavelength_m"]),
            len(variables),
        )
        for count, data in db.execute(
            "SELECT n_rows, data FROM optics_chunks ORDER BY first_time"
        ):
            buffer = np.frombuffer(data, dtype="<f8")
            optical_end = count * (1 + int(np.prod(shape)))
            values = buffer[count:optical_end].reshape((count, *shape))
            arrays = {"time": buffer[:count]}
            if "relative_humidity" in metadata:
                arrays["relative_humidity"] = buffer[optical_end:].reshape(
                    (count, metadata["boxes"])
                )
            elif buffer.size != optical_end:
                raise ValueError("Unexpected optical history payload length")
            arrays.update(
                {
                    name: values[..., i].astype(bool)
                    if name.endswith("_valid")
                    else values[..., i]
                    for i, name in enumerate(variables)
                }
            )
            yield metadata, arrays


class OpticalObserver:
    """Bind a bulk optical history to a CPU or fixed-step Warp owner.

    Use run_until or run_prepared with optics=observer, or call record()
    at chosen accepted states and flush() before closing the owner.
    """

    def __init__(
        self,
        simulation,
        history,
        *,
        batch_size=32,
        particle_tile=256,
        coefficient_capacity=10090,
    ):
        """Allocate bounded observation storage for this owner."""
        from particula.execution.cpu_simulation import CPUSimulation
        from particula.execution.prepared import PreparedSimulation
        from particula.execution.prepared_recording import _ready

        if not isinstance(history, OpticalHistory) or not isinstance(
            history.config, OpticalConfig
        ):
            raise TypeError(
                "An OpticalHistory with an OpticalConfig is required"
            )
        self.simulation = simulation
        self.history = history
        self._batch_size = index(batch_size)
        if self._batch_size <= 0 or isinstance(batch_size, bool):
            raise ValueError("Optical batch size must be a positive integer")
        self._times = []
        self._cpu_rows = []
        self._cpu_humidity = []
        self._device_batch = None
        if isinstance(simulation, CPUSimulation):
            particles = simulation._state.particles
            self._water_lane = history._water_lane(simulation._state.gas.name)
            history.config.validate_materials(particles.n_species)
            history._bind(particles.n_boxes, particles.distribution_type)
        elif type(simulation) is PreparedSimulation:
            from particula.optics.gpu.state import DeviceOpticalBatch

            with simulation._lock:
                _ready(simulation)
                particles = simulation._session.particles
                water_lane = history._water_lane(
                    simulation._session.metadata.gas_names
                )
                distribution = {1: "discrete", 3: "particle_resolved"}.get(
                    particles.distribution_kind
                )
                with simulation._scope():
                    self._device_batch = DeviceOpticalBatch(
                        simulation,
                        history.config,
                        self._batch_size,
                        particle_tile,
                        coefficient_capacity,
                        water_lane=water_lane,
                    )
                history._bind(particles.masses.shape[0], distribution)
        else:
            raise TypeError(
                "Optical recording requires CPUSimulation or PreparedSimulation"
            )

    def _check_start(self, time):
        last = self._times[-1] if self._times else self.history.last_time
        if last is not None and last > time:
            raise ValueError("Optical history is ahead of the simulation")
        return last

    def record(self):
        """Evaluate live state and skip duplicate continuation times."""
        if self._device_batch is None:
            self._record(self.simulation.time)
        else:
            from particula.execution.prepared_recording import _ready, _time

            with self.simulation._lock:
                _ready(self.simulation)
                self._record(_time(self.simulation))

    def _record(self, time):
        if len(self._times) == self._batch_size:
            self.flush()
        if self._check_start(time) == time:
            return
        if self._device_batch is None:
            state = self.simulation._state
            values = population_optics(state.particles, self.history.config)
            if self._water_lane is not None:
                humidity = WaterBuckStrategy().saturation_ratio(
                    state.gas.concentration[:, self._water_lane],
                    state.gas.molar_mass[self._water_lane],
                    state.environment.temperature,
                )
                self._cpu_humidity.append(humidity)
            self._cpu_rows.append(np.stack(values, axis=-1))
        else:
            self._device_batch.enqueue(len(self._times))
        self._times.append(time)
        if len(self._times) == self._batch_size:
            self.flush()

    def flush(self):
        """Publish all buffered rows once every row has passed validation."""
        if not self._times:
            return
        if self._device_batch is None:
            humidity = (
                np.stack(self._cpu_humidity) if self._cpu_humidity else None
            )
            self._publish(np.stack(self._cpu_rows), humidity)
        else:
            with self.simulation._lock:
                self._publish(*self._device_batch.download(len(self._times)))

    def _publish(self, values, humidity):
        self.history._write(self._times, values, humidity)
        self._times.clear()
        self._cpu_rows.clear()
        self._cpu_humidity.clear()
