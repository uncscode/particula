"""Accepted-step equilibrium CCN histories for CPU and prepared Warp owners."""

from dataclasses import asdict
from operator import index

import numpy as np

from particula.execution.analysis_io import AnalysisResult
from particula.execution.archive import read_archive, write_archive
from particula.particles.properties.ccn import CCNConfig, _sample_state
from particula.particles.properties.kappa_kohler import (
    get_kappa_kohler_critical_dry_diameter,
)


def _history_units(config):
    units = {"time": "s", "equilibrium_ccn_number_concentration": "m^-3"}
    if config.kappa is not None:
        units["critical_dry_diameter"] = "m"
    return units


class CCNObserver:
    """Sample equilibrium CCN on the owner's backend, retaining bounded history.

    Common-κ configurations also record critical dry diameters. Slots with
    different compositions can have different thresholds. Their bulk output
    retains number concentration; evaluate_ccn exposes individual thresholds.
    """

    def __init__(
        self, simulation, config, *, batch_size=32, max_bytes=64 << 20
    ):
        """Bind a frozen configuration and reserve capacity for compact rows."""
        from particula.execution.cpu_simulation import CPUSimulation
        from particula.execution.prepared import PreparedSimulation
        from particula.execution.prepared_recording import _ready

        if not isinstance(config, CCNConfig):
            raise TypeError("CCNObserver requires a CCNConfig")
        self.config = config
        self.simulation = simulation
        self._device_batch = None
        self._capacity = index(batch_size)
        self._max_bytes = index(max_bytes)
        if self._capacity < 1 or self._max_bytes < 1:
            raise ValueError("CCN batch_size and max_bytes must be positive")
        if isinstance(simulation, CPUSimulation):
            p = simulation._state.particles
            boxes, materials, distribution = (
                p.n_boxes,
                p.n_species,
                p.distribution_type,
            )
        elif type(simulation) is PreparedSimulation:
            with simulation._lock:
                _ready(simulation)
                device_p = simulation._session.particles
                boxes, _, materials = device_p.masses.shape
                distribution = {1: "discrete", 3: "particle_resolved"}.get(
                    device_p.distribution_kind
                )
        else:
            raise TypeError(
                "CCN recording requires a CPU or prepared simulation"
            )
        common, length = config.bind(boxes, materials, distribution)
        self._fixed = None
        if (
            isinstance(simulation, CPUSimulation)
            and common is not None
            and length is not None
        ):
            self._fixed = get_kappa_kohler_critical_dry_diameter(
                common[:, None], config.saturation_ratios, length[:, None]
            )
        self._metadata = {
            "schema": "equilibrium-ccn-v1",
            "configuration": asdict(config),
            "distribution_type": distribution,
            "boxes": boxes,
            "materials": materials,
            "axes": ["time", "box", "saturation_ratio"],
            "units": _history_units(config),
        }
        self._row_bytes = 8 * (
            1
            + boxes
            * len(config.saturation_ratios)
            * (1 + int(config.kappa is not None))
        )
        self._capacity = min(self._capacity, self._max_bytes // self._row_bytes)
        if not self._capacity:
            raise ValueError("CCN storage limit below one row")
        self._times = []
        self._published = []
        self._pending = []
        self._size = 0
        if type(simulation) is PreparedSimulation:
            from particula.execution._ccn_device import DeviceCCNBatch

            with simulation._lock, simulation._scope():
                _ready(simulation)
                self._device_batch = DeviceCCNBatch(
                    simulation, config, self._capacity
                )

    def _check_start(self, time):
        last = (
            self._times[-1]
            if self._times
            else (self._published[-1]["time"][-1] if self._published else None)
        )
        if last is not None and last > time:
            raise ValueError("CCN history is ahead of the simulation")
        return last

    def record(self):
        """Sample one accepted state, skipping the equal-time continuation."""
        if self._device_batch is None:
            self._record(self.simulation.time)
        else:
            from particula.execution.prepared_recording import _ready, _time

            with self.simulation._lock, self.simulation._scope():
                _ready(self.simulation)
                self._record(_time(self.simulation))

    def _record(self, time):
        if self._check_start(time) == time:
            return
        if (
            self._size + len(self._times) + 1
        ) * self._row_bytes > self._max_bytes:
            raise ValueError("CCN history storage limit exceeded")
        if len(self._times) == self._capacity:
            self.flush()
        if self._device_batch is not None:
            self._device_batch.enqueue(len(self._times))
        else:
            result = _sample_state(
                self.simulation._state, self.config, self._fixed
            )
            number = result.equilibrium_ccn_number_concentration
            row = {"equilibrium_ccn_number_concentration": number}
            if self.config.kappa is not None:
                row["critical_dry_diameter"] = result.critical_dry_diameter
            self._pending.append(row)
        self._times.append(time)
        if len(self._times) == self._capacity:
            self.flush()

    def flush(self):
        """Transfer and publish the complete pending batch after validation."""
        if not self._times:
            return
        if self._device_batch is None:
            chunk = {
                key: np.stack([row[key] for row in self._pending])
                for key in self._pending[0]
            }
        else:
            with self.simulation._lock:
                chunk = self._device_batch.download(len(self._times))
        chunk["time"] = np.array(self._times)
        self._published.append(chunk)
        self._size += len(self._times)
        self._times.clear()
        self._pending.clear()

    def arrays(self):
        """Flush and return detached arrays on their recorded time grid."""
        self.flush()
        if not self._published:
            raise ValueError("No CCN observations recorded")
        return {
            key: np.concatenate([chunk[key] for chunk in self._published])
            for key in self._published[0]
        }

    def latest(self):
        """Flush and return only the latest detached observation."""
        self.flush()
        if not self._published:
            raise ValueError("No CCN observations recorded")
        return {
            key: values[-1].copy()
            for key, values in self._published[-1].items()
        }

    def save(self, path):
        """Save compact rows and configuration by atomic replacement."""
        write_archive(
            path, kind="analysis", metadata=self._metadata, arrays=self.arrays()
        )


def read_ccn_history(path, *, max_bytes=128 << 20):
    """Read CCN output with units, axes and material configuration."""
    metadata, arrays, blobs = read_archive(
        path, kind="analysis", max_bytes=max_bytes
    )
    if metadata.get("schema") != "equilibrium-ccn-v1" or blobs:
        raise ValueError("Expected an equilibrium CCN history")
    config = CCNConfig(**metadata["configuration"])
    if (
        metadata.get("axes") != ["time", "box", "saturation_ratio"]
        or metadata.get("units") != _history_units(config)
        or any(
            type(metadata.get(key)) is not int or metadata[key] < 1
            for key in ("boxes", "materials")
        )
    ):
        raise ValueError("Invalid CCN units or dimension metadata")
    config.bind(
        metadata["boxes"], metadata["materials"], metadata["distribution_type"]
    )
    keys = set(_history_units(config))
    times = arrays.get("time", np.empty(0))
    shape = (times.size, metadata["boxes"], len(config.saturation_ratios))
    if (
        set(arrays) != keys
        or times.ndim != 1
        or not times.size
        or np.any(np.diff(times) <= 0)
        or any(arrays[key].shape != shape for key in keys - {"time"})
        or any(not np.all(np.isfinite(x) & (x >= 0)) for x in arrays.values())
        or (
            "critical_dry_diameter" in keys
            and np.any(arrays["critical_dry_diameter"] == 0)
        )
    ):
        raise ValueError("Invalid CCN history values or axes")
    return AnalysisResult(metadata, arrays)
