"""Explicit prepared-GPU observation transfers on the frozen step grid.

Attaching an observer downloads one native snapshot to bind analysis metadata.
Subsequent observations explicitly choose reduced diagnostics or full snapshots.
The runner never changes the prepared timestep, capture plan or RNG schedule.
"""

import numpy as np

from particula import _cpu_batch
from particula.execution.analysis_io import AnalysisWriter
from particula.execution.prepared import PreparedSimulation
from particula.execution.recording import Recorder

_REDUCED = {
    "number_density": "number_concentration",
    "particle_mass_density": "particle_mass_concentration",
    "gas_concentration": "gas_concentration",
    "temperature": "temperature",
    "pressure": "pressure",
    "volume": "volume",
}


def _ready(simulation):
    if type(simulation) is not PreparedSimulation:
        raise TypeError("requires an exact PreparedSimulation owner")
    if simulation._wp.get_device(simulation.device.native).is_capturing:
        raise RuntimeError(
            "workflow observation cannot run during CUDA capture"
        )
    simulation._require_active()
    if simulation.time_step <= 0:
        raise ValueError("physical-time recording requires a positive timestep")


def _time(simulation):
    # Derive observation labels from the integer step index, not a repeated
    # floating-point sum. This gives stable labels after exact restart.
    return (
        simulation._initial_time
        + simulation.completed_steps * simulation.time_step
    )


def _index(simulation, time):
    position = (time - simulation._initial_time) / simulation.time_step
    if not np.isfinite(position):
        raise ValueError("output time cannot be represented on the step grid")
    tolerance = 8 * np.spacing(max(abs(position), 1.0))
    if tolerance >= 0.25:
        raise ValueError("step grid cannot distinguish this output time")
    index = round(position)
    if abs(position - index) > tolerance:
        raise ValueError("output times must lie on the prepared step grid")
    return index


class PreparedObserver:
    """Bind a Recorder to one prepared owner and an explicit download policy.

    Construction downloads one full native snapshot for immutable species,
    dimensions and measure metadata, but appends no row. download='diagnostics'
    subsequently downloads only existing per-box/species reductions and forcing;
    slot variables require download='snapshot'. Returned history is detached.
    Existing captured *replay* may be observed between steps; running this host
    workflow inside an active CUDA capture is rejected before any transfer.
    """

    def __init__(self, simulation, recorder, *, download):
        """Validate the observation profile before the initial full download."""
        self._writer = (
            recorder if isinstance(recorder, AnalysisWriter) else None
        )
        if self._writer is not None:
            recorder = self._writer.recorder
        if not isinstance(recorder, Recorder):
            raise TypeError("recorder must be a Recorder")
        if download not in ("diagnostics", "snapshot"):
            raise ValueError("choose diagnostics or snapshot downloads")
        if download == "diagnostics" and any(
            name not in _REDUCED for name in recorder.variables
        ):
            raise ValueError("slot variables require snapshot downloads")
        if type(simulation) is not PreparedSimulation:
            raise TypeError("requires an exact PreparedSimulation owner")
        self.simulation, self.recorder, self.download = (
            simulation,
            recorder,
            download,
        )
        self.initial_snapshot_downloads = 0
        self.observation_downloads = 0
        with simulation._lock:
            _ready(simulation)
            probe = Recorder(
                recorder.variables,
                max_records=1,
                max_bytes=recorder.max_bytes,
            )
            probe.record(_time(simulation), simulation.snapshot())
            self._metadata = probe.metadata
            if recorder.metadata is not None and (
                recorder.metadata != self._metadata
            ):
                raise ValueError(
                    "observation layout or species provenance changed"
                )
            self.initial_snapshot_downloads = 1

    def record(self):
        """Synchronize and append one complete-step observation, if new."""
        simulation = self.simulation
        with simulation._lock:
            _ready(simulation)
            time = _time(simulation)
            if self.recorder.last_time == time:
                self._flush()
                return
            if self.recorder.last_time is not None and (
                self.recorder.last_time > time
            ):
                raise ValueError("observation time must not move backwards")
            if self.download == "snapshot":
                state = simulation.snapshot()
                self.observation_downloads += 1
                self.recorder.record(time, state)
                self._flush()
                return
            values = simulation.diagnostics()
            self.observation_downloads += 1
            if tuple(values.gas_names) != tuple(self._metadata["gas_names"]):
                raise ValueError(
                    "prepared diagnostic species provenance changed"
                )
            row = {
                name: getattr(values, _REDUCED[name])
                for name in self.recorder.variables
            }
            for name, value in row.items():
                if np.any(value < 0) or (
                    name in ("volume", "temperature", "pressure")
                    and np.any(value == 0)
                ):
                    raise ValueError(
                        "prepared diagnostic is outside its domain"
                    )
            self.recorder._append(time, self._metadata, row)
            self._flush()

    def _flush(self):
        if self._writer is not None:
            self._writer.flush()


def run_prepared(simulation, end_time, *, output_times=(), observer=None):
    """Run complete fixed steps and explicitly observe the requested grid.

    All requested/final times must be on the frozen grid before any work is
    enqueued. Older requests are skipped on continuation. Output labels use
    initial_time + integer_step * dt, including the final row; decimal inputs
    within floating-point roundoff of that grid are admitted. No interpolation,
    shortened step, automatic capture or checkpoint is performed. The final
    synchronize surfaces asynchronous failures even without an observer.
    """
    if type(simulation) is not PreparedSimulation:
        raise TypeError("requires an exact PreparedSimulation owner")
    if observer is not None and (
        not isinstance(observer, PreparedObserver)
        or observer.simulation is not simulation
    ):
        raise ValueError("observer belongs to a different prepared owner")
    end_time = _cpu_batch.duration(end_time, 1)
    times = np.asarray(output_times, dtype=float)
    if (
        times.ndim != 1
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or np.any(times > end_time)
    ):
        raise ValueError("output times must be ordered within the run interval")
    with simulation._lock:
        _ready(simulation)
        end = _index(simulation, end_time)
        start = simulation.completed_steps
        if end < start:
            raise ValueError("end_time precedes accepted simulation time")
        indices = sorted(
            {_index(simulation, float(time)) for time in times} | {end}
        )
        # Preflight the recorder's time too, so a mismatched history cannot
        # consume physics steps before its first failed observation.
        if observer is not None and observer.recorder.last_time is not None:
            if _index(simulation, observer.recorder.last_time) > start:
                raise ValueError("recorder is ahead of the prepared simulation")
        for index in indices:
            if index < start:
                continue
            simulation.run(index - simulation.completed_steps)
            if observer is not None:
                observer.record()
        simulation.synchronize()
    return simulation
