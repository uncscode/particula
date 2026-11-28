"""Detached native observations at accepted complete-step physical times.

Recording has no access to process RNG or intermediate trial states. Requested
output times explicitly shorten outer steps; this may alter a numerical or
stochastic trajectory. An unchanged step schedule is required for bitwise
observer-invariance claims. Full particle arrays are opt-in and byte bounded.
"""

from copy import deepcopy

import numpy as np

from particula import _cpu_batch

# Public observation keys map to native property, SI unit, and named axes.
OBSERVABLES = {
    "number_density": ("particles.number_density", "m^-3", ("box",)),
    "particle_mass_density": (
        "particles.species_mass_density",
        "kg m^-3",
        ("box", "material"),
    ),
    "gas_concentration": ("gas.concentration", "kg m^-3", ("box", "gas")),
    "temperature": ("environment.temperature", "K", ("box",)),
    "pressure": ("environment.pressure", "Pa", ("box",)),
    "volume": ("particles.volume", "m^3", ("box",)),
    "particle_masses": ("particles.masses", "kg", ("box", "slot", "material")),
    "particle_population": ("particles.population", "count", ("box", "slot")),
    "particle_concentration": (
        "particles.concentration",
        "population_measure",
        ("box", "slot"),
    ),
    "particle_charge": (
        "particles.charge",
        "elementary_charge",
        ("box", "slot"),
    ),
}


class Recorder:
    """Collect selected detached observations with explicit storage limits.

    metadata retains full gas order, distribution interpretation, material
    densities and dimensions. It is analysis provenance, not a restart state.
    max_bytes bounds retained numeric observations including times; Python
    object overhead is additional. Full slot arrays are opt-in. Volume is always
    recorded so population measures remain interpretable.
    """

    def __init__(
        self,
        variables=(
            "number_density",
            "particle_mass_density",
            "gas_concentration",
        ),
        *,
        max_records=100000,
        max_bytes=64 * 1024 * 1024,
    ):
        """Validate names and limits without accessing simulation state."""
        self.variables = tuple(variables)
        if not self.variables or len(set(self.variables)) != len(
            self.variables
        ):
            raise ValueError("choose nonempty unique observation names")
        if any(name not in OBSERVABLES for name in self.variables):
            raise ValueError("unknown observation name")
        if "volume" not in self.variables:
            self.variables += ("volume",)
        for value in (max_records, max_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "record and byte limits must be positive integers"
                )
        self.max_records, self.max_bytes = max_records, max_bytes
        self._times: list[float] = []
        self._rows: dict[str, list[np.ndarray]] = {
            name: [] for name in self.variables
        }
        self._metadata = None
        self._bytes = 0

    @property
    def metadata(self):
        """Return detached dimensions, units and species provenance."""
        return deepcopy(self._metadata)

    @property
    def last_time(self):
        """Return the last observation time or None for an empty history."""
        return self._times[-1] if self._times else None

    def record(self, time, state):
        """Append one accepted native observation; equal-time resume is a no-op.

        All metadata, shapes and storage limits validate before append. Layout
        changes require a new recorder; widths do not imply species order.
        """
        time = _cpu_batch.duration(time, 1)
        if self.last_time is not None:
            if time < self.last_time:
                raise ValueError("observation time must not move backwards")
            if time == self.last_time:
                return
        _cpu_batch.validate(state)
        particle = state.particles
        metadata = {
            "gas_names": list(state.gas.name),
            "distribution_type": particle.distribution_type,
            "gas_molar_mass_kg_mol": state.gas.molar_mass.tolist(),
            "gas_partitioning": state.gas.partitioning.tolist(),
            "radius_grid_m": (
                None
                if particle.radius_grid is None
                else particle.radius_grid.tolist()
            ),
            "material_density_kg_m3": particle.density.tolist(),
            "dimensions": {
                "box": particle.n_boxes,
                "slot": particle.masses.shape[1],
                "material": particle.n_species,
                "gas": len(state.gas.name),
            },
            "observables": {
                name: {
                    "unit": OBSERVABLES[name][1],
                    "axes": ["time", *OBSERVABLES[name][2]],
                }
                for name in self.variables
            },
        }
        metadata["population_storage"] = "extensive"
        if "particle_population" in metadata["observables"]:
            metadata["observables"]["particle_population"]["unit"] = (
                "m^-1"
                if particle.distribution_type == "continuous_pdf"
                else "count"
            )
        if "particle_concentration" in metadata["observables"]:
            metadata["observables"]["particle_concentration"]["unit"] = {
                "particle_resolved": "count",
                "discrete": "m^-3",
                "continuous_pdf": "m^-4",
            }[particle.distribution_type]
        if self._metadata is not None and metadata != self._metadata:
            raise ValueError("observation layout or species provenance changed")
        row = {}
        for name in self.variables:
            container, field = OBSERVABLES[name][0].split(".")
            value = np.array(
                getattr(getattr(state, container), field), copy=True
            )
            if not np.all(np.isfinite(value)):
                raise ValueError("observations must be finite")
            row[name] = value
        self._append(time, metadata, row)

    def _append(self, time, metadata, row):
        """Append native/reduced values through one atomic boundary."""
        time = _cpu_batch.duration(time, 1)
        if self.last_time is not None:
            if time < self.last_time:
                raise ValueError("observation time must not move backwards")
            if time == self.last_time:
                return
        if self._metadata is not None and metadata != self._metadata:
            raise ValueError("observation layout or species provenance changed")
        if set(row) != set(self.variables):
            raise ValueError("observation variables changed")
        row = {name: np.array(value, copy=True) for name, value in row.items()}
        for name, value in row.items():
            shape = tuple(
                metadata["dimensions"][axis] for axis in OBSERVABLES[name][2]
            )
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("observations require finite native shapes")
        size = 8 + sum(value.nbytes for value in row.values())
        if (
            len(self._times) >= self.max_records
            or self._bytes + size > self.max_bytes
        ):
            raise ValueError("recording storage limit exceeded")
        self._metadata = deepcopy(metadata)
        self._times.append(time)
        for name, value in row.items():
            self._rows[name].append(value)
        self._bytes += size

    def arrays(self):
        """Return detached stacked arrays; time always has units of seconds."""
        if not self._times:
            raise ValueError("no observations recorded")
        return {
            "time": np.array(self._times),
            **{name: np.stack(rows) for name, rows in self._rows.items()},
        }


def _observation_times(output_times, end_time):
    times = np.asarray(output_times, dtype=float)
    if (
        times.ndim != 1
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or np.any(times > end_time)
    ):
        raise ValueError(
            "Output times must be finite, increasing and within the run"
        )
    return set(times) | {end_time}


def _optical_schedule(simulation, observer, times, initial, start, end):
    if observer is None:
        if times is not None or initial:
            raise ValueError("An optical schedule requires an optical observer")
        return False, set()
    from particula.execution.optical_recording import OpticalObserver

    if (
        not isinstance(observer, OpticalObserver)
        or observer.simulation is not simulation
    ):
        raise ValueError("optical observer belongs to a different owner")
    observer._check_start(start)
    targets = set() if times is None else _observation_times(times, end)
    if initial:
        targets.add(start)
    return times is None, targets


def run_until(  # noqa: C901
    simulation,
    end_time,
    *,
    output_times=(),
    recorder=None,
    optics=None,
    optical_output_times=None,
    optical_initial=False,
):
    """Advance an owner and observe requested times plus final time.

    The owner supplies time, step(bound), and detached snapshot(). Each output
    is at a complete accepted step. With an OpticalObserver, the default optical
    schedule observes every accepted step without taking snapshots. Explicit
    optical_output_times may shorten CPU steps to reach those times. Set
    optical_initial=True to observe the starting state on the dense schedule.
    Fixed prepared owners instead use run_prepared.
    """
    end_time = _cpu_batch.duration(end_time, 1)
    start = _cpu_batch.duration(simulation.time, 1)
    if end_time < start:
        raise ValueError("end_time precedes accepted simulation time")
    targets = _observation_times(output_times, end_time)
    dense, optical_targets = _optical_schedule(
        simulation,
        optics,
        optical_output_times,
        optical_initial,
        start,
        end_time,
    )
    for target in sorted(targets | optical_targets):
        if target < start:
            continue
        while simulation.time < target:
            before = simulation.time
            simulation.step(target)
            if not before < simulation.time <= target:
                raise RuntimeError(
                    "owner failed to advance to an accepted bounded time"
                )
            if dense:
                optics.record()
        if target in optical_targets:
            optics.record()
        if target in targets and recorder is not None:
            if isinstance(recorder, Recorder) and (
                recorder.last_time == simulation.time
            ):
                continue
            recorder.record(simulation.time, simulation.snapshot())
    if optics is not None:
        optics.flush()
    return simulation
