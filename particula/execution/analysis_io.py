"""Portable analysis results with explicit scientific and numerical provenance.

ZIP/NPY with a JSON manifest retains exact numeric dtypes and arbitrary named
axes without a new runtime dependency. HDF5/Zarr offer partial/chunked access
but require additional libraries and more complex atomic multi-object writes.
This bounded single-file format suits selected histories; it is not a streaming
store for unlimited particle snapshots, nor an exact-restart format.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particula import __version__
from particula.execution.archive import _json_bytes, read_archive, write_archive
from particula.execution.recording import OBSERVABLES, Recorder


@dataclass(frozen=True)
class AnalysisResult:
    """Own loaded numeric observations and JSON scientific provenance."""

    metadata: dict
    arrays: dict[str, np.ndarray]


class AnalysisWriter:
    """Compose accepted recording with atomic replacement of an analysis file.

    Each new output saves the bounded history so far. A write failure leaves
    accepted physics and the new in-memory row intact; the prior file remains
    usable. flush() retries that row without advancing physics or duplicating
    history. File writing is a host observation action, outside CUDA capture.
    """

    def __init__(
        self,
        recorder,
        path,
        *,
        configuration,
        numerical_settings,
        source_identity,
        compressed=True,
    ):
        """Detach and validate provenance without creating an output file."""
        if not isinstance(recorder, Recorder):
            raise TypeError("AnalysisWriter requires a Recorder")
        self.recorder = recorder
        self.path = Path(path)
        self._options = deepcopy(
            dict(
                configuration=configuration,
                numerical_settings=numerical_settings,
                source_identity=source_identity,
                compressed=compressed,
            )
        )
        for value in (configuration, numerical_settings, source_identity):
            if not isinstance(value, dict):
                raise TypeError("provenance inputs must be JSON objects")
            _json_bytes(value)
        self._saved_time = None

    @property
    def last_time(self):
        """Return the last recorded time, including an unsaved row."""
        return self.recorder.last_time

    def record(self, time, state):
        """Record a complete accepted state and save its accumulated history."""
        self.recorder.record(time, state)
        self.flush()

    def flush(self):
        """Save an unsaved row; successful repeats do not replace the file."""
        if self.last_time is not None and self.last_time != self._saved_time:
            save_analysis(self.recorder, self.path, **self._options)
            self._saved_time = self.last_time


def save_analysis(
    recorder,
    path,
    *,
    configuration,
    numerical_settings,
    source_identity,
    compressed=True,
):
    """Save observations atomically with reproducibility inputs.

    configuration, numerical_settings and source_identity are JSON objects, not
    executable process instances. The caller supplies physical assumptions,
    parameters, process order, tolerances, source revision and input identities.
    Time is seconds; the scientific metadata carries all other units and axes.
    """
    if not isinstance(recorder, Recorder):
        raise TypeError("save_analysis requires a Recorder")
    for value in (configuration, numerical_settings, source_identity):
        if not isinstance(value, dict):
            raise TypeError("provenance inputs must be JSON objects")
    metadata = {
        "schema": "particula-analysis",
        "version": 1,
        "package_version": __version__,
        "time_unit": "s",
        "science": recorder.metadata,
        "configuration": configuration,
        "numerical_settings": numerical_settings,
        "source_identity": source_identity,
    }
    arrays = recorder.arrays()
    _validate(metadata, arrays)
    write_archive(
        path,
        kind="analysis",
        metadata=metadata,
        arrays=arrays,
        compressed=compressed,
    )


def _validate_header(metadata):
    if (
        metadata["schema"] != "particula-analysis"
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
    ):
        raise ValueError("unsupported analysis schema")
    if metadata["time_unit"] != "s" or not isinstance(
        metadata["package_version"], str
    ):
        raise ValueError("missing analysis time/package identity")
    for name in ("configuration", "numerical_settings", "source_identity"):
        if not isinstance(metadata[name], dict):
            raise ValueError("analysis provenance must contain JSON objects")
    return metadata["science"]


def _validate_science(science):
    dimensions = science["dimensions"]
    if (
        set(dimensions) != {"box", "slot", "material", "gas"}
        or any(
            type(value) is not int or value < 0 for value in dimensions.values()
        )
        or dimensions["box"] < 1
    ):
        raise ValueError("invalid analysis dimensions")
    names = science["gas_names"]
    if (
        len(names) != dimensions["gas"]
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid ordered gas names")
    density = np.asarray(science["material_density_kg_m3"], dtype=float)
    if (
        density.shape != (dimensions["material"],)
        or np.any(density <= 0)
        or not np.all(np.isfinite(density))
    ):
        raise ValueError("invalid material density provenance")
    measure_units = {
        "particle_resolved": "count",
        "discrete": "m^-3",
        "continuous_pdf": "m^-4",
    }
    if science["distribution_type"] not in measure_units:
        raise ValueError("invalid population measure")
    _validate_species(science, dimensions)
    return dimensions, measure_units


def _validate(metadata, arrays):
    science = _validate_header(metadata)
    dimensions, measure_units = _validate_science(science)
    time = arrays["time"]
    if (
        time.ndim != 1
        or not time.size
        or np.any(time < 0)
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "analysis times must be nonnegative and strictly increasing"
        )
    observables = science["observables"]
    if set(arrays) != {"time", *observables} or "volume" not in observables:
        raise ValueError(
            "analysis array set does not match selected observables"
        )
    for name, specification in observables.items():
        if name not in OBSERVABLES:
            raise ValueError("unknown analysis observable")
        _, unit, axes = OBSERVABLES[name]
        if name == "particle_concentration":
            unit = measure_units[science["distribution_type"]]
        if specification != {"unit": unit, "axes": ["time", *axes]}:
            raise ValueError("analysis observable units/axes mismatch")
        if arrays[name].shape != (
            len(time),
            *(dimensions[axis] for axis in axes),
        ):
            raise ValueError("analysis observable dimensions mismatch")
    if any(
        not np.all(np.isfinite(array)) for array in arrays.values()
    ) or np.any(arrays["volume"] <= 0):
        raise ValueError("analysis values must be finite with positive volume")


def load_analysis(path, *, max_bytes=128 * 1024 * 1024):
    """Load and validate analysis data without executable deserialization."""
    metadata, arrays, blobs = read_archive(
        path, kind="analysis", max_bytes=max_bytes
    )
    if blobs:
        raise ValueError(
            "analysis archives cannot contain restart byte payloads"
        )
    try:
        _validate(metadata, arrays)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed analysis metadata") from error
    return AnalysisResult(metadata, arrays)


def _validate_species(science, dimensions):
    molar_mass = np.asarray(science["gas_molar_mass_kg_mol"], dtype=float)
    partitioning = science["gas_partitioning"]
    if (
        molar_mass.shape != (dimensions["gas"],)
        or not np.all(np.isfinite(molar_mass))
        or np.any(molar_mass <= 0)
        or len(partitioning) != dimensions["gas"]
        or any(type(value) is not bool for value in partitioning)
    ):
        raise ValueError("invalid gas species metadata")
    grid = science["radius_grid_m"]
    if science["distribution_type"] == "continuous_pdf":
        radius = np.asarray(grid, dtype=float)
        if (
            radius.shape != (dimensions["slot"],)
            or radius.size < 2
            or not np.all(np.isfinite(radius))
            or np.any(radius <= 0)
            or np.any(np.diff(radius) <= 0)
        ):
            raise ValueError("invalid PDF quadrature grid")
    elif grid is not None:
        raise ValueError("radius grid requires a PDF population")
