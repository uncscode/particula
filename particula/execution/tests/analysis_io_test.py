"""Safe numeric analysis round trips and last-good-file preservation."""

import io
import zipfile

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution.analysis_io import load_analysis, save_analysis
from particula.execution.archive import read_archive, write_archive
from particula.execution.recording import Recorder
from particula.execution.tests.recording_test import state


def recorded():
    """Retain two detached physical observations with explicit species metadata."""
    native, recorder = (
        state(),
        Recorder(("particle_masses", "gas_concentration")),
    )
    recorder.record(0, native)
    native.gas.concentration *= 0.5
    recorder.record(1, native)
    return recorder


def save(recorder, path, **options):
    """Supply explicit reproducibility metadata to an analysis save."""
    save_analysis(
        recorder,
        path,
        configuration={"gas_names": ["vapor"]},
        numerical_settings={"time_step_s": 1},
        source_identity={"revision": "test-fixture"},
        **options,
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_roundtrip_arrays_units_species_and_provenance(tmp_path, compressed):
    """Compressed and plain archives preserve every value and its dimensions."""
    recorder = recorded()
    path = tmp_path / "history.pca"
    save(recorder, path, compressed=compressed)
    result = load_analysis(path)
    for name, value in recorder.arrays().items():
        npt.assert_array_equal(result.arrays[name], value)
    assert result.metadata["science"]["gas_names"] == ["vapor"]
    assert (
        result.metadata["science"]["observables"]["particle_masses"]["unit"]
        == "kg"
    )
    assert result.metadata["source_identity"] == {"revision": "test-fixture"}
    assert result.metadata["numerical_settings"] == {"time_step_s": 1}


def test_atomic_failure_retains_last_good_archive(tmp_path, monkeypatch):
    """A failed replacement leaves the previous complete result readable."""
    path = tmp_path / "history.pca"
    save(recorded(), path)
    original = path.read_bytes()

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("particula.execution.archive.os.replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        save(recorded(), path)
    assert path.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 1
    load_analysis(path)


def test_wrong_kind_corruption_and_limits_are_explicit(tmp_path):
    """Analysis cannot masquerade as restart and corrupted bytes fail."""
    path = tmp_path / "history.pca"
    save(recorded(), path)
    with pytest.raises(ValueError, match="kind"):
        read_archive(path, kind="restart")
    with pytest.raises(ValueError, match="budget"):
        load_analysis(path, max_bytes=10)
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    name = next(key for key in parts if key.endswith(".npy"))
    parts[name] = parts[name][:-1] + bytes([parts[name][-1] ^ 1])
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="digest"):
        load_analysis(path)


def test_metadata_schema_validation_after_transport(tmp_path):
    """Valid ZIP/hash transport does not substitute for scientific validation."""
    path = tmp_path / "history.pca"
    save(recorded(), path)
    metadata, arrays, _ = read_archive(path, kind="analysis")
    metadata["science"]["observables"]["gas_concentration"]["unit"] = "wrong"
    write_archive(path, kind="analysis", metadata=metadata, arrays=arrays)
    with pytest.raises(ValueError, match="units"):
        load_analysis(path)


def test_object_arrays_and_nonfinite_json_cannot_be_written(tmp_path):
    """Loading never requires pickle, and metadata excludes nonfinite constants."""
    path = tmp_path / "unsafe.pca"
    with pytest.raises(ValueError, match="dtype"):
        write_archive(
            path,
            kind="analysis",
            metadata={},
            arrays={"x": np.array([object()])},
        )
    with pytest.raises(ValueError):
        write_archive(
            path, kind="analysis", metadata={"nan": float("nan")}, arrays={}
        )
    assert not path.exists()


def test_npy_shape_bomb_is_rejected_before_array_allocation():
    """A tiny payload cannot request an enormous array from its header."""
    from particula.execution.archive import _decode_array

    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
    )
    with pytest.raises(ValueError, match="payload length"):
        _decode_array(stream.getvalue(), {"shape": [10**12], "dtype": "<f8"})


def test_duplicate_json_keys_are_rejected(tmp_path):
    """Ambiguous metadata is rejected before interpreting array references."""
    path = tmp_path / "ambiguous.pca"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", '{"format":1,"format":2}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_analysis(path)
