"""Exact prepared continuation after an atomic disk round trip."""

import numpy as np
import pytest
import warp as wp

from particula.execution.checkpoint_io import load_checkpoint, save_checkpoint
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
    scenario,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "name",
    [
        "dilution",
        "coagulation",
        "wall_loss",
        "sequence",
        "condensation",
        "nucleation",
    ],
)
def test_saved_prepared_checkpoint_preserves_exact_future(
    tmp_path, name, device
):
    """Canonical primary, sidecar and RNG bytes produce identical continuation."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, process, dt = scenario(name)
    with process.prepare(
        initial, time_step=dt, device=device, root_seed=171
    ) as owner:
        owner.step()
        if name == "nucleation":
            # End the integral unit-event pulse before fractional demand.
            owner.update_gas(np.zeros_like(initial.gas.concentration))
        owner.run(2)
        saved = owner.checkpoint()
        path = tmp_path / "restart.pcr"
        save_checkpoint(saved, path)
        loaded = load_checkpoint(path)
        canonical_equal(saved, loaded)
        assert saved.processes == loaded.processes
        assert saved.auxiliary == loaded.auxiliary
        with process.resume(loaded, device=device) as resumed:
            owner.run(2)
            resumed.run(2)
            canonical_equal(owner.checkpoint(), resumed.checkpoint())
            assert owner.current_time == resumed.current_time


def test_changed_process_rejected_after_disk_load(tmp_path):
    """Existing process-compatibility validation remains authoritative."""
    from particula.dynamics import Dilution
    from particula.dynamics.dilution import DilutionStrategy

    initial, process, dt = scenario("dilution")
    with process.prepare(initial, time_step=dt, device="cpu") as owner:
        path = tmp_path / "restart.pcr"
        save_checkpoint(owner.checkpoint(), path)
    different = Dilution(DilutionStrategy(0.9))
    with pytest.raises(ValueError, match="configuration"):
        different.resume(load_checkpoint(path), device="cpu")


def test_edited_inspection_is_not_disk_restart_authority(tmp_path):
    """The canonical resident bytes override edited CPU inspection carriers."""
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, time_step=dt, device="cpu") as owner:
        owner.step()
        saved = owner.checkpoint()
        saved.resident.particles.masses[:] = -1
        saved.resident.gas.name[:] = ["edited"] * len(saved.resident.gas.name)
        saved.resident.environment.temperature[:] = -1
        path = tmp_path / "canonical.pcr"
        save_checkpoint(saved, path)
        loaded = load_checkpoint(path)
        canonical_equal(saved, loaded)
        with process.resume(loaded, device="cpu") as restored:
            owner.step()
            restored.step()
            canonical_equal(owner.checkpoint(), restored.checkpoint())


def test_restart_version_and_unknown_record_reject(tmp_path):
    """Only installed supported schema and explicitly listed records decode."""
    from particula.execution.archive import read_archive, write_archive

    initial, process, dt = scenario("dilution")
    path = tmp_path / "restart.pcr"
    with process.prepare(initial, time_step=dt, device="cpu") as owner:
        save_checkpoint(owner.checkpoint(), path)
    metadata, arrays, blobs = read_archive(path, kind="restart")
    metadata["version"] = 999
    write_archive(
        path, kind="restart", metadata=metadata, arrays=arrays, blobs=blobs
    )
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)
    metadata["version"] = 1
    metadata["checkpoint"]["record"] = "os.system"
    write_archive(
        path, kind="restart", metadata=metadata, arrays=arrays, blobs=blobs
    )
    with pytest.raises(ValueError, match="unsupported"):
        load_checkpoint(path)
