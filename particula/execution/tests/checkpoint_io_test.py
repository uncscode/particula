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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["clock", "stochastic"])
def test_registered_extension_disk_continuation(tmp_path, device, kind):
    """Data-only plans preserve custom fields, clock, persistent state and RNG."""
    from particula.execution.tests.extension_rng_test import case as noise_case
    from particula.execution.tests.process_extensions_test import make_case

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    if kind == "clock":
        initial, process, registry = make_case()
        ids = ("chamber",)
    else:
        initial, processes, registry = noise_case()
        process = processes[0] | processes[1]
        ids = ("A", "B")
    with process.prepare(
        initial,
        time_step=0.25,
        time=7,
        device=device,
        root_seed=19,
        logical_box_ids=ids,
        process_registry=registry,
    ) as original:
        original.run(3)
        saved = original.checkpoint()
        path = tmp_path / "extension.pcr"
        save_checkpoint(saved, path)
        loaded = load_checkpoint(path)
        assert loaded.processes == saved.processes
        assert loaded.field_payloads == saved.field_payloads
        assert loaded.extension_state == saved.extension_state
        assert loaded.clock == saved.clock
        with process.resume(
            loaded, device=device, process_registry=registry
        ) as restored:
            original.run(5)
            restored.run(5)
            expected, actual = original.checkpoint(), restored.checkpoint()
            canonical_equal(expected, actual)
            assert expected.field_payloads == actual.field_payloads
            assert expected.extension_state == actual.extension_state
            assert expected.clock == actual.clock
            assert expected.initial_time == actual.initial_time == 7
        with pytest.raises(
            ValueError, match="translation|registered|registry|adapter"
        ):
            process.resume(loaded, device=device)


def test_all_archive_writers_reject_active_cuda_capture(tmp_path):
    """Saving existing host data must not write files inside native capture."""
    from particula.execution.analysis_io import save_analysis
    from particula.execution.condensation_ode import CondensationODE
    from particula.execution.cpu_restart import (
        checkpoint_condensation,
        save_cpu_checkpoint,
    )
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.gpu_restart import (
        checkpoint_gpu_condensation,
        save_gpu_checkpoint,
    )
    from particula.execution.recording import Recorder
    from particula.execution.tests.condensation_ode_test import case

    if not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, time_step=dt, device="cuda:0") as owner:
        prepared = owner.checkpoint()
    initial, strategy = case()
    cpu = checkpoint_condensation(CondensationODE(initial, strategy))
    gpu = checkpoint_gpu_condensation(
        GPUCondensationODE(initial, strategy, device="cuda:0")
    )
    history = Recorder()
    history.record(0, initial)
    destination = tmp_path / "last-good"
    destination.write_bytes(b"previous complete file")
    with wp.ScopedCapture(device="cuda:0"):
        for save, saved in (
            (save_checkpoint, prepared),
            (save_cpu_checkpoint, cpu),
            (save_gpu_checkpoint, gpu),
        ):
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                save(saved, destination)
        with pytest.raises(RuntimeError, match="during CUDA capture"):
            save_analysis(
                history,
                destination,
                configuration={},
                numerical_settings={},
                source_identity={},
            )
    assert destination.read_bytes() == b"previous complete file"
    assert list(tmp_path.iterdir()) == [destination]


def test_host_archive_does_not_initialize_warp(tmp_path, monkeypatch):
    """The capture guard does not initialize a runtime for native CPU output."""
    from warp._src import context

    from particula.execution.archive import read_archive, write_archive

    monkeypatch.setattr(context, "runtime", None)
    destination = tmp_path / "cpu.analysis"
    write_archive(destination, kind="analysis", metadata={}, arrays={})
    assert context.runtime is None
    assert read_archive(destination, kind="analysis") == ({}, {}, {})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cpu_transfer_disk_restart_preserves_host_rng(tmp_path, device):
    """Explicit mixed execution retains PCG64 words and custom field bytes."""
    from particula.execution.tests.cpu_transfer_test import case

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, _, process = case(stochastic=True)
    with process.prepare(
        initial, time_step=0.5, time=3, device=device, root_seed=17
    ) as original:
        original.run(3)
        saved = original.checkpoint()
        path = tmp_path / "mixed.restart"
        save_checkpoint(saved, path)
        loaded = load_checkpoint(path)
        assert loaded.host_rng == saved.host_rng
        with process.resume(loaded, device=device) as restored:
            original.run(4)
            restored.run(4)
            expected, actual = original.checkpoint(), restored.checkpoint()
            canonical_equal(expected, actual)
            assert expected.host_rng == actual.host_rng
            assert expected.field_payloads == actual.field_payloads
            assert original.current_time == restored.current_time == 6.5
