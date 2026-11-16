"""Exact scaled-state GPU continuation, disk safety and device admission."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp

from particula.execution.archive import read_archive, write_archive
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.gpu_restart import (
    checkpoint_gpu_condensation,
    load_gpu_checkpoint,
    resume_gpu_condensation,
    save_gpu_checkpoint,
)
from particula.execution.recording import Recorder, run_until
from particula.execution.tests.condensation_ode_test import case


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Use actual CUDA for exact-device restart evidence."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def assert_same(left, right):
    """Compare continuation authority including scaled tolerance roundoff bits."""
    for name in ("_state", "_atol", "_rtol"):
        npt.assert_array_equal(
            getattr(left, name).numpy(), getattr(right, name).numpy()
        )
    assert left.time == right.time
    assert left.next_step == right.next_step
    assert left.last_error == right.last_error
    assert left.statistics == right.statistics
    assert left.control_readbacks == right.control_readbacks


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_disk_roundtrip_preserves_adaptive_future_with_rejections(
    device, tmp_path, method
):
    """Saved next-step/controller state gives identical future accepted trials."""
    initial, strategy = case(3)
    initial.gas.concentration[:, 1] *= [0.1, 1, 10]
    owner = GPUCondensationODE(
        initial,
        strategy,
        device=device,
        method=method,
        rtol=1e-8,
        particle_atol=3.14159e-27,
        gas_atol=2.71828e-17,
        first_step=10,
        max_step=10,
    )
    history = Recorder()
    run_until(owner, 0.2, output_times=[0, 0.1], recorder=history)
    assert owner.statistics.rejected > 0
    checkpoint = checkpoint_gpu_condensation(owner)
    path = tmp_path / "adaptive.pcr"
    save_gpu_checkpoint(checkpoint, path)
    loaded = load_gpu_checkpoint(path)
    restored = resume_gpu_condensation(loaded, strategy, device=device)
    assert_same(owner, restored)
    for end in (0.3, 0.7, 1.0):
        run_until(owner, end)
        run_until(restored, end, recorder=history)
        assert_same(owner, restored)
        npt.assert_array_equal(
            owner.snapshot().particles.masses,
            restored.snapshot().particles.masses,
        )
    checkpoint.state[:] = -1
    assert np.all(restored._state.numpy() >= 0)


def test_checkpoint_after_rejected_trial_budget_is_reusable(device, tmp_path):
    """Trial scratch is not required to continue the accepted state exactly."""
    initial, strategy = case()
    owner = GPUCondensationODE(
        initial,
        strategy,
        device=device,
        first_step=100,
        max_step=100,
        max_trials=1,
    )
    with pytest.raises(RuntimeError, match="trial count"):
        owner.step(100)
    path = tmp_path / "rejected.pcr"
    save_gpu_checkpoint(checkpoint_gpu_condensation(owner), path)
    restored = resume_gpu_condensation(
        load_gpu_checkpoint(path), strategy, device=device
    )
    assert_same(owner, restored)
    owner.max_trials = restored.max_trials = 100
    run_until(owner, 0.2)
    run_until(restored, 0.2)
    assert_same(owner, restored)


def test_incompatible_runtime_physics_and_device_reject_before_upload(
    device, monkeypatch
):
    """Checkpoint admission never falls back or partially creates an owner."""
    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device=device)
    saved = checkpoint_gpu_condensation(owner)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid restart must reject before allocation")

    monkeypatch.setattr(GPUCondensationODE, "__init__", forbidden)
    for malformed in (
        replace(saved, runtime=("wrong", *saved.runtime[1:])),
        replace(saved, configuration=("different",)),
        replace(
            saved, device=(saved.device[0], "different", *saved.device[2:])
        ),
        replace(saved, state=np.full_like(saved.state, -1)),
        replace(saved, state=saved.state[:-1]),
        replace(saved, next_step=0),
        replace(saved, last_error=2),
        replace(saved, control_readbacks=-1),
    ):
        with pytest.raises(ValueError):
            resume_gpu_condensation(malformed, strategy, device=device)


def test_loader_does_not_create_device_arrays(device, tmp_path, monkeypatch):
    """Loading validates host data; device selection is an explicit later action."""
    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device=device)
    path = tmp_path / "host-only.pcr"
    save_gpu_checkpoint(checkpoint_gpu_condensation(owner), path)

    def forbidden(*args, **kwargs):
        raise AssertionError("load must not allocate device arrays")

    monkeypatch.setattr(wp, "array", forbidden)
    loaded = load_gpu_checkpoint(path)
    assert loaded.time == 0


def test_malformed_file_and_failed_save_preserve_previous_checkpoint(
    tmp_path, monkeypatch
):
    """A bad replacement cannot destroy the last usable accepted state."""
    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device="cpu")
    saved = checkpoint_gpu_condensation(owner)
    path = tmp_path / "safe.pcr"
    save_gpu_checkpoint(saved, path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_gpu_checkpoint(replace(saved, max_trials=0), path)
    assert path.read_bytes() == original
    metadata, arrays, blobs = read_archive(path, kind="restart")
    metadata["checkpoint"]["record"] = "os.system"
    write_archive(
        path, kind="restart", metadata=metadata, arrays=arrays, blobs=blobs
    )
    with pytest.raises(ValueError, match="malformed"):
        load_gpu_checkpoint(path)


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_active_capture_rejects_checkpoint_and_resume(method):
    """No adaptive checkpoint transfer or restore is admitted inside capture."""
    if not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, strategy = case()
    owner = GPUCondensationODE(
        initial, strategy, device="cuda:0", method=method
    )
    saved = checkpoint_gpu_condensation(owner)
    with wp.ScopedCapture(device="cuda:0"):
        with pytest.raises(RuntimeError, match="capture"):
            checkpoint_gpu_condensation(owner)
        with pytest.raises(RuntimeError, match="capture"):
            resume_gpu_condensation(saved, strategy, device="cuda:0")
    assert owner.time == 0
