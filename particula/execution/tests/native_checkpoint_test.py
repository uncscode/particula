"""Schema-v5 native population authority and exact-device continuation."""

from dataclasses import replace

import numpy as np
import pytest

from particula.execution import Backend, Device
from particula.execution.checkpoint import restart_resident_session
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.gpu_session import ResidentStepGuard
from particula.execution.tests.native_resident_test import session_for

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Exercise installed Warp CPU and native CUDA when available."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_native_checkpoint_restores_canonical_bytes_and_semantics(device, kind):
    """Inspection edits never alter canonical count, volume or gas authority."""
    _, session = session_for(device, kind)
    registry = GPUResourceRegistry(session)
    registry.acquire_condensation()
    registry.acquire_nucleation()
    guard = ResidentStepGuard(session, registry)
    token = guard.begin_step(0.25)
    guard.complete_step(token)
    checkpoint = session.checkpoint(registry, guard)
    assert checkpoint.schema_version == 5
    assert checkpoint.distribution_type == kind
    assert checkpoint.particles.distribution_type == kind
    assert checkpoint.dimensions.n_species == 2
    assert checkpoint.dimensions.n_gas_species == 3
    checkpoint.particles.concentration[:] = 999
    checkpoint.particles.distribution_type = None
    checkpoint.gas.concentration[:] = 999
    restored, restored_registry, restored_guard = restart_resident_session(
        checkpoint, session.metadata.device
    )
    assert restored.metadata.distribution_type == kind
    assert restored.dimensions == session.dimensions
    assert restored.metadata.gas_names == session.metadata.gas_names
    assert restored_guard.completed_steps == 1
    assert restored_guard.simulated_time == 0.25
    for family, roles in (
        (
            "particles",
            ("masses", "concentration", "charge", "density", "volume"),
        ),
        (
            "gas",
            ("molar_mass", "concentration", "partitioning", "vapor_pressure"),
        ),
        ("environment", ("temperature", "pressure", "saturation_ratio")),
    ):
        for role in roles:
            source = getattr(getattr(session, family), role)
            target = getattr(getattr(restored, family), role)
            assert source is not target
            np.testing.assert_array_equal(target.numpy(), source.numpy())
    assert restored_registry.acquire_condensation().scratch_buffers is not (
        registry.acquire_condensation().scratch_buffers
    )
    # A second canonical snapshot must retain every acquired sidecar byte too.
    again = restored.checkpoint(restored_registry, restored_guard)
    assert again.payloads == checkpoint.payloads
    assert again.rng_continuation == checkpoint.rng_continuation


@pytest.mark.parametrize(
    "failure", ["kind", "pdf", "gas_width", "missing_width", "device"]
)
def test_native_checkpoint_rejects_semantic_drift_before_setup(
    device, failure, monkeypatch
):
    """Malformed semantic metadata cannot trigger upload or inferred mapping."""
    import particula.execution.checkpoint as module

    _, session = session_for(device)
    registry = GPUResourceRegistry(session)
    checkpoint = session.checkpoint(
        registry, ResidentStepGuard(session, registry)
    )
    target = session.metadata.device
    if failure in ("kind", "pdf"):
        checkpoint = replace(
            checkpoint,
            distribution_type=None if failure == "kind" else "continuous_pdf",
        )
    elif failure in ("gas_width", "missing_width"):
        checkpoint = replace(
            checkpoint,
            dimensions=replace(
                checkpoint.dimensions,
                n_gas_species=4 if failure == "gas_width" else None,
            ),
        )
    else:
        target = Device(Backend.WARP, "cuda:0" if device == "cpu" else "cpu")
    monkeypatch.setattr(
        module,
        "setup_resident_session",
        lambda *a, **k: pytest.fail("invalid metadata reached setup"),
    )
    with pytest.raises(ValueError):
        restart_resident_session(checkpoint, target)


def test_native_stochastic_restart_continues_actual_adapters(
    device, monkeypatch
):
    """Native coagulation and wall loss resume current words without reseeding."""
    import particula.execution.gpu_resources as resources
    from particula.execution.tests.checkpoint_test import (
        _dispatch_real_processes,
    )

    _, session = session_for(device)
    session.particles.concentration.assign(np.ones((1, 2)))
    registry = GPUResourceRegistry(session)
    guard = ResidentStepGuard(session, registry)
    coagulation, wall = _dispatch_real_processes(session, registry)
    checkpoint = session.checkpoint(registry, guard)
    original_words = [
        coagulation.rng_states.numpy().copy(),
        wall.rng_states.numpy().copy(),
    ]
    with monkeypatch.context() as patch:
        for method in ("initialize", "initialize_process"):
            patch.setattr(
                resources.StreamRegistry,
                method,
                lambda *a, **k: pytest.fail(
                    "restart reseeded a published stream"
                ),
            )
        restored, restored_registry, _ = restart_resident_session(
            checkpoint, session.metadata.device
        )
        _dispatch_real_processes(session, registry)
        restored_coagulation, restored_wall = _dispatch_real_processes(
            restored, restored_registry
        )
    for role in ("masses", "concentration", "charge", "volume"):
        np.testing.assert_array_equal(
            getattr(restored.particles, role).numpy(),
            getattr(session.particles, role).numpy(),
        )
    for source, target, old in zip(
        (coagulation, wall),
        (restored_coagulation, restored_wall),
        original_words,
        strict=True,
    ):
        assert source.rng_states is not target.rng_states
        np.testing.assert_array_equal(
            source.rng_states.numpy(), target.rng_states.numpy()
        )
        assert np.any(source.rng_states.numpy() != old)
