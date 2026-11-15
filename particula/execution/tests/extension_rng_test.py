"""Managed resident random streams follow logical identity and exact restart."""

from typing import no_type_check

import numpy as np
import pytest

from particula import (
    FieldSpec,
    FunctionProcess,
    GPUProcessPlan,
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.tests.cpu_batch_test import state

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@no_type_check
@wp.kernel
def draw(
    values: wp.array(dtype=wp.float64),
    words: wp.array(dtype=wp.uint32),
    failure: wp.array(dtype=wp.int32),
):
    """Give one thread exclusive ownership of each logical box's random word."""
    box = wp.tid()
    if failure[0] == 0:
        word = words[box]
        values[box] += wp.float64(wp.randf(word))
        words[box] = word


class NoiseAdapter:
    """Public adapter using only the context's managed random array."""

    def plan(self, process, _aerosol):
        """Declare a per-instance stochastic field update."""
        return GPUProcessPlan(
            process.name,
            "test.noise",
            "1",
            writes=("lab." + process.name,),
            stochastic=True,
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
        )

    def prepare(self, plan, context):
        """Retain the random words without initializing them in the step."""
        target = context.arrays[plan.writes[0]]
        args = [target, context.rng, context.failure]

        def enqueue():
            wp.launch(
                draw, dim=target.shape, inputs=args, device=context.device
            )

        return PreparedOperation(
            plan.name, enqueue, tuple(args), plan.capabilities
        )


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Separate explicit Warp CPU evidence from actual CUDA coverage."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def case():
    """Two process instances share an adapter but never a random stream."""
    aerosol = state()
    processes = []
    for name in ("first", "second"):
        aerosol.fields.add(FieldSpec("lab." + name, ("box",), "1"))
        processes.append(
            FunctionProcess(
                lambda *_a, **_k: None,
                name=name,
                stochastic=True,
                writes=("lab." + name,),
            )
        )
    registry = ProcessRegistry()
    registry.register(FunctionProcess, NoiseAdapter())
    return aerosol, processes, registry


def test_streams_follow_box_identity_and_not_process_order(device, monkeypatch):
    """Reordering lanes/processes preserves each independently named stream."""
    aerosol, (first, second), registry = case()
    with (first | second).prepare(
        aerosol,
        time_step=1,
        device=device,
        root_seed=19,
        logical_box_ids=("A", "B"),
        process_registry=registry,
    ) as left:
        with (second | first).prepare(
            aerosol,
            time_step=1,
            device=device,
            root_seed=19,
            logical_box_ids=("B", "A"),
            process_registry=registry,
        ) as right:
            import particula.execution.extension_rng as implementation

            def no_reseed(*_a, **_k):
                raise AssertionError("resident stepping reseeded RNG")

            monkeypatch.setattr(implementation, "initial_words", no_reseed)
            left.run(7)
            right.run(7)
            a, b = left.snapshot(), right.snapshot()
            for name in ("lab.first", "lab.second"):
                np.testing.assert_array_equal(
                    a.fields[name], b.fields[name][::-1]
                )
            assert not np.array_equal(
                a.fields["lab.first"], a.fields["lab.second"]
            )


def test_rng_capture_is_inert_and_restart_keeps_exact_future_words(device):
    """Capture and restart preserve scientific values and random continuation."""
    aerosol, (process, _), registry = case()
    with process.prepare(
        aerosol, time_step=0.5, device=device, process_registry=registry
    ) as original:
        before = original.checkpoint()
        if device != "cpu":
            original.capture()
        assert original.checkpoint().extension_state == before.extension_state
        original.run(3)
        saved = original.checkpoint()
        with process.resume(saved, process_registry=registry) as restored:
            if device != "cpu":
                restored.capture()
            original.run(4)
            restored.run(4)
            expected, actual = original.checkpoint(), restored.checkpoint()
            assert expected.extension_state == actual.extension_state
            assert expected.field_payloads == actual.field_payloads
            assert expected.extension_state != before.extension_state


def test_zero_duration_preserves_custom_rng_and_fields(device):
    """Zero-duration execution validates but neither draws nor advances fields."""
    aerosol, (process, _), registry = case()
    with process.prepare(
        aerosol, time_step=0, device=device, process_registry=registry
    ) as simulation:
        before = simulation.checkpoint()
        simulation.run(3)
        after = simulation.checkpoint()
        assert before.extension_state == after.extension_state
        assert before.field_payloads == after.field_payloads


def test_missing_random_words_reject_before_restart_allocation(
    device, monkeypatch
):
    """Removing saved RNG bytes cannot silently produce a reseeded simulation."""
    from dataclasses import replace

    from particula.execution.prepared import PreparedSimulation

    aerosol, (process, _), registry = case()
    with process.prepare(
        aerosol, time_step=1, device=device, process_registry=registry
    ) as simulation:
        simulation.run(2)
        saved = replace(simulation.checkpoint(), extension_state=())

    def no_allocate(*_a, **_k):
        raise AssertionError("missing random continuation reached allocation")

    monkeypatch.setattr(PreparedSimulation, "_empty", no_allocate)
    with pytest.raises(ValueError, match="inventory"):
        process.resume(saved, process_registry=registry)
