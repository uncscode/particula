"""Resident user adapters run through narrow, explicit preparation contracts."""

from typing import no_type_check

import numpy as np
import pytest

from particula import FieldSpec, FunctionProcess
from particula.dynamics.tests.native_runnable_test import state
from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.execution.process_extensions import GPUProcessPlan

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@no_type_check
@wp.kernel
def integrate_time(
    values: wp.array(dtype=wp.float64),
    clock: wp.array(dtype=wp.float64),
    dt: wp.float64,
    failure: wp.array(dtype=wp.int32),
):
    """Left endpoint integral with a closed-form arithmetic-series oracle."""
    box = wp.tid()
    if failure[0] == 0:
        values[box] += clock[0] * dt


class Adapter:
    """A third-party implementation needs no private simulation hooks."""

    def plan(self, process, aerosol):
        """Declare one field without retaining mutable process state."""
        assert "lab.integral" in aerosol.fields.specs
        return GPUProcessPlan(
            name=process.name,
            extension_id="test.integral",
            version="1",
            writes=("lab.integral",),
            capabilities=ProcessCapabilities(resident=True, capture=True),
        )

    def prepare(self, plan, context):
        """Capture only explicit resources for the kernel launch."""
        assert set(context.arrays) == {"lab.integral"}
        assert context.logical_box_ids == ("chamber",)
        target = context.arrays["lab.integral"]
        args = [
            target,
            context.clock,
            wp.float64(context.time_step),
            context.failure,
        ]

        def enqueue():
            wp.launch(
                integrate_time,
                dim=target.shape,
                inputs=args,
                device=context.device,
            )

        return PreparedOperation(
            plan.name, enqueue, tuple(args), plan.capabilities
        )


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """CUDA rows require real CUDA, with an explicit CPU regression row."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def make_case():
    """Use one public function process and its explicit resident adapter."""
    initial = state()
    initial.fields.add(FieldSpec("lab.integral", ("box",), "s^2"))
    process = FunctionProcess(
        lambda *_a, **_k: None, name="integral", writes=("lab.integral",)
    )
    registry = ProcessRegistry()
    registry.register(FunctionProcess, Adapter())
    return initial, process, registry


def test_registered_adapter_time_and_detached_fields(device, monkeypatch):
    """Resident launches retain buffers and perform no transfer or allocation."""
    initial, process, registry = make_case()
    with process.prepare(
        initial,
        time_step=0.5,
        device=device,
        logical_box_ids=("chamber",),
        process_registry=registry,
    ) as simulation:
        initial.fields["lab.integral"][:] = 100
        target = simulation._extension_fields.arrays["lab.integral"]
        pointer = target.ptr
        with monkeypatch.context() as patch:

            def forbidden(*_a, **_k):
                raise AssertionError("resident step allocated or read back")

            patch.setattr(wp, "zeros", forbidden)
            patch.setattr(wp, "empty", forbidden)
            patch.setattr(wp.array, "numpy", forbidden)
            simulation.run(4)
        assert target.ptr == pointer
        result = simulation.snapshot()
        np.testing.assert_allclose(result.fields["lab.integral"], [1.5])
        result.fields["lab.integral"][:] = 55
        np.testing.assert_allclose(
            simulation.snapshot().fields["lab.integral"], [1.5]
        )


def test_registered_capture_is_inert_and_uses_evolving_clock(device):
    """Recorded replay uses device time rather than a frozen host scalar."""
    if device == "cpu":
        pytest.skip("native CUDA capture only")
    initial, process, registry = make_case()
    with process.prepare(
        initial,
        time_step=0.5,
        device=device,
        logical_box_ids=("chamber",),
        process_registry=registry,
    ) as simulation:
        simulation.capture()
        np.testing.assert_array_equal(
            simulation.snapshot().fields["lab.integral"], [0]
        )
        simulation.run(4)
        np.testing.assert_allclose(
            simulation.snapshot().fields["lab.integral"], [1.5]
        )


@pytest.mark.parametrize(
    "parameters", [(("x", []),), (("x", np.ones(1)),), (("x", float("nan")),)]
)
def test_plan_rejects_mutable_or_nonfinite_configuration(parameters):
    """A frozen plan cannot hide changing parameters behind an array reference."""
    with pytest.raises(ValueError, match="immutable"):
        GPUProcessPlan("p", "test.p", "1", parameters=parameters)


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing", "unavailable"),
        ("metadata", "immutable"),
        ("mutation", "mutate"),
        ("duplicate", "one occurrence"),
        ("unregistered", "no prepared GPU translation"),
    ],
)
def test_invalid_adapters_reject_before_upload(monkeypatch, case, match):
    """Planning errors cannot partially upload a simulation."""
    from dataclasses import replace

    from particula.execution.prepared import PreparedSimulation

    initial, process, _ = make_case()

    class InvalidAdapter(Adapter):
        def plan(self, process, aerosol):
            """Deliberately break one independently checked admission rule."""
            plan = super().plan(process, aerosol)
            if case == "missing":
                return replace(plan, writes=("lab.missing",))
            if case == "metadata":
                return replace(plan, writes=("particles.density",))
            if case == "mutation":
                aerosol.gas.concentration[:] = 0
            return plan

    registry = ProcessRegistry()
    if case != "unregistered":
        registry.register(FunctionProcess, InvalidAdapter())

    def no_upload(*_a, **_k):
        raise AssertionError("invalid plan reached device upload")

    monkeypatch.setattr(PreparedSimulation, "_upload", no_upload)
    pipeline = process | process if case == "duplicate" else process
    with pytest.raises(ValueError, match=match):
        pipeline.prepare(
            initial, time_step=1, device="cpu", process_registry=registry
        )


def test_registry_config_is_detached_from_process_after_preparation(device):
    """Changing the caller's process cannot alter the admitted device operation."""
    initial, process, registry = make_case()
    with process.prepare(
        initial,
        time_step=0.5,
        device=device,
        logical_box_ids=("chamber",),
        process_registry=registry,
    ) as simulation:
        process.name = "changed"
        process.writes = ("gas.concentration",)
        simulation.run(3)
        np.testing.assert_allclose(
            simulation.snapshot().fields["lab.integral"], [0.75]
        )


def test_nonzero_initial_time_is_shared_by_host_and_device(device):
    """Absolute time starts at the requested origin; elapsed time stays zero."""
    initial, process, registry = make_case()
    with process.prepare(
        initial,
        time_step=0.5,
        time=10,
        device=device,
        logical_box_ids=("chamber",),
        process_registry=registry,
    ) as simulation:
        assert simulation.current_time == 10
        assert simulation.simulated_time == 0
        simulation.run(4)
        assert simulation.current_time == 12
        assert simulation.simulated_time == 2
        np.testing.assert_allclose(
            simulation.snapshot().fields["lab.integral"], [21.5]
        )
    assert simulation.current_time == 12
