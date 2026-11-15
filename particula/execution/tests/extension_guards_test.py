"""Device-only invalid-value checks stop dependent work before observation."""

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
from particula.dynamics import Dilution, DilutionStrategy
from particula.dynamics.tests.native_runnable_test import state

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@no_type_check
@wp.kernel
def write_float(
    values: wp.array(dtype=wp.float64),
    value: wp.float64,
    failure: wp.array(dtype=wp.int32),
):
    """An extension deliberately proposes a value outside its field contract."""
    if failure[0] == 0:
        values[wp.tid()] = value


@no_type_check
@wp.kernel
def write_int(
    values: wp.array(dtype=wp.int64),
    value: wp.int64,
    failure: wp.array(dtype=wp.int32),
):
    """Exercise a bound whose neighboring integers are not distinguishable in f64."""
    if failure[0] == 0:
        values[wp.tid()] = value


@no_type_check
@wp.kernel
def write_int32(
    values: wp.array(dtype=wp.int32),
    value: wp.int32,
    failure: wp.array(dtype=wp.int32),
):
    """Propose an invalid small integer through the int32 field path."""
    if failure[0] == 0:
        values[wp.tid()] = value


class Writer:
    """An explicit adapter for controlled field-validation failures."""

    def __init__(self, dtype="float64"):
        """Select a field dtype without retaining scientific state."""
        self.dtype = dtype

    def plan(self, process, _aerosol):
        """Freeze the deliberate invalid proposal."""
        return GPUProcessPlan(
            process.name,
            "test.invalid",
            "1",
            writes=("lab.value",),
            parameters=(("dtype", self.dtype),),
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
        )

    def prepare(self, plan, context):
        """Bind a deterministic invalid write and the shared failure flag."""
        target = context.arrays["lab.value"]
        dtype = plan.values["dtype"]
        kernel, value = {
            "float64": (write_float, wp.float64(2)),
            "int32": (write_int32, wp.int32(2)),
            "int64": (write_int, wp.int64(2**53 + 1)),
        }[dtype]
        args = [target, value, context.failure]

        def enqueue():
            wp.launch(
                kernel, dim=target.shape, inputs=args, device=context.device
            )

        return PreparedOperation(
            plan.name, enqueue, tuple(args), plan.capabilities
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", ["float64", "int32", "int64"])
@pytest.mark.parametrize("capture", [False, True])
def test_invalid_fields_stop_later_kernels_and_captured_replay(
    device, dtype, capture
):
    """Failure is resident and persistent; synchronization reports it later."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu" and capture:
        pytest.skip("CUDA capture only")
    initial = state()
    initial.fields.add(
        FieldSpec(
            "lab.value",
            ("box",),
            "1",
            dtype=dtype,
            maximum=2**53 if dtype == "int64" else 1,
        )
    )
    process = FunctionProcess(
        lambda *_a, **_k: None, name="invalid", writes=("lab.value",)
    )
    registry = ProcessRegistry()
    registry.register(FunctionProcess, Writer(dtype))
    pipeline = process | Dilution(DilutionStrategy(0.2))
    with pipeline.prepare(
        initial, time_step=1, device=device, process_registry=registry
    ) as simulation:
        if capture:
            simulation.capture()
        simulation.run(3)
        with pytest.raises(ValueError, match="invalid field"):
            simulation.synchronize()
        np.testing.assert_array_equal(
            simulation._session.gas.concentration.numpy(),
            initial.gas.concentration,
        )
        np.testing.assert_array_equal(simulation._clock.numpy(), [0])
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.step()
