"""Custom saturation readers observe preceding native updates in the same step."""

from typing import no_type_check

import numpy as np
import pytest

from particula import (
    CPUTransfer,
    FieldSpec,
    FunctionProcess,
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.dynamics import Dilution, DilutionStrategy
from particula.execution.process_extensions import GPUProcessPlan
from particula.execution.tests.prepared_restart_capture_test import scenario
from particula.gas import get_buck_vapor_pressure
from particula.util.constants import GAS_CONSTANT

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@no_type_check
@wp.kernel
def record_saturation(
    saturation: wp.array2d(dtype=wp.float64),
    recorded: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Copy the ratio visible at this operation, before end-of-step refresh."""
    box, species = wp.tid()
    if failure[0] == 0:
        recorded[box, species] = saturation[box, species]


class SaturationReader:
    """Retain a declared resident saturation reader with a registered result."""

    def plan(self, process, _aerosol):
        """Expose the same reads/writes as the ordinary Python callback."""
        return GPUProcessPlan(
            name=process.name,
            extension_id="test.saturation_reader",
            version="1",
            reads=process.reads,
            writes=process.writes,
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
        )

    def prepare(self, plan, context):
        """Bind fixed resources without observation or allocation in the step."""
        recorded = context.arrays["lab.saturation"]
        args = (
            context.arrays["environment.saturation_ratio"],
            recorded,
            context.failure,
        )

        def enqueue():
            wp.launch(
                record_saturation,
                dim=recorded.shape,
                inputs=args,
                device=context.device,
            )

        return PreparedOperation(plan.name, enqueue, args, plan.capabilities)


def reader():
    """Declare a public callback that persists exactly what it observed."""

    def observe(box, **_):
        box.fields["lab.saturation"][:] = box.environment.saturation_ratio

    return FunctionProcess(
        observe,
        name="reader",
        reads=("environment.saturation_ratio",),
        writes=("lab.saturation",),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("mode", ["transfer", "resident", "captured"])
@pytest.mark.parametrize("last_native", ["condensation", "dilution"])
def test_custom_reader_sees_current_native_thermodynamics(
    device, mode, last_native
):
    """Nonzero native transfer and dilution precede a physically checked reader."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    if mode == "captured" and device == "cpu":
        pytest.skip("native CUDA capture only")
    aerosol, condensation, dt = scenario("condensation")
    aerosol.fields.add(FieldSpec("lab.saturation", ("box", "gas_species"), "1"))
    process = reader()
    registry = ProcessRegistry()
    registry.register(FunctionProcess, SaturationReader())
    native = condensation
    if last_native == "dilution":
        native = condensation | Dilution(DilutionStrategy(0.7))
    pipeline = native | (
        CPUTransfer(process) if mode == "transfer" else process
    )
    with pipeline.prepare(
        aerosol, device=device, time_step=dt, process_registry=registry
    ) as simulation:
        if mode == "captured":
            simulation.capture()
        for _ in range(3):
            simulation.step()
            actual = simulation.snapshot()
            temperature = actual.environment.temperature
            # The fixture's first two vapor pressures are 1 and 2 Pa; the
            # mapped water lane uses Buck. This ideal-gas oracle does not read
            # the refreshed saturation array under test.
            vapor_pressure = np.column_stack(
                (
                    np.ones_like(temperature),
                    2 * np.ones_like(temperature),
                    get_buck_vapor_pressure(temperature),
                )
            )
            expected = (
                actual.gas.concentration
                * GAS_CONSTANT
                * temperature[:, None]
                / actual.gas.molar_mass
                / vapor_pressure
            )
            np.testing.assert_allclose(
                actual.fields["lab.saturation"], expected, rtol=1e-12, atol=0
            )
            assert np.any(actual.gas.concentration != aerosol.gas.concentration)
        saved = simulation.checkpoint()
        with pipeline.resume(saved, process_registry=registry) as restored:
            if mode == "captured":
                restored.capture()
            simulation.step()
            restored.step()
            assert (
                simulation.checkpoint().field_payloads
                == restored.checkpoint().field_payloads
            )


@pytest.mark.parametrize("family", ["dilution", "nucleation"])
@pytest.mark.parametrize("reader_first", [False, True])
def test_native_writer_and_custom_reader_require_refresh_provider(
    monkeypatch, family, reader_first
):
    """Native invalidation must reject before upload, including next-step reads."""
    from particula.execution.prepared import PreparedSimulation

    aerosol, native, dt = scenario(family)
    aerosol.fields.add(FieldSpec("lab.saturation", ("box", "gas_species"), "1"))
    custom = CPUTransfer(reader())
    pipeline = custom | native if reader_first else native | custom

    def no_upload(*_args, **_kwargs):
        pytest.fail("stale saturation consumer reached device upload")

    monkeypatch.setattr(PreparedSimulation, "_upload", no_upload)
    with pytest.raises(ValueError, match="refresh provider"):
        pipeline.prepare(aerosol, device="cpu", time_step=dt)
