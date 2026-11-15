"""Explicit mixed CPU/GPU composition and host continuation."""

import numpy as np
import pytest

from particula import CPUTransfer, FieldSpec, FunctionProcess
from particula.dynamics import Dilution, DilutionStrategy
from particula.dynamics.tests.native_runnable_test import state

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Exercise transfers on both explicit Warp targets."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def case(stochastic=False):
    """A simple external forcing composes after native dilution."""
    aerosol = state()
    aerosol.fields.add(FieldSpec("lab.forcing", ("box",), "1"))

    def advance(box, *, time, time_step, context):
        amount = context.rng.uniform() if stochastic else time + time_step
        box.fields["lab.forcing"][:] += amount
        box.gas.concentration[:] += amount * 1e-15

    process = FunctionProcess(
        advance,
        name="forcing",
        stochastic=stochastic,
        writes=("lab.forcing", "gas.concentration"),
    )
    return (
        aerosol,
        process,
        Dilution(DilutionStrategy(0.1)) | CPUTransfer(process),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_transfer_composes_at_current_time_and_keeps_device_pointers(
    device, nested
):
    """The transfer sees prior GPU writes and changes only permitted buffers."""
    aerosol, process, pipeline = case()
    if nested:
        noop = CPUTransfer(FunctionProcess(lambda *_a, **_k: None, name="noop"))
        pipeline = Dilution(DilutionStrategy(0.1)) | (
            CPUTransfer(process) | noop
        )
    expected = aerosol.gas.concentration.copy()
    with pipeline.prepare(
        aerosol, time_step=0.5, time=3, device=device
    ) as simulation:
        target = simulation._session.gas.concentration
        pointer = target.ptr
        for time in (3.0, 3.5, 4.0):
            expected = expected * np.exp(-0.05) + (time + 0.5) * 1e-15
            simulation.step()
        assert target.ptr == pointer
        result = simulation.snapshot()
        np.testing.assert_allclose(result.gas.concentration, expected)
        np.testing.assert_allclose(result.fields["lab.forcing"], [12])
        assert simulation.current_time == 4.5
        if device != "cpu":
            with pytest.raises(ValueError, match="capture"):
                simulation.capture()
        assert not simulation.captured


def test_transfer_rng_checkpoint_preserves_future_draws(device):
    """PCG64 host words survive restart independently of device state."""
    aerosol, _, pipeline = case(stochastic=True)
    with pipeline.prepare(
        aerosol, time_step=0.5, device=device, root_seed=17
    ) as original:
        original.run(3)
        saved = original.checkpoint()
        assert saved.execution_mode == "cpu_transfer"
        assert isinstance(saved.host_rng, bytes)
        with pipeline.resume(saved) as restored:
            assert restored.checkpoint().host_rng == saved.host_rng
            original.run(4)
            restored.run(4)
            expected, actual = original.checkpoint(), restored.checkpoint()
            assert expected.host_rng == actual.host_rng
            assert expected.field_payloads == actual.field_payloads
            np.testing.assert_array_equal(
                original.snapshot().gas.concentration,
                restored.snapshot().gas.concentration,
            )


def test_rejected_cpu_callback_keeps_prior_gpu_values_and_faults(device):
    """No rejected CPU candidate reaches resident arrays; prior work remains."""
    aerosol, _, _ = case()

    def fail(box, **_):
        box.gas.concentration[:] = -1

    pipeline = Dilution(DilutionStrategy(0.1)) | CPUTransfer(
        FunctionProcess(fail, name="failure", writes=("gas.concentration",))
    )
    with pipeline.prepare(aerosol, time_step=1, device=device) as simulation:
        target = simulation._session.gas.concentration
        with pytest.raises(ValueError, match="nonnegative"):
            simulation.step()
        np.testing.assert_allclose(
            target.numpy(), aerosol.gas.concentration * np.exp(-0.1)
        )
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.step()


def test_later_box_failure_keeps_device_state_and_host_rng(device):
    """The explicit host boundary remains atomic across the complete batch."""
    from particula.tests.cpu_batch_test import state as batch_state

    aerosol = batch_state()

    def fail_later(box, *, context, **_):
        box.gas.concentration[:] += context.rng.uniform() * 1e-15
        if context.box_id == "1":
            raise ValueError("second box failed")

    process = CPUTransfer(
        FunctionProcess(
            fail_later,
            name="random",
            stochastic=True,
            writes=("gas.concentration",),
        )
    )
    with process.prepare(aerosol, time_step=1, device=device) as simulation:
        rng_before = simulation._host_context._checkpoint()
        target = simulation._session.gas.concentration
        with pytest.raises(ValueError, match="second box"):
            simulation.step()
        np.testing.assert_array_equal(target.numpy(), aerosol.gas.concentration)
        assert simulation._host_context._checkpoint() == rng_before


def test_transfer_does_not_reset_native_gpu_rng(device):
    """Host random draws are independent of existing stochastic GPU physics."""
    from particula.execution.tests.prepared_restart_capture_test import scenario

    aerosol, wall_loss, dt = scenario("wall_loss")
    aerosol.fields.add(FieldSpec("lab.random", ("box",), "1"))

    def draw(box, *, context, **_):
        box.fields["lab.random"][:] = context.rng.uniform()

    transfer = CPUTransfer(
        FunctionProcess(
            draw, name="random", stochastic=True, writes=("lab.random",)
        )
    )
    with wall_loss.prepare(
        aerosol, time_step=dt, device=device, root_seed=27
    ) as native:
        with (wall_loss | transfer).prepare(
            aerosol, time_step=dt, device=device, root_seed=27
        ) as mixed:
            native.run(4)
            mixed.run(4)
            expected, actual = native.checkpoint(), mixed.checkpoint()
            assert (
                expected.resident.rng_continuation
                == actual.resident.rng_continuation
            )
            assert expected.resident.payloads == actual.resident.payloads


@pytest.mark.parametrize("change", ["bytes", "inventory", "mode"])
def test_corrupt_host_rng_rejects_before_upload(device, monkeypatch, change):
    """A missing host stream cannot silently restart from its original seed."""
    import json
    from dataclasses import replace

    from particula.execution.prepared import PreparedSimulation

    aerosol, _, pipeline = case(stochastic=True)
    with pipeline.prepare(aerosol, time_step=1, device=device) as original:
        original.step()
        saved = original.checkpoint()
    if change == "bytes":
        saved = replace(saved, host_rng=b"bad")
    elif change == "mode":
        saved = replace(saved, execution_mode="resident")
    else:
        payload = json.loads(saved.host_rng)
        payload["streams"] = []
        saved = replace(
            saved,
            host_rng=json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("ascii"),
        )

    def no_upload(*_a, **_k):
        raise AssertionError("bad host RNG reached device upload")

    monkeypatch.setattr(PreparedSimulation, "_empty", no_upload)
    with pytest.raises(ValueError):
        pipeline.resume(saved)


def test_stale_saturation_reader_rejects_without_refresh_provider(monkeypatch):
    """Repeated execution may not expose derived saturation after gas writes."""
    from particula.execution.prepared import PreparedSimulation

    writer = CPUTransfer(
        FunctionProcess(
            lambda *_a, **_k: None, name="writer", writes=("gas.concentration",)
        )
    )
    reader = CPUTransfer(
        FunctionProcess(
            lambda *_a, **_k: None,
            name="reader",
            reads=("environment.saturation_ratio",),
        )
    )

    def no_upload(*_a, **_k):
        raise AssertionError("stale derived consumer reached upload")

    monkeypatch.setattr(PreparedSimulation, "_upload", no_upload)
    with pytest.raises(ValueError, match="refresh provider"):
        (reader | writer).prepare(state(), time_step=1, device="cpu")


def test_transfer_refreshes_saturation_before_next_consumer(device):
    """A later Python reader sees refreshed thermodynamics in the same step."""
    from particula.execution.tests.prepared_restart_capture_test import scenario

    aerosol, condensation, dt = scenario("condensation")
    aerosol.fields.add(FieldSpec("lab.saturation", ("box", "gas_species"), "1"))
    seen = []

    def heat(box, **_):
        seen.append(box.environment.saturation_ratio.copy())
        box.environment.temperature[:] += 10

    def observe(box, **_):
        box.fields["lab.saturation"][:] = box.environment.saturation_ratio

    pipeline = (
        condensation
        | CPUTransfer(
            FunctionProcess(
                heat, name="heat", writes=("environment.temperature",)
            )
        )
        | CPUTransfer(
            FunctionProcess(
                observe,
                name="observe",
                reads=("environment.saturation_ratio",),
                writes=("lab.saturation",),
            )
        )
    )
    with pipeline.prepare(aerosol, time_step=dt, device=device) as simulation:
        simulation.step()
        result = simulation.snapshot()
        np.testing.assert_allclose(
            result.fields["lab.saturation"], result.environment.saturation_ratio
        )
        assert not np.array_equal(seen[0], result.fields["lab.saturation"])


def test_transfer_rng_identity_components_are_independent_and_restart(device):
    """Delimiter-bearing identities retain separate streams through restart."""
    aerosol = state()
    for field in ("lab.first", "lab.second"):
        aerosol.fields.add(FieldSpec(field, ("box",), "1"))

    def process(extension_id, name, field):
        def draw(box, *, context, **_):
            box.fields[field][:] = context.rng.random()

        return CPUTransfer(
            FunctionProcess(
                draw,
                name=name,
                extension_id=extension_id,
                stochastic=True,
                writes=(field,),
            )
        )

    first = process("a:b", "c", "lab.first")
    second = process("a", "b:c", "lab.second")
    noop = CPUTransfer(FunctionProcess(lambda *_a, **_k: None, name="noop"))
    pipeline = first | (second | noop)
    with first.prepare(
        aerosol, time_step=1, device=device, root_seed=17
    ) as alone:
        with pipeline.prepare(
            aerosol, time_step=1, device=device, root_seed=17
        ) as combined:
            for _ in range(3):
                alone.step()
                combined.step()
                np.testing.assert_array_equal(
                    alone.snapshot().fields["lab.first"],
                    combined.snapshot().fields["lab.first"],
                )
            with pipeline.resume(combined.checkpoint()) as restored:
                combined.step()
                restored.step()
                assert (
                    combined.checkpoint().host_rng
                    == restored.checkpoint().host_rng
                )
                assert (
                    combined.checkpoint().field_payloads
                    == restored.checkpoint().field_payloads
                )
