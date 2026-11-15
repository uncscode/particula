"""Registered particle fields follow actual native GPU population events."""

import numpy as np
import pytest

from particula import FieldSpec
from particula.execution.tests.prepared_restart_capture_test import scenario

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Separate Warp CPU from real CUDA evidence."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def add_fields(initial):
    """Give two independent tracers nonuniform values and birth defaults."""
    counts = initial.particles.concentration
    for name, axes, scale in (
        ("lab.scalar", ("box", "particle"), 1),
        ("lab.vector", ("box", "particle", "particle_species"), 3),
    ):
        values = np.arange(counts.size, dtype=float).reshape(counts.shape) + 1
        values[counts == 0] = 0
        if len(axes) == 3:
            values = np.repeat(
                values[..., None], initial.particles.n_species, axis=2
            )
        initial.fields.add(
            FieldSpec(
                name,
                axes,
                "kg",
                initial_value=2 * scale,
                particle_policy="additive",
            ),
            values * scale,
        )


def inventory(state, name):
    """Represented per-box inventory, independent of slot layout."""
    counts = state.particles.concentration
    values = state.fields[name]
    return np.sum(
        values * (counts if values.ndim == 2 else counts[..., None]), axis=1
    )


@pytest.mark.parametrize(
    "family",
    [
        "dilution",
        "coagulation",
        "wall_loss",
        "condensation",
        "nucleation",
        "nucleation_resampling",
        "sequence",
    ],
)
def test_native_particle_inventory(device, family):
    """Mergers/remapping preserve inventory; sources/sinks close the budget."""
    initial, process, dt = scenario(family)
    add_fields(initial)
    with process.prepare(
        initial, device=device, time_step=dt, root_seed=47
    ) as sim:
        sim.step()
        result = sim.snapshot()
        for name in initial.fields.particle_names():
            ledger = result.fields.inventory_changes(name)
            np.testing.assert_allclose(
                inventory(result, name),
                inventory(initial, name) + ledger["source"] - ledger["removal"],
                rtol=2e-14,
            )
            assert np.all(
                result.fields[name][result.particles.concentration == 0] == 0
            )
        if family == "nucleation_resampling":
            # Three unit-weight donors become two weight-1.5 survivors;
            # the middle donor is split equally across their intervals.
            np.testing.assert_allclose(
                result.fields["lab.scalar"], [[4 / 3, 8 / 3, 2, 2]]
            )
        if family == "coagulation":
            assert np.count_nonzero(result.particles.concentration) < 64
        saved = sim.checkpoint()
        with process.resume(saved) as resumed:
            assert resumed.checkpoint().field_payloads == saved.field_payloads


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_passive_birth(device, dtype):
    """Passive integer fields initialize only actual newborn slots."""
    from particula.execution.tests.prepared_test import nucleation_case

    initial, process = nucleation_case(resampling=False)
    dt = 1.0
    values = np.where(initial.particles.concentration > 0, 9, 0).astype(dtype)
    initial.fields.add(
        FieldSpec(
            "lab.label", ("box", "particle"), "1", dtype=dtype, initial_value=4
        ),
        values,
    )
    with process.prepare(initial, device=device, time_step=dt) as sim:
        sim.step()
        result = sim.snapshot()
    born = (initial.particles.concentration == 0) & (
        result.particles.concentration > 0
    )
    assert np.any(born)
    np.testing.assert_array_equal(result.fields["lab.label"][born], 4)


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["sequence", "nucleation_resampling"])
def test_particle_capture_restart(family, monkeypatch):
    """Recording is inert and captured field/ledger continuation is exact."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario(family)
    add_fields(initial)
    with process.prepare(initial, time_step=dt, root_seed=47) as ordinary:
        with process.prepare(initial, time_step=dt, root_seed=47) as captured:
            before = captured.checkpoint()
            captured.capture()
            assert captured.checkpoint().field_payloads == before.field_payloads
            for step in range(3):
                for sim in (ordinary, captured):
                    if step == 1 and family.startswith("nucleation"):
                        sim.update_gas(np.zeros_like(initial.gas.concentration))
                    with monkeypatch.context() as patch:

                        def forbidden(*_, **__):
                            pytest.fail("resident step allocated or observed")

                        for name in (
                            "empty",
                            "zeros",
                            "empty_like",
                            "zeros_like",
                            "synchronize",
                            "synchronize_device",
                            "synchronize_stream",
                        ):
                            patch.setattr(wp, name, forbidden)
                        patch.setattr(wp.array, "numpy", forbidden)
                        sim.step()
                left, right = ordinary.checkpoint(), captured.checkpoint()
                assert left.field_payloads == right.field_payloads
                assert left.resident.payloads == right.resident.payloads
            with process.resume(right) as resumed:
                resumed.capture()
                resumed.step()
                captured.step()
                assert (
                    resumed.checkpoint().field_payloads
                    == captured.checkpoint().field_payloads
                )


@pytest.mark.parametrize("family", ["coagulation", "nucleation_resampling"])
def test_passive_population_rejects_before_upload(family, monkeypatch):
    """Undefined passive mergers and resampling reject without GPU allocation."""
    from particula.execution.prepared import PreparedSimulation

    initial, process, dt = scenario(family)
    initial.fields.add(FieldSpec("lab.label", ("box", "particle"), "1"))
    monkeypatch.setattr(
        PreparedSimulation,
        "_upload",
        lambda *_, **__: pytest.fail("uploaded invalid plan"),
    )
    with pytest.raises(ValueError, match="additive"):
        process.prepare(initial, device="cpu", time_step=dt)


class ParticleWriter:
    """Bind the shared test writer to a declared particle array."""

    def __init__(self, name, value):
        """Select a destination and proposed scalar value."""
        self.name, self.value = name, value

    def plan(self, process, _aerosol):
        """Declare the particle array mutation explicitly."""
        from particula import GPUProcessPlan, ProcessCapabilities

        return GPUProcessPlan(
            process.name,
            "test.particle",
            "1",
            writes=(self.name,),
            capabilities=ProcessCapabilities(True, True, True),
        )

    def prepare(self, plan, context):
        """Retain the launch arguments and shared failure guard."""
        from particula import PreparedOperation
        from particula.execution.tests.extension_guards_test import write_float

        target = context.arrays[self.name].flatten()
        args = (target, wp.float64(self.value), context.failure)

        def enqueue():
            wp.launch(
                write_float, dim=target.size, inputs=args, device=context.device
            )

        return PreparedOperation(plan.name, enqueue, args, plan.capabilities)


@pytest.mark.parametrize("change", ["activity", "bound"])
def test_custom_particle_contract_faults_before_later_work(device, change):
    """Custom value freedom cannot retire slots or break active field bounds."""
    from particula import FunctionProcess, ProcessRegistry
    from particula.dynamics import Dilution, DilutionStrategy

    initial, _, _ = scenario("dilution")
    initial.fields.add(
        FieldSpec(
            "lab.value",
            ("box", "particle"),
            "1",
            initial_value=1,
            minimum=1,
            maximum=2,
        )
    )
    name = "particles.concentration" if change == "activity" else "lab.value"
    process = FunctionProcess(
        lambda *_a, **_k: None, name="writer", writes=(name,)
    )
    registry = ProcessRegistry()
    registry.register(FunctionProcess, ParticleWriter(name, 0))
    pipeline = process | Dilution(DilutionStrategy(0.2))
    with pipeline.prepare(
        initial, device=device, time_step=1, process_registry=registry
    ) as sim:
        sim.run(2)
        with pytest.raises(ValueError, match="contract|invalid field"):
            sim.synchronize()
        np.testing.assert_array_equal(
            sim._session.gas.concentration.numpy(), initial.gas.concentration
        )


def test_positive_particle_bounds_ignore_inactive_slots(device):
    """Inactive zeros are valid even when active particles require positive data."""
    from particula import FunctionProcess, ProcessRegistry

    initial, _, dt = scenario("nucleation")
    values = np.where(initial.particles.concentration > 0, 1.0, 0.0)
    initial.fields.add(
        FieldSpec(
            "lab.value", ("box", "particle"), "1", initial_value=1, minimum=1
        ),
        values,
    )
    # Write only a non-particle field; all particle bounds are still checked.
    process = FunctionProcess(
        lambda *_a, **_k: None, name="gas", writes=("gas.concentration",)
    )
    registry = ProcessRegistry()
    registry.register(FunctionProcess, ParticleWriter("gas.concentration", 0))
    with process.prepare(
        initial, device=device, time_step=dt, process_registry=registry
    ) as sim:
        sim.step()
        np.testing.assert_array_equal(
            sim.snapshot().fields["lab.value"], values
        )
