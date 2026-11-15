"""Field descriptor admission and independent ownership."""

import numpy as np
import pytest
from particula.extension_fields import ExtensionFields, FieldSpec


def test_fields_copy_values_and_preserve_descriptors():
    """Input arrays and detached copies cannot alias live field storage."""
    fields = ExtensionFields({"box": 2})
    spec = FieldSpec("lab.exposure", ("box",), "s")
    values = np.array([2.0, 3.0])
    fields.add(spec, values)
    detached = fields.copy()
    values[:] = 0
    detached[spec.name][:] = 1
    np.testing.assert_array_equal(fields[spec.name], [2, 3])
    assert detached.specs[spec.name] is spec
    with pytest.raises(ValueError, match="already"):
        fields.add(spec)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float32"},
        {"axes": ("unknown",)},
        {"particle_policy": "additive"},
        {"minimum": 2.0, "maximum": 1.0},
        {"initial_value": float("nan")},
        {"dtype": "int32", "initial_value": 0.5},
    ],
)
def test_invalid_specs_reject(kwargs):
    """Unsupported representations fail before field allocation."""
    args = dict(name="lab.value", axes=("box",), units="1")
    args.update(kwargs)
    with pytest.raises(ValueError):
        FieldSpec(**args)


def test_custom_field_batch_updates_and_rollback():
    """Extension writes participate in all-box publication and isolation."""
    from particula import FunctionProcess
    from particula.tests.cpu_batch_test import state

    aerosol = state()
    aerosol.fields.add(FieldSpec("lab.exposure", ("box",), "s"))
    target = aerosol.fields["lab.exposure"]

    def advance(box, *, time_step, **_):
        box.fields["lab.exposure"][:] += time_step

    process = FunctionProcess(
        advance, name="exposure", writes=("lab.exposure",)
    )
    process.execute(aerosol, 3, sub_steps=3)
    assert aerosol.fields["lab.exposure"] is target
    np.testing.assert_array_equal(target, [3, 3])

    def fail(box, *, context, **kwargs):
        advance(box, **kwargs)
        if context.box_id == "1":
            raise ValueError("late failure")

    failing = FunctionProcess(fail, name="failure", writes=("lab.exposure",))
    with pytest.raises(ValueError, match="late failure"):
        failing.execute(aerosol, 4)
    np.testing.assert_array_equal(target, [3, 3])


def test_detached_fields_and_box_views():
    """A view writes its staged box while detached state owns its arrays."""
    from particula import _cpu_batch
    from particula.tests.cpu_batch_test import state

    aerosol = state()
    aerosol.fields.add(FieldSpec("lab.value", ("box", "particle"), "1"))
    staged = _cpu_batch.detached(aerosol)
    view = _cpu_batch.box_view(staged, 1)
    view.fields["lab.value"][:] = 8
    np.testing.assert_array_equal(aerosol.fields["lab.value"], 0)
    np.testing.assert_array_equal(staged.fields["lab.value"][0], 0)
    np.testing.assert_array_equal(staged.fields["lab.value"][1], 8)


def test_native_names_cannot_be_shadowed():
    """User fields cannot intercept native publication destinations."""
    with pytest.raises(ValueError, match="conflicts"):
        FieldSpec("gas.concentration", ("box",), "1")


def test_native_coagulation_uses_actual_pairs_for_tracers(monkeypatch):
    """A controlled collision conserves represented additive inventory."""
    from particula import FieldSpec
    from particula.dynamics import BrownianCoagulationStrategy, Coagulation
    from particula.dynamics.coagulation.particle_resolved_step import (
        particle_resolved_method,
    )
    from particula.dynamics.tests.native_runnable_test import state

    aerosol = state()
    aerosol.fields.add(
        FieldSpec(
            "lab.tracer", ("box", "particle"), "kg", particle_policy="additive"
        ),
        np.array([[3.0, 4.0]]),
    )
    before = np.sum(
        aerosol.particles.concentration * aerosol.fields["lab.tracer"]
    )
    monkeypatch.setattr(
        particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        lambda **_: np.array([[0, 1]], dtype=np.int64),
    )
    Coagulation(BrownianCoagulationStrategy("particle_resolved")).execute(
        aerosol, 1
    )
    np.testing.assert_array_equal(aerosol.fields["lab.tracer"], [[0, 7]])
    assert (
        np.sum(aerosol.particles.concentration * aerosol.fields["lab.tracer"])
        == before
    )


def test_dilution_accounts_tracer_removal_without_changing_per_particle_value():
    """Count decay exports tracer inventory rather than shrinking each value."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.dynamics.tests.native_runnable_test import state

    aerosol = state()
    aerosol.fields.add(
        FieldSpec(
            "lab.tracer", ("box", "particle"), "kg", particle_policy="additive"
        ),
        np.array([[3.0, 4.0]]),
    )
    Dilution(DilutionStrategy(np.log(2))).execute(aerosol, 1)
    np.testing.assert_array_equal(aerosol.fields["lab.tracer"], [[3, 4]])
    changes = aerosol.fields.inventory_changes("lab.tracer")
    np.testing.assert_allclose(changes["removal"], [7])
    np.testing.assert_array_equal(changes["source"], [0])
    changes["removal"][:] = 100
    np.testing.assert_allclose(
        aerosol.fields.inventory_changes("lab.tracer")["removal"], [7]
    )


def test_sink_clears_retired_tracers_and_preserves_budget():
    """Complete count loss clears reusable slots and accounts their inventory."""
    fields = ExtensionFields({"box": 1, "particle": 2})
    fields.add(
        FieldSpec(
            "lab.tracer", ("box", "particle"), "kg", particle_policy="additive"
        ),
        np.array([[3.0, 4.0]]),
    )
    fields.retire_particles(np.array([[2.0, 2.0]]), np.array([[0.0, 2.0]]))
    np.testing.assert_array_equal(fields["lab.tracer"], [[0, 4]])
    np.testing.assert_array_equal(
        fields.inventory_changes("lab.tracer")["removal"], [6]
    )
    detached = fields.copy()
    detached.retire_particles(np.array([[0.0, 2.0]]), np.zeros((1, 2)))
    np.testing.assert_array_equal(
        detached.inventory_changes("lab.tracer")["removal"], [14]
    )
    np.testing.assert_array_equal(
        fields.inventory_changes("lab.tracer")["removal"], [6]
    )
