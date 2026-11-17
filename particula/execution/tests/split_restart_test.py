"""Complete-cycle split disk continuation preserves grouping and native fields."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest

from particula import FieldSpec
from particula.execution.cpu_restart import (
    load_cpu_checkpoint,
    save_cpu_checkpoint,
)
from particula.execution.recording import Recorder, run_until
from particula.execution.split_restart import checkpoint_split, resume_split
from particula.execution.splitting import (
    Flow,
    SplitGroup,
    SplitSimulation,
    condensation_flow,
    dilution_flow,
)
from particula.execution.tests.condensation_ode_test import case


def group(strategy, method="strang", coefficient=0.2):
    """Nested deterministic flows retain independently named components."""
    return SplitGroup(
        (
            dilution_flow("ventilation", coefficient),
            SplitGroup(
                (
                    condensation_flow(
                        "growth",
                        strategy,
                        method="rk23",
                        rtol=1e-7,
                        particle_atol=1e-27,
                        gas_atol=1e-17,
                    ),
                    dilution_flow("other_loss", 0.1),
                )
            ),
        ),
        method=method,
    )


@pytest.mark.parametrize("method", ["sequential", "strang"])
def test_split_disk_restart_preserves_exact_future(tmp_path, method):
    """A shortened final cycle and a nested schedule resume bitwise."""
    initial, strategy = case(2)
    initial.fields.add(
        FieldSpec("lab.marker", ("box",), "1"), np.array([3.0, 4.0])
    )
    schedule = group(strategy, method)
    owner = SplitSimulation(initial, schedule, time_step=0.05, time=2)
    history = Recorder()
    run_until(owner, 2.27, output_times=[2, 2.1], recorder=history)
    saved = checkpoint_split(owner)
    path = tmp_path / "split.pcr"
    save_cpu_checkpoint(saved, path)
    restored = resume_split(load_cpu_checkpoint(path), group(strategy, method))
    assert restored.time == owner.time
    assert restored.completed_steps == owner.completed_steps
    for bound in (2.4, 2.5, 2.7):
        run_until(owner, bound)
        run_until(restored, bound, recorder=history)
        assert restored.completed_steps == owner.completed_steps
        left, right = owner.snapshot(), restored.snapshot()
        npt.assert_array_equal(left.particles.masses, right.particles.masses)
        npt.assert_array_equal(
            left.particles.concentration, right.particles.concentration
        )
        npt.assert_array_equal(left.gas.concentration, right.gas.concentration)
        npt.assert_array_equal(right.fields["lab.marker"], [3, 4])
    npt.assert_array_equal(
        history.arrays()["time"], [2, 2.1, 2.27, 2.4, 2.5, 2.7]
    )


def test_changed_nested_order_configuration_and_controller_reject():
    """Identical species widths cannot substitute for identical future physics."""
    initial, strategy = case()
    original = group(strategy)
    owner = SplitSimulation(initial, original, time_step=0.1)
    saved = checkpoint_split(owner)
    for changed in (
        group(strategy, "sequential"),
        group(strategy, coefficient=0.3),
        SplitGroup(tuple(reversed(original.children))),
    ):
        with pytest.raises(ValueError, match="differs"):
            resume_split(saved, changed)
    for malformed in (
        replace(saved, completed_steps=-1),
        replace(saved, schema_version=True),
        replace(saved, time_step=0),
        replace(saved, time=np.nan),
    ):
        with pytest.raises(ValueError):
            resume_split(malformed, original)


def test_unknown_callbacks_are_not_serialized_as_restartable_physics():
    """Custom execution is allowed, but its opaque closure has no disk identity."""
    initial, _ = case()
    owner = SplitSimulation(
        initial,
        Flow("custom", lambda state, time, dt: state, kind="exact"),
        time_step=0.1,
    )
    owner.step(0.1)
    with pytest.raises(TypeError, match="custom flow"):
        checkpoint_split(owner)
    assert owner.time == 0.1
