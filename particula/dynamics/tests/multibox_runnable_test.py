"""Compare batched dynamics with independent same-ID single-box executions."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from particula import _cpu_batch
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    Dilution,
    DilutionStrategy,
    MassCondensation,
    SphericalWallLossStrategy,
    WallLoss,
)
from particula.dynamics.tests.native_nucleation_test import fixture, runnable
from particula.dynamics.tests.native_runnable_test import condensation, state
from particula.execution import CPUExecutionContext


def batch(single):
    """Expand independent state, with distinct environment and gas per box."""
    result = _cpu_batch.detached(single)
    for container, names in (
        (result.particles, ("masses", "concentration", "charge", "volume")),
        (result.gas, ("concentration",)),
        (result.environment, ("temperature", "pressure", "saturation_ratio")),
    ):
        for name in names:
            values = getattr(container, name)
            setattr(container, name, np.concatenate([values, values.copy()]))
    result.particles.volume[1] = 4.0
    result.environment.temperature[1] = 310.0
    result.gas.concentration[1] *= 0.5
    return result


def test_failed_later_wall_box_does_not_advance_data_or_streams():
    """A late physical failure rolls back staged draws as well as storage."""
    aerosol = batch(state())
    reference = _cpu_batch.detached(aerosol)
    process = WallLoss(
        SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
    )
    context = CPUExecutionContext(91, ("a", "b"))
    before = {
        name: value.copy() for name, value in _cpu_batch.arrays(aerosol).items()
    }
    aerosol.particles.masses[1, 0] = 0
    invalid = aerosol.particles.masses.copy()
    with pytest.raises(ValueError, match="positive mass"):
        process.execute(aerosol, 10.0, context=context)
    np.testing.assert_array_equal(aerosol.particles.masses, invalid)
    for name in process._writer_fields(aerosol):
        if name != "particles.masses":
            np.testing.assert_array_equal(
                _cpu_batch.arrays(aerosol)[name], before[name]
            )
    assert context._streams == {}
    aerosol.particles.masses[:] = before["particles.masses"]
    process.execute(aerosol, 10.0, context=context)
    process.execute(
        reference, 10.0, context=CPUExecutionContext(91, ("a", "b"))
    )
    for name in process._writer_fields(aerosol):
        np.testing.assert_array_equal(
            _cpu_batch.arrays(aerosol)[name], _cpu_batch.arrays(reference)[name]
        )


def test_wall_streams_follow_box_identity_across_reordering():
    """Changing lane order preserves each box's state and RNG continuation."""
    original = batch(state())
    reordered = _cpu_batch.detached(original)
    for name, value in _cpu_batch.arrays(reordered).items():
        if name not in (
            "particles.density",
            "gas.molar_mass",
            "gas.partitioning",
        ):
            value[:] = value[::-1].copy()
    process = WallLoss(
        SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
    )
    first = CPUExecutionContext(13, ("a", "b"))
    second = CPUExecutionContext(13, ("b", "a"))
    for _ in range(3):
        process.execute(original, 0.01, context=first)
        process.execute(reordered, 0.01, context=second)
    for name in process._writer_fields(original):
        np.testing.assert_array_equal(
            _cpu_batch.arrays(original)[name],
            _cpu_batch.arrays(reordered)[name][::-1],
        )


@pytest.mark.parametrize(
    "family", ["condensation", "coagulation", "wall", "dilution", "nucleation"]
)
def test_batch_matches_single_boxes_and_preserves_identity(family):
    """All five runners preserve per-box numerical and stream independence."""
    if family == "nucleation":
        single, config = fixture()
        process = runnable(single, config)
        process.commit_config = replace(
            process.commit_config,
            **{
                name: np.repeat(getattr(process.commit_config, name), 2)
                for name in (
                    "requested_scale",
                    "minimum_scale",
                    "minimum_volume",
                )
            },
        )
    else:
        single = state()
        process = {
            "condensation": lambda: MassCondensation(condensation(single.gas)),
            "coagulation": lambda: Coagulation(
                BrownianCoagulationStrategy("particle_resolved")
            ),
            "wall": lambda: WallLoss(
                SphericalWallLossStrategy(0.001, 0.5, "particle_resolved")
            ),
            "dilution": lambda: Dilution(DilutionStrategy(0.1)),
        }[family]()
    aerosol = batch(single)
    originals = _cpu_batch.arrays(aerosol)
    references = []
    for box in range(2):
        reference = _cpu_batch.detached(_cpu_batch.box_view(aerosol, box))
        worker = deepcopy(process)
        if family == "nucleation":
            worker.commit_config = replace(
                process.commit_config,
                **{
                    name: getattr(process.commit_config, name)[box : box + 1]
                    for name in (
                        "requested_scale",
                        "minimum_scale",
                        "minimum_volume",
                    )
                },
            )
        worker.execute(
            reference, 0.01, context=CPUExecutionContext(17, (str(box),))
        )
        references.append(reference)
    if family == "coagulation":
        # Resolved coagulation has no continuous number-rate API.
        with pytest.raises(ValueError, match="distribution type"):
            process.rate(aerosol)
    else:
        assert process.rate(aerosol).shape[0] == 2
    assert (
        process.execute(
            aerosol, 0.01, context=CPUExecutionContext(17, ("0", "1"))
        )
        is aerosol
    )
    for name, values in _cpu_batch.arrays(aerosol).items():
        assert values is originals[name]
        if name in process._writer_fields(aerosol):
            expected = np.concatenate(
                [_cpu_batch.arrays(ref)[name] for ref in references]
            )
            np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0)
