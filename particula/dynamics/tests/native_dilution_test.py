"""Native all-gas dilution, raw storage and transaction regressions."""

import numpy as np
import pytest
from particula.dynamics import Dilution, DilutionStrategy
from particula.dynamics import dilution as module
from particula.dynamics.tests.native_runnable_test import state


def snapshots(aerosol):
    """Capture all physical arrays and their identities independently."""
    return [
        (array, array.copy())
        for container in (aerosol.particles, aerosol.gas, aerosol.environment)
        for array in vars(container).values()
        if isinstance(array, np.ndarray)
    ]


@pytest.mark.parametrize(
    "kind,volume",
    [
        ("discrete", 1.0),
        ("continuous_pdf", 1.0),
        ("particle_resolved", 0.25),
        ("particle_resolved", 1.0),
        ("particle_resolved", 4.0),
    ],
)
@pytest.mark.parametrize("mask", [[False, True, True], [False] * 3, [True] * 3])
def test_native_all_gas_decay_and_measure_correct_rate(kind, volume, mask):
    """Raw density or counts decay once; resolved rates use N/V exactly once."""
    aerosol = state(volume, kind)
    aerosol.gas.partitioning[:] = mask
    before = snapshots(aerosol)
    old_counts = aerosol.particles.concentration.copy()
    old_gas = aerosol.gas.concentration.copy()
    process = Dilution(DilutionStrategy(0.25))
    np.testing.assert_allclose(
        process.rate(aerosol), -0.25 * old_counts / volume, rtol=1e-15
    )
    assert process.execute(aerosol, 4.0, 4) is aerosol
    np.testing.assert_allclose(
        aerosol.particles.concentration,
        old_counts * np.exp(-1),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        aerosol.gas.concentration, old_gas * np.exp(-1), rtol=1e-12, atol=0
    )
    for array, values in before:
        if (
            array is not aerosol.particles.concentration
            and array is not aerosol.gas.concentration
        ):
            np.testing.assert_array_equal(array, values)


@pytest.mark.parametrize(
    "failure", ["nan_gas", "nan_counts", "volume", "readonly", "alias"]
)
@pytest.mark.parametrize("time", [0.0, 2.0])
def test_native_rejections_leave_every_array_unchanged(failure, time):
    """All-source admission runs even on no-op calls."""
    aerosol = state()
    if failure == "nan_gas":
        aerosol.gas.concentration[0, 0] = np.nan
    elif failure == "nan_counts":
        aerosol.particles.concentration[0, 0] = np.nan
    elif failure == "volume":
        aerosol.particles.volume[0] = 0
    elif failure == "readonly":
        aerosol.gas.concentration.flags.writeable = False
    else:
        aerosol.particles.charge = aerosol.particles.concentration.view()
    before = snapshots(aerosol)
    with pytest.raises(ValueError):
        Dilution(DilutionStrategy(0.25)).execute(aerosol, time, 2)
    for array, values in before:
        np.testing.assert_array_equal(array, values)


def test_failed_later_substep_restores_its_entry_and_keeps_prior_commit(
    monkeypatch,
):
    """A partially failing writer is rolled back without undoing an earlier step."""
    aerosol = state()
    original = module.np.copyto
    calls = []
    error = RuntimeError("second substep gas write")
    before_counts = aerosol.particles.concentration.copy()
    before_gas = aerosol.gas.concentration.copy()

    def write(target, values):
        if (
            target is aerosol.particles.concentration
            or target is aerosol.gas.concentration
        ):
            calls.append(target)
        original(target, values)
        if len(calls) == 4:
            raise error

    monkeypatch.setattr(module.np, "copyto", write)
    with pytest.raises(RuntimeError) as caught:
        Dilution(DilutionStrategy(0.25)).execute(aerosol, 4.0, 4)
    assert caught.value is error
    assert len(calls) == 4
    np.testing.assert_allclose(
        aerosol.particles.concentration,
        before_counts * np.exp(-0.25),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        aerosol.gas.concentration, before_gas * np.exp(-0.25), rtol=1e-15
    )


def test_rollback_failure_chains_original_error(monkeypatch):
    """An unexpected inability to restore is diagnostic context, not replacement."""
    aerosol = state()
    original = module.np.copyto
    error = RuntimeError("writer failure")

    def write(target, values):
        original(target, values)
        if target is aerosol.gas.concentration:
            target.flags.writeable = False
            raise error

    monkeypatch.setattr(module.np, "copyto", write)
    with pytest.raises(RuntimeError) as caught:
        DilutionStrategy(0.25).step(aerosol, 1.0)
    assert caught.value is error
    assert isinstance(caught.value.__cause__, ValueError)


def test_invalid_candidate_rejects_before_any_writer(monkeypatch):
    """A faulty candidate calculation cannot bypass transactional preflight."""
    aerosol = state()
    before = snapshots(aerosol)
    monkeypatch.setattr(
        module, "get_dilution_step", lambda *args: np.array([-1.0])
    )
    with pytest.raises(ValueError, match="candidate"):
        DilutionStrategy(0.25).step(aerosol, 1.0)
    for array, values in before:
        np.testing.assert_array_equal(array, values)
