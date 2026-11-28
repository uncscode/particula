"""Live equilibrium CCN, independent optical sampling and passive continuation."""

import numpy as np
import pytest
import warp as wp
from particula.execution.ccn_recording import CCNObserver, read_ccn_history
from particula.execution.cpu_simulation import CPUSimulation
from particula.execution.optical_recording import (
    OpticalHistory,
    OpticalObserver,
    read_optical_history,
)
from particula.execution.prepared_recording import run_prepared
from particula.execution.recording import run_until
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
    scenario,
)
from particula.optics.state import OpticalConfig
from particula.particles.properties.ccn import CCNConfig, evaluate_ccn


def _advance_and_sample(plain, t, dt, device, config):
    # Fix the CPU step grid independently of observation requests.
    if device == "numpy":
        while plain.time < t:
            plain.step(min(plain.time + dt, t))
    else:
        plain.run(round(t / dt) - plain.completed_steps)
    return evaluate_ccn(plain.snapshot(), config)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
@pytest.mark.parametrize("mixed", [False, True])
def test_observe_and_replay(device, mixed, tmp_path, monkeypatch):
    """Independent sampling grids match offline thresholds and stochastic state."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario(
        "coagulation" if device != "numpy" else "dilution"
    )

    def owner():
        if device == "numpy":
            return CPUSimulation(initial, process, time_step=dt)
        return process.prepare(
            initial, time_step=dt, device=device, root_seed=42
        )

    run = run_until if device == "numpy" else run_prepared
    model = {"material_kappa": [0.0, 0.3]} if mixed else {"kappa": 0.3}
    config = CCNConfig(
        saturation_ratios=[1.0001, 1.002], water_index=0, **model
    )
    observed, plain = owner(), owner()

    def forbidden():
        raise AssertionError("full snapshot in observer")

    monkeypatch.setattr(observed, "snapshot", forbidden)
    path = tmp_path / "optics.sqlite"
    optics = OpticalObserver(
        observed,
        OpticalHistory(
            path,
            OpticalConfig([550e-9], particle_index=1.5 + 0.01j),
            variables=("extinction",),
        ),
    )
    ccn = CCNObserver(observed, config, batch_size=2)
    try:
        run(
            observed,
            3 * dt,
            optics=optics,
            optical_output_times=[dt],
            ccn=ccn,
            ccn_output_times=[0, 2 * dt],
        )
        run(observed, 5 * dt, ccn=ccn, ccn_initial=True)
        arrays = ccn.arrays()
        np.testing.assert_allclose(
            arrays["time"], dt * np.array([0, 2, 3, 4, 5])
        )
        optical_times = np.concatenate(
            [rows["time"] for _, rows in read_optical_history(path)]
        )
        np.testing.assert_allclose(optical_times, dt * np.array([1, 3]))
        for i, t in enumerate(arrays["time"]):
            reference = _advance_and_sample(plain, t, dt, device, config)
            np.testing.assert_allclose(
                arrays["equilibrium_ccn_number_concentration"][i],
                reference.equilibrium_ccn_number_concentration,
                rtol=2e-12,
                atol=0,
            )
            if not mixed:
                np.testing.assert_allclose(
                    arrays["critical_dry_diameter"][i],
                    reference.critical_dry_diameter,
                    rtol=1e-12,
                    atol=0,
                )
        if device != "numpy":
            canonical_equal(observed.checkpoint(), plain.checkpoint())
        else:
            np.testing.assert_array_equal(
                observed._state.particles.population,
                plain.snapshot().particles.population,
            )
        ccn.save(tmp_path / "history.ccn")
        saved = read_ccn_history(tmp_path / "history.ccn")
        assert (
            saved.metadata["units"]["equilibrium_ccn_number_concentration"]
            == "m^-3"
        )
        for name, values in arrays.items():
            np.testing.assert_array_equal(saved.arrays[name], values)
    finally:
        if device != "numpy":
            observed.close()
            plain.close()


def test_observer_capacity_and_scheduling():
    """Reject wrong owners before advancing and publish complete bounded rows."""
    initial, process, dt = scenario("dilution")
    owner = CPUSimulation(initial, process, time_step=dt)
    other = CPUSimulation(initial, process, time_step=dt)
    config = CCNConfig(saturation_ratios=[1.001], kappa=0.0, water_index=0)
    # One timestamp and two 1x1 float64 arrays use exactly 24 bytes.
    observer = CCNObserver(owner, config, max_bytes=24)
    with pytest.raises(ValueError, match="different owner"):
        run_until(other, dt, ccn=observer)
    assert other.time == 0
    with pytest.raises(ValueError, match="requires"):
        run_until(owner, dt, ccn_output_times=[dt])
    assert owner.time == 0
    with pytest.raises(ValueError, match="limit"):
        run_until(owner, dt, ccn=observer, ccn_initial=True)
    assert owner.time == dt
    latest = observer.latest()
    assert latest["time"] == 0
    latest["critical_dry_diameter"][:] = 0
    assert np.all(observer.latest()["critical_dry_diameter"] > 0)


def test_wrong_archive_units_rejected(tmp_path):
    """The saved unit contract also applies to independently written files."""
    from particula.execution.archive import write_archive

    initial, process, dt = scenario("dilution")
    owner = CPUSimulation(initial, process, time_step=dt)
    sampler = CCNObserver(
        owner, CCNConfig(saturation_ratios=[1.002], kappa=0.3, water_index=0)
    )
    sampler.record()
    path = tmp_path / "units.ccn"
    sampler.save(path)
    result = read_ccn_history(path)
    result.metadata["units"]["critical_dry_diameter"] = "nm"
    write_archive(
        path, kind="analysis", metadata=result.metadata, arrays=result.arrays
    )
    with pytest.raises(ValueError, match="units"):
        read_ccn_history(path)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_live_box_temperature_and_material_weights(device):
    """Thresholds follow sampled box temperature and dry-volume composition."""
    from particula.particles.properties.tests.ccn_test import seed_boxes

    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial = seed_boxes("discrete")
    _, dilution, dt = scenario("dilution")
    config = CCNConfig(
        saturation_ratios=[1.002, 1.004], kappa=(0.3, 0.6), water_index=0
    )
    mixture = CCNConfig(
        saturation_ratios=[1.002, 1.004],
        material_kappa=(0, 0.2, 0.4),
        water_index=0,
    )
    with dilution.prepare(initial, device=device, time_step=dt) as owner:
        sampler, mixed = CCNObserver(owner, config), CCNObserver(owner, mixture)
        sampler.record()
        mixed.record()
        before = sampler.latest()["critical_dry_diameter"]
        np.testing.assert_allclose(
            mixed.latest()["equilibrium_ccn_number_concentration"],
            evaluate_ccn(initial, mixture).equilibrium_ccn_number_concentration,
        )
        owner.update_environment(temperature=[300.0, 270.0])
        owner.run(1)
        sampler.record()
        mixed.record()
        after = sampler.latest()["critical_dry_diameter"]
        np.testing.assert_allclose(
            after / before,
            np.broadcast_to(np.array([280 / 300, 300 / 270])[:, None], (2, 2)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            mixed.latest()["equilibrium_ccn_number_concentration"],
            evaluate_ccn(
                owner.snapshot(), mixture
            ).equilibrium_ccn_number_concentration,
            rtol=1e-12,
        )
