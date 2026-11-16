"""Prepared histories retain exact physics/RNG with explicit download profiles."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp

from particula.execution.analysis_io import load_analysis, save_analysis
from particula.execution.checkpoint_io import load_checkpoint, save_checkpoint
from particula.execution.prepared_recording import (
    PreparedObserver,
    run_prepared,
)
from particula.execution.recording import Recorder
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
    scenario,
)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Qualify actual CUDA independently of the Warp CPU baseline."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


@pytest.mark.parametrize("download", ["diagnostics", "snapshot"])
def test_prepared_history_has_independent_decay_and_explicit_downloads(
    device, download, monkeypatch, tmp_path
):
    """Fixed-grid output closes independent dilution budgets without slot copies."""
    initial, process, dt = scenario("dilution")
    variables = (
        "number_density",
        "particle_mass_density",
        "gas_concentration",
        "temperature",
        "pressure",
    )
    if download == "snapshot":
        variables += ("particle_masses", "particle_concentration")
    recorder = Recorder(variables)
    with process.prepare(initial, device=device, time_step=dt) as owner:
        observer = PreparedObserver(owner, recorder, download=download)
        assert recorder.last_time is None
        assert observer.initial_snapshot_downloads == 1
        if download == "diagnostics":

            def forbidden():
                raise AssertionError(
                    "reduced observations must not download slots"
                )

            monkeypatch.setattr(owner, "snapshot", forbidden)
        run_prepared(owner, 0.5, output_times=[0, 0.1, 0.3], observer=observer)
        assert owner.completed_steps == 5
        assert observer.observation_downloads == 4
        observer.record()
        assert observer.observation_downloads == 4
    values = recorder.arrays()
    times = np.array([0, dt, 3 * dt, 5 * dt])
    npt.assert_array_equal(values["time"], times)
    factor = np.exp(-0.2 * times)
    for name, value in (
        ("number_density", initial.particles.number_density),
        ("particle_mass_density", initial.particles.species_mass_density),
        ("gas_concentration", initial.gas.concentration),
    ):
        shape = (-1,) + (1,) * value.ndim
        npt.assert_allclose(
            values[name], factor.reshape(shape) * value, rtol=3e-14
        )
    npt.assert_array_equal(
        values["volume"], np.tile(initial.particles.volume, (4, 1))
    )
    path = tmp_path / "gpu-history.pca"
    save_analysis(
        recorder,
        path,
        configuration={"dilution_s-1": 0.2},
        numerical_settings={"time_step_s": dt, "download": download},
        source_identity={"revision": "test-fixture"},
    )
    loaded = load_analysis(path)
    for name, array in values.items():
        npt.assert_array_equal(loaded.arrays[name], array)
    assert loaded.metadata["science"]["gas_names"] == initial.gas.name


@pytest.mark.parametrize("name", ["coagulation", "sequence", "condensation"])
def test_observation_preserves_canonical_future_and_rng(device, name):
    """Diagnostic reductions neither consume physics draws nor change steps."""
    initial, process, dt = scenario(name)
    with (
        process.prepare(
            initial, device=device, time_step=dt, root_seed=137
        ) as plain,
        process.prepare(
            initial, device=device, time_step=dt, root_seed=137
        ) as observed,
    ):
        observer = PreparedObserver(
            observed, Recorder(), download="diagnostics"
        )
        plain.run(5)
        run_prepared(
            observed, 5 * dt, output_times=[0, dt, 3 * dt], observer=observer
        )
        canonical_equal(plain.checkpoint(), observed.checkpoint())
        plain.run(3)
        observed.run(3)
        canonical_equal(plain.checkpoint(), observed.checkpoint())


def test_disk_resume_preserves_grid_labels_without_duplicate_join(
    device, tmp_path
):
    """Restart reuses integer clock labels and emits the resumed join once."""
    initial, process, dt = scenario("coagulation")
    history = Recorder()
    with process.prepare(
        initial, device=device, time_step=dt, root_seed=17
    ) as owner:
        observer = PreparedObserver(owner, history, download="diagnostics")
        run_prepared(owner, 0.3, output_times=[0, 0.1], observer=observer)
        path = tmp_path / "restart.pcr"
        save_checkpoint(owner.checkpoint(), path)
        with process.resume(load_checkpoint(path), device=device) as resumed:
            continuation = PreparedObserver(
                resumed, history, download="diagnostics"
            )
            run_prepared(
                resumed,
                0.5,
                output_times=[0, 0.1, 0.3, 0.4],
                observer=continuation,
            )
            owner.run(2)
            canonical_equal(owner.checkpoint(), resumed.checkpoint())
            assert continuation.observation_downloads == 2
    npt.assert_array_equal(
        history.arrays()["time"], np.array([0, 1, 3, 4, 5]) * dt
    )


def test_bad_schedule_and_history_reject_before_physics(device):
    """Off-grid, unordered and future rows cannot partially run a simulation."""
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, device=device, time_step=dt) as owner:
        before = owner.checkpoint()
        observer = PreparedObserver(owner, Recorder(), download="diagnostics")
        for outputs in ([0.15], [0.2, 0.1], [np.nan], [0.6]):
            with pytest.raises(ValueError):
                run_prepared(
                    owner, 0.5, output_times=outputs, observer=observer
                )
        with pytest.raises(ValueError, match="step grid"):
            run_prepared(owner, 0.55, observer=observer)
        observer.recorder.record(0.2, initial)
        with pytest.raises(ValueError, match="ahead"):
            run_prepared(owner, 0.5, observer=observer)
        assert owner.completed_steps == 0
        canonical_equal(before, owner.checkpoint())


def test_slot_profile_rejects_before_setup_download(device, monkeypatch):
    """Full particles require the explicitly chosen full-download profile."""
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, device=device, time_step=dt) as owner:

        def forbidden():
            raise AssertionError(
                "unsupported profile must reject before transfer"
            )

        monkeypatch.setattr(owner, "snapshot", forbidden)
        with pytest.raises(ValueError, match="slot variables"):
            PreparedObserver(
                owner, Recorder(("particle_masses",)), download="diagnostics"
            )


def test_capture_replay_is_observable_but_active_capture_rejects():
    """The host observer stays outside capture without disabling captured replay."""
    if not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, device="cuda:0", time_step=dt) as owner:
        history = Recorder()
        observer = PreparedObserver(owner, history, download="diagnostics")
        with wp.ScopedCapture(device="cuda:0"):
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                observer.record()
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                run_prepared(owner, 0.3, observer=observer)
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                PreparedObserver(owner, history, download="snapshot")
        assert owner.completed_steps == 0
        owner.capture()
        run_prepared(owner, 0.3, output_times=[0], observer=observer)
        assert owner.captured
        npt.assert_allclose(
            history.arrays()["gas_concentration"][-1],
            initial.gas.concentration * np.exp(-0.2 * 0.3),
            rtol=3e-14,
        )


def test_prepared_writer_saves_after_reduced_observation(device, tmp_path):
    """The same save action composes with GPU reductions outside capture."""
    from particula.execution.analysis_io import AnalysisWriter

    initial, process, dt = scenario("dilution")
    path = tmp_path / "prepared.pca"
    history = Recorder()
    writer = AnalysisWriter(
        history,
        path,
        configuration={"dilution_s-1": 0.2},
        numerical_settings={"time_step_s": dt},
        source_identity={"test": True},
    )
    with process.prepare(initial, device=device, time_step=dt) as owner:
        observer = PreparedObserver(owner, writer, download="diagnostics")
        run_prepared(owner, 0.3, output_times=[0, 0.1], observer=observer)
        assert observer.observation_downloads == 3
    npt.assert_array_equal(
        load_analysis(path).arrays["time"], np.array([0, 1, 3]) * dt
    )
