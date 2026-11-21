"""Live bulk optics sampled without copying the native particle state."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp

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
from particula.optics import get_mie_efficiencies
from particula.optics.state import OpticalConfig, mix_refractive_index


@pytest.mark.parametrize("backend", ["numpy", "cpu", "cuda:0"])
@pytest.mark.parametrize("mixed", [False, True])
def test_dense_optics_without_snapshots(backend, mixed, tmp_path, monkeypatch):
    """Check absolute attenuation and continuation with resolved weights."""
    if backend == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario("dilution")
    particles = initial.particles
    wave = np.array([450e-9, 650e-9])
    table = np.tile(
        np.linspace(1.4, 1.7, particles.n_species) + 0.02j, (wave.size, 1)
    )
    config = (
        OpticalConfig(wave, species_index=table)
        if mixed
        else OpticalConfig(wave, particle_index=1.55 + 0.02j)
    )

    def setup():
        if backend == "numpy":
            return CPUSimulation(initial, process, time_step=dt)
        return process.prepare(initial, time_step=dt, device=backend)

    run = run_until if backend == "numpy" else run_prepared
    owner = setup()

    def forbidden():
        raise AssertionError("Optical observer took a full snapshot")

    monkeypatch.setattr(owner, "snapshot", forbidden)
    path = tmp_path / "optical.sqlite"
    try:
        for end in (2 * dt, 5 * dt):
            history = OpticalHistory(path, config, variables=("extinction",))
            run(
                owner,
                end,
                optical_initial=True,
                optics=OpticalObserver(owner, history, batch_size=2),
            )
        chunks = list(read_optical_history(path))
        assert [len(row["time"]) for _, row in chunks] == [2, 1, 2, 1]
        times = np.concatenate([row["time"] for _, row in chunks])
        npt.assert_allclose(times, dt * np.arange(6), rtol=1e-15)
        radii = particles.radii[:, None, :]
        optical_index = 1.55 + 0.02j
        if mixed:
            v = particles.masses / particles.density
            fraction = v / v.sum(axis=-1, keepdims=True)
            dielectric = table[None, :, None, :] ** 2
            p = np.sum(
                fraction[:, None, :, :] * (dielectric - 1) / (dielectric + 2),
                axis=-1,
            )
            optical_index = np.sqrt((1 + 2 * p) / (1 - p))
        qext = get_mie_efficiencies(
            radii, wave[None, :, None], optical_index
        ).extinction
        initial_extinction = np.sum(
            qext
            * np.pi
            * radii**2
            * (particles.concentration / particles.volume[:, None])[:, None, :],
            axis=-1,
        )
        expected = np.exp(-0.2 * times[:, None, None]) * initial_extinction
        obtained = np.concatenate([row["extinction"] for _, row in chunks])
        npt.assert_allclose(obtained, expected, rtol=4e-10, atol=1e-15)

        if backend != "numpy":
            with setup() as plain:
                plain.run(5)
                canonical_equal(owner.checkpoint(), plain.checkpoint())
    finally:
        if backend != "numpy":
            owner.close()


def test_lorentz_lorenz_limits():
    """Pure components and equal-index mixtures retain the input index."""
    m = np.array([1.33, 1.72 + 0.25j])
    npt.assert_allclose(mix_refractive_index(m, np.eye(2)), m, rtol=2e-15)
    npt.assert_allclose(
        mix_refractive_index([m[1], m[1]], [0.25, 0.75]), m[1], rtol=2e-15
    )
    with pytest.raises(ValueError, match="sum to one"):
        mix_refractive_index(m, [0.5, 0.6])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_independent_snapshot_schedule_through_growth(device, tmp_path):
    """Sample changed radii densely and compare sparse native states."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")

    initial, process, dt = scenario("condensation")
    config = OpticalConfig([450e-9, 550e-9, 700e-9], particle_index=1.5 + 0.04j)
    path = tmp_path / "growth.sqlite"
    with process.prepare(initial, time_step=dt, device=device) as owner:
        reference = process.prepare(initial, time_step=dt, device=device)
        with reference:
            observer = OpticalObserver(
                owner, OpticalHistory(path, config), batch_size=3
            )
            run_prepared(owner, 6 * dt, optics=observer, optical_initial=True)
            extinction = np.concatenate(
                [row["extinction"] for _, row in read_optical_history(path)]
            )
            assert extinction.shape[0] == 7
            for row, steps in ((0, 0), (2, 2), (6, 4)):
                reference.run(steps)
                particles = reference.snapshot().particles
                radius = particles.radii[:, None, :]
                wave = np.asarray(config.wavelength)[None, :, None]
                qext = get_mie_efficiencies(
                    radius, wave, config.particle_index[0]
                ).extinction
                weight = particles.concentration / particles.volume[:, None]
                npt.assert_allclose(
                    extinction[row],
                    np.sum(
                        qext * np.pi * radius**2 * weight[:, None, :], axis=-1
                    ),
                    rtol=4e-10,
                )
            assert np.any(extinction[-1] != extinction[0])
            canonical_equal(owner.checkpoint(), reference.checkpoint())
