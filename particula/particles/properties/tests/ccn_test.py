"""Independent dry-volume, extensive weight and analysis-history controls."""

import numpy as np
import particula as par
import pytest
from particula.execution.analysis_io import load_analysis, save_analysis
from particula.execution.recording import Recorder
from particula.particles.properties.ccn import (
    CCNConfig,
    evaluate_ccn,
    evaluate_ccn_history,
)


def seed_boxes(kind):
    """Two boxes with unequal volumes, a wet-only slot and an empty slot."""
    # The pure Kelvin limit at S=e yields dry cutoff A=100 nm exactly.
    diameters = np.array([98, 100, 102, 0, 0], dtype=float) * 1e-9
    dry = np.pi / 6 * diameters**3
    densities = np.array([995.0, 1500.0, 2000.0])
    volumes = np.stack(
        [np.array([2, 1, 2, 4, 0]) * 1e-21, dry / 2, dry / 2], axis=-1
    )
    counts = np.array([[3, 7, 11, 13, 0], [31, 37, 41, 43, 0]], dtype=float)
    p = par.particles.ParticleData(
        masses=np.tile(volumes * densities, (2, 1, 1)),
        density=densities,
        population=counts,
        volume=np.array([2.0, 0.5]),
        charge=np.zeros_like(counts),
        distribution_type=kind,
    )
    return par.Aerosol(
        particles=p,
        gas=par.gas.GasData(
            ["gas"], np.array([0.028]), np.zeros((2, 1)), [False]
        ),
        environment=par.gas.EnvironmentData(
            [280.0, 300.0], [101325.0] * 2, np.zeros((2, 1))
        ),
    )


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_dry_diameter_and_represented_volume(kind, tmp_path):
    """Hand counts include ties, exclude water-only seeds and recover density."""
    state = seed_boxes(kind)
    config = CCNConfig(
        saturation_ratios=[np.e], kappa=0.0, kelvin_diameter=1e-7, water_index=0
    )
    expected = np.array([[(7 + 11) / 2], [(37 + 41) / 0.5]])
    np.testing.assert_allclose(
        evaluate_ccn(state, config).equilibrium_ccn_number_concentration,
        expected,
    )
    state.particles.masses[:, :, 0] *= 1000
    assert np.array_equal(
        evaluate_ccn(state, config).equilibrium_ccn_number_concentration,
        expected,
    )
    history = Recorder(("particle_population", "particle_masses"))
    history.record(0.0, state)
    path = tmp_path / "offline.analysis"
    save_analysis(
        history,
        path,
        configuration={},
        numerical_settings={},
        source_identity={},
    )
    reread = load_analysis(path)
    config2 = CCNConfig(
        saturation_ratios=[np.e, np.e**2],
        kappa=0.0,
        kelvin_diameter=1e-7,
        water_index=0,
    )
    row = evaluate_ccn_history(reread, config2)
    np.testing.assert_allclose(
        row.equilibrium_ccn_number_concentration[0], [[9, 10.5], [156, 218]]
    )
    with pytest.raises(ValueError, match="temperature"):
        evaluate_ccn_history(
            reread,
            CCNConfig(saturation_ratios=[np.e], kappa=0.0, water_index=0),
        )


def test_volume_weighted_kappa_and_temperature():
    """Equal dry volumes contribute equally at unequal material densities."""
    state = seed_boxes("particle_resolved")
    mixed = CCNConfig(
        saturation_ratios=[1.002], material_kappa=[0, 0.2, 0.4], water_index=0
    )
    out = evaluate_ccn(state, mixed)
    common = CCNConfig(saturation_ratios=[1.002], kappa=0.3, water_index=0)
    reference = evaluate_ccn(state, common)
    expected = reference.critical_dry_diameter
    np.testing.assert_allclose(
        out.critical_dry_diameter[:, :3, 0],
        np.repeat(expected, 3, axis=1),
        rtol=1e-12,
    )
    assert np.all(np.isnan(out.critical_dry_diameter[:, 3:]))
    assert expected[0, 0] / expected[1, 0] == pytest.approx(300 / 280)
    state.particles.masses[:, :, 1] = 0.0
    second = evaluate_ccn(state, mixed)
    assert np.all(
        second.critical_dry_diameter[:, :3] < out.critical_dry_diameter[:, :3]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"saturation_ratios": [1]},
        {"kelvin_diameter": np.inf},
        {"kappa": -1},
        {"kappa": None},
        {"water_index": 3},
        {"kappa": [0, 1, 2]},
        {"material_kappa": [0, 0.1, 0.2]},
        {"kappa": None, "material_kappa": [1, 0, 0]},
        {"kappa": None, "material_kappa": [0, 0.1]},
    ],
)
def test_reject_bad_config(overrides):
    """Reject invalid domains and ambiguous box or material axes."""
    opts = dict(saturation_ratios=[1.002], kappa=0.3, water_index=0)
    opts.update(overrides)
    with pytest.raises(ValueError):
        evaluate_ccn(seed_boxes("discrete"), CCNConfig(**opts))


def test_new_settings_require_raw_particles_and_count_measure(tmp_path):
    """Dry eligibility cannot be reconstructed from bulk inventories alone."""
    config = CCNConfig(saturation_ratios=[1.002], kappa=0.3, water_index=0)
    snapshot = seed_boxes("discrete")
    recorder = Recorder(
        ("temperature", "number_density", "particle_mass_density")
    )
    recorder.record(0.0, snapshot)
    path = tmp_path / "bulk.analysis"
    save_analysis(
        recorder,
        path,
        configuration={},
        numerical_settings={},
        source_identity={},
    )
    with pytest.raises(ValueError, match="particle_masses"):
        evaluate_ccn_history(load_analysis(path), config)
    with pytest.raises(ValueError, match="discrete or resolved"):
        config.bind(2, 3, "continuous_pdf")
