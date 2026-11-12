"""Bounded scientific checks of the actual published simulation setups.

Full notebook workloads are separately executed for publication. These checks
retain each example's species configuration and test a short evolution of a
small resolved population (the binned example retains its original grid).
"""

import ast
import copy
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
import pytest
from particula.particles.slot_management import activate_slots

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/Examples"
SIMULATIONS = EXAMPLES / "Simulations/Notebooks"


def _setup_through(path, target):
    """Execute the published setup up to the named process assignment."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id == target
            for name in node.targets
        ):
            tree.body = tree.body[: index + 1]
            break
    else:
        raise AssertionError(f"Missing setup boundary: {target}")
    namespace = {"__name__": "example_fixture"}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - trusted repository example
    return namespace


def _inventory(aerosol):
    """Extensive material inventory, independently of example diagnostics."""
    particles = aerosol.particles
    weighted = np.sum(
        particles.masses * particles.concentration[..., None], axis=1
    )
    if particles.distribution_type == "particle_resolved":
        return (
            weighted
            + aerosol.gas.concentration[:, : particles.n_species]
            * particles.volume[:, None]
        )
    return (weighted + aerosol.gas.concentration) * particles.volume[:, None]


@pytest.mark.parametrize(
    "filename,process_name,aerosol_name,width",
    [
        (
            "Biomass_Burning_Cloud_Interactions",
            "condensation_process",
            "aerosol",
            3,
        ),
        ("Cough_Droplets_Partitioning", "condensation_process", "aerosol", 12),
        (
            "Organic_Partitioning_and_Coagulation",
            "condensation_process",
            "aerosol",
            11,
        ),
        ("Soot_Formation_in_Flames", "condensation_process", "aerosol", 24),
        ("Cloud_Chamber_Single_Cycle", "condensation", "aerosol", 3),
        ("Cloud_Chamber_Multi_Cycle", "condensation_as", "aerosol_as", 2),
    ],
)
def test_simulation_species_inventory_and_storage(
    monkeypatch, filename, process_name, aerosol_name, width
):
    """Native condensation preserves every mapped inventory and held arrays."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    values = _setup_through(SIMULATIONS / f"{filename}.py", process_name)
    aerosol = copy.deepcopy(values[aerosol_name])
    if aerosol.particles.distribution_type == "particle_resolved":
        original = aerosol.particles
        aerosol = par.Aerosol(
            particles=par.particles.create_resolved_particle_data(
                masses=original.masses[0, :8].copy(),
                concentration=original.concentration[0, :8].copy(),
                density=original.density.copy(),
                volume=float(original.volume[0]),
            ),
            gas=aerosol.gas,
            environment=aerosol.environment,
        )
    if filename == "Soot_Formation_in_Flames":
        # The hot precursor has no uptake; exercise the cooled parcel.
        aerosol.environment.temperature[0] = 300.0
    assert aerosol.particles.n_species == width
    particles, gas = aerosol.particles, aerosol.gas
    masses, concentrations = particles.masses, gas.concentration
    initial = _inventory(aerosol)
    before = masses.copy()
    carrier = gas.concentration[:, width:].copy()
    for _ in range(3):
        values[process_name].execute(aerosol, time_step=1e-5, sub_steps=2)
        np.testing.assert_allclose(
            _inventory(aerosol), initial, rtol=1e-12, atol=1e-30
        )
    assert aerosol.particles is particles and aerosol.gas is gas
    assert particles.masses is masses and gas.concentration is concentrations
    assert np.any(before != masses), "Fixture must exercise mass transfer"
    assert np.all(np.isfinite(masses)) and np.all(masses >= 0)
    assert np.all(np.isfinite(concentrations)) and np.all(concentrations >= 0)
    np.testing.assert_array_equal(gas.concentration[:, width:], carrier)
    plt.close("all")


def test_particle_injection_counts_inventory_and_exhaustion(monkeypatch):
    """Two external injections fill reserved capacity; another is atomic."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    values = runpy.run_path(
        str(
            EXAMPLES
            / "Dynamics/Customization/Adding_Particles_During_Simulation.py"
        )
    )
    aerosol = values["aerosol"]
    particles = aerosol.particles
    assert particles.masses.shape == (1, 120_000, 1)
    assert particles.concentration.sum() == 120_000
    np.testing.assert_allclose(
        particles.species_mass_density[0, 0],
        (
            values["particle_mass_sample"].sum()
            + 2 * values["particle_mass_add"].sum()
        )
        / 1e-6,
        rtol=1e-12,
        atol=1e-30,
    )
    before = particles.masses.copy()
    with pytest.raises(ValueError, match="capacity"):
        activate_slots(
            particles,
            values["request_masses"],
            values["request_counts"],
            values["request_charge"],
            values["requested_counts"],
        )
    np.testing.assert_array_equal(particles.masses, before)
    np.testing.assert_array_equal(particles.concentration, 1.0)
    plt.close("all")


def test_cloud_cycle_diagnostics_track_dry_identity_and_each_peak():
    """Changing wet size cannot masquerade as deposited dry material."""
    path = SIMULATIONS / "Cloud_Chamber_Multi_Cycle.py"
    wanted = {
        "masses_to_diameter_generic",
        "cumulative_loss_by_bin",
        "get_water_fraction_at_phase_peaks",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"np": np}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - trusted repository example
    initial = np.array([[1e-18, 0.0], [2e-18, 0.0]])
    wet = np.array([[1e-18, 9e-18], [0.0, 0.0]])
    history = [{"masses": initial}, {"masses": wet}]
    loss, fraction = namespace["cumulative_loss_by_bin"](
        history, np.array([1000.0, 1000.0]), np.array([0.0, 1e-6])
    )
    np.testing.assert_allclose(loss, [2e-18], rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(fraction, [2 / 3], rtol=1e-12)
    records = []
    for water in (1.0, 2.0, 3.0, 4.0):
        records.extend(
            [
                {"phase": "humid", "masses": np.array([[1.0, water]])},
                {"phase": "dry", "masses": np.array([[1.0, 0.0]])},
            ]
        )
    np.testing.assert_allclose(
        namespace["get_water_fraction_at_phase_peaks"](records, 1),
        [1 / 2, 2 / 3, 3 / 4, 4 / 5],
        rtol=1e-12,
    )


def test_cooled_soot_condensation_then_coagulation_inventory(monkeypatch):
    """The retained process order conserves material along a cooling path."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    values = _setup_through(
        SIMULATIONS / "Soot_Formation_in_Flames.py", "coagulation_process"
    )
    source = values["aerosol"]
    aerosol = par.Aerosol(
        particles=par.particles.create_resolved_particle_data(
            masses=source.particles.masses[0, :16].copy(),
            density=source.particles.density.copy(),
            volume=float(source.particles.volume[0]),
        ),
        gas=copy.deepcopy(source.gas),
        environment=copy.deepcopy(source.environment),
    )
    initial = _inventory(aerosol)
    count = aerosol.particles.concentration.sum()
    for temperature in (700.0, 400.0, 300.0):
        aerosol.environment.temperature[0] = temperature
        values["condensation_process"].execute(
            aerosol, time_step=0.04, sub_steps=100
        )
        values["coagulation_process"].execute(aerosol, time_step=0.04)
        np.testing.assert_allclose(
            _inventory(aerosol), initial, rtol=1e-12, atol=1e-30
        )
        assert 0 < aerosol.particles.concentration.sum() <= count
        count = aerosol.particles.concentration.sum()
    assert np.any(aerosol.particles.masses[0, :, :-1] > 0)
    assert aerosol.environment.temperature[0] == 300.0
    plt.close("all")


def test_three_cloud_compositions_humidity_reset_and_dry_mass(monkeypatch):
    """Short cycles retain all three material mappings and prescribed forcing."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    values = _setup_through(
        SIMULATIONS / "Cloud_Chamber_Multi_Cycle.py", "condensation_mixed"
    )
    for suffix, water_index in (("as", 1), ("sucrose", 1), ("mixed", 2)):
        source = values[f"aerosol_{suffix}"]
        aerosol = par.Aerosol(
            particles=par.particles.create_resolved_particle_data(
                masses=source.particles.masses[0, :8].copy(),
                density=source.particles.density.copy(),
                volume=0.25,
            ),
            gas=copy.deepcopy(source.gas),
            environment=copy.deepcopy(source.environment),
        )
        dry_before = aerosol.particles.masses[0, :, :water_index].sum()
        final, history = values["run_cycle"](
            aerosol,
            values[f"condensation_{suffix}"],
            values["wall_loss"],
            humid_duration=2,
            dry_duration=3,
            water_index=water_index,
            c_water_humid=values["c_water_humid"],
            c_water_dry=values["c_water_dry"],
        )
        assert final is aerosol
        assert [r["time"] for r in history] == [0, 1, 2, 3, 4, 5]
        assert [r["phase"] for r in history] == ["humid"] * 3 + ["dry"] * 3
        dry_after = final.particles.masses[0, :, :water_index].sum()
        assert 0 <= dry_after <= dry_before
        assert np.all(np.isfinite(final.particles.masses))
        assert np.all(final.particles.masses >= 0)
        np.testing.assert_array_equal(final.gas.concentration[0, -1], 1.2)
    plt.close("all")
