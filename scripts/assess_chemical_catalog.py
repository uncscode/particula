"""Qualify pinned catalog sources against isolated ozone-titration trajectories.

Run with python -m scripts.assess_chemical_catalog. The first run downloads
the published files into the Particula source cache. Results are printed as
JSON. The SOAP cases compare the archived biomass-burning aging coefficients.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from particula.aerosol import Aerosol
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.catalog import ChemistryCatalog
from particula.dynamics.chemistry.catalog.native import compile_channel
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

# Source labels, A [cm3 molecule-1 s-1], activation temperature [K].
CHANNELS = {
    "waccm": ("NO_O3", 3.0e-12, 1500),
    "e3sm": ("lo3_no", 3.0e-12, 1500),
    "mcm": ("7", 1.4e-12, 1310),
    "mcm-full": ("7", 1.4e-12, 1310),
    "wrf-chem": ("T019", 3.0e-12, 1500),
    "cmaq": ("R036", 3.0e-12, 1500),
    "camx": ("3", 2.07e-12, 1400),
    "pmcamx": ("3", 3.0e-12, 1500),
}


def _trajectory(mechanism, a, activation):
    temperatures = np.array([250.0, 298.0, 320.0])
    masses = np.array([0.030006, 0.047997, 0.046005, 0.031998])
    molar = np.tile([1e-6, 1e-6, 0.0, 0.0], (3, 1))
    state = Aerosol(
        particles=ParticleData(
            masses=np.full((3, 1, 1), 1e-18),
            population=np.ones((3, 1)),
            charge=np.zeros((3, 1)),
            density=np.array([1000.0]),
            volume=np.ones(3),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["NO", "O3", "NO2", "O2"], masses, molar * masses, np.zeros(4, bool)
        ),
        environment=EnvironmentData(
            temperature=temperatures,
            pressure=np.full(3, 101325.0),
            saturation_ratio=np.zeros((3, 4)),
        ),
    )
    process = GasChemistry(
        mechanism.bind(state.gas),
        ChemistrySolver(method="rk23", rtol=1e-8, atol=1e-16),
    )
    process.execute(state, time_step=60.0)
    k_si = a * 6.02214076e17 * np.exp(-activation / temperatures)
    remaining = 1 / (1 + 60e-6 * k_si)
    actual = state.gas.concentration / masses * 1e6
    expected = np.column_stack(
        (remaining, remaining, 1 - remaining, 1 - remaining)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-8)
    np.testing.assert_allclose(
        state.gas.concentration.sum(axis=1),
        (molar * masses).sum(axis=1),
        rtol=1e-12,
    )
    return dict(
        NO_at_298_K_umol_m3=actual[1, 0],
        max_error_umol_m3=float(abs(actual - expected).max()),
    )


def _counts(catalog, snapshot):
    records = catalog.selection([snapshot])["reactions"]
    ready = 0
    for record in records:
        try:
            compile_channel(record)
        except ValueError:
            continue
        ready += 1
    return dict(records=len(records), native_without_closure=ready)


def _check_deck_units(catalog, snapshot, expected):
    """Reproduce the archived host's unit vote from the original deck bytes."""
    primary = catalog.show(snapshot)["files"][0]["sha256"]
    deck = catalog.source_bytes(primary).decode().split("Reaction Records")[1]
    molecular = ppm = 0
    for line in deck.splitlines()[2:]:
        fields = line.split()
        if fields and abs(int(fields[1])) in (2, 3):
            if float(fields[3].upper().replace("D", "E")) > 1:
                ppm += 1
            else:
                molecular += 1
    np.testing.assert_equal((molecular, ppm), expected)
    model = catalog.show(f"{snapshot}/3")["rate_model"]
    np.testing.assert_equal(model["concentration_unit"], "molecule/cm^3")
    np.testing.assert_equal(model["time_unit"], "s")
    return dict(molecular_votes=molecular, ppm_votes=ppm)


def assess(catalog):
    """Return per-snapshot inventory, trajectory errors and SOAP yields."""
    report = {}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "gas.json"
        for key, (label, a, activation) in CHANNELS.items():
            snap = catalog.fetch(key)
            catalog.export([f"{snap}/{label}"], path, close_oxygen=True)
            report[key] = dict(
                snapshot=snap,
                **_counts(catalog, snap),
                **_trajectory(ReactionMechanism.load(path), a, activation),
            )
            if key in {"camx", "pmcamx"}:
                expected_votes = (96, 3) if key == "camx" else (73, 2)
                report[key].update(
                    _check_deck_units(catalog, snap, expected_votes)
                )
    for key, expected in (("pmcamx-soap", 1.075), ("pmcamx-soap-alt", 1.0)):
        snap = catalog.fetch(key)
        record = catalog.show(f"{snap}/aging-20-19")
        gain = record["products"]["CG19"]
        np.testing.assert_equal(gain, expected)
        precursor = catalog.show(f"{snap}/SOAP1")
        np.testing.assert_equal(precursor["metadata"]["product_bins"], ["CAS3"])
        np.testing.assert_equal(precursor["metadata"]["yield_unit"], "ppm/ppm")
        np.testing.assert_equal(
            precursor["metadata"]["prefactor_unit"], "cm^3/molecule/s"
        )
        np.testing.assert_equal(record["metadata"]["host_rate_unit"], "1/ppm/h")
        report[key] = dict(
            snapshot=snap, **_counts(catalog, snap), CG20_to_CG19_gain=gain
        )
    return report


if __name__ == "__main__":
    print(json.dumps(assess(ChemistryCatalog()), indent=2))
