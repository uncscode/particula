"""Independent two-volume budgets for the represented population tutorial."""

import runpy
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/studies/tutorials/building-an-aerosol/particles"
    / "represented_population.py"
)


@pytest.fixture(scope="module")
def example():
    """Load the published functions without drawing the figure."""
    return runpy.run_path(str(SOURCE))


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_slot_and_species_budgets(example, kind):
    """Evaluate the two slots and both species from the stated inputs."""
    box = example["discrete_and_resolved"]()[kind]
    np.testing.assert_array_equal(box["volume_m3"], [0.25, 4])
    np.testing.assert_array_equal(box["population"], [[1, 2], [16, 32]])
    np.testing.assert_array_equal(box["slot_density"], [[4, 8], [4, 8]])
    np.testing.assert_array_equal(box["number_per_m3"], [12, 12])
    np.testing.assert_array_equal(box["represented_number"], [3, 48])
    np.testing.assert_allclose(
        box["species_kg_per_m3"],
        np.array([[20, 44]] * 2) * 1e-21,
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        box["species_kg"],
        np.array([[5, 11], [80, 176]]) * 1e-21,
        rtol=1e-14,
        atol=0,
    )


def test_pdf_node_quadrature_and_species(example):
    """Use explicit trapezoids for number and the two material inventories."""
    pdf = example["radius_distribution"]()
    expected = {
        "radius_nm": [1, 2, 4],
        "node_width_nm": [0.5, 1.5, 1],
        "dn_dr_per_m3_per_nm": [2, 4, 2],
        "node_number_per_m3": [1, 6, 2],
        "population": np.array([[0.5, 1, 0.5], [8, 16, 8]]) * 1e9,
        "slot_density": np.array([[2, 4, 2]] * 2) * 1e9,
        "number_per_m3": [9, 9],
        "represented_number": [2.25, 36],
    }
    for field, values in expected.items():
        np.testing.assert_allclose(pdf[field], values, rtol=1e-14, atol=0)

    # A one-nanometre-radius sphere weighs 4π/3 × 10⁻²⁴ kg.
    # The three spheres weigh [1, 8, 64] times that reference.
    # A: 1×0.25 + 6×4 + 2×48. B: 1×0.75 + 6×4 + 2×16.
    mass_density = np.array([120.25, 56.75]) * (4 * np.pi / 3 * 1e-24)
    np.testing.assert_allclose(
        pdf["species_kg_per_m3"],
        [mass_density, mass_density],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        pdf["species_kg"],
        [mass_density / 4, mass_density * 4],
        rtol=1e-14,
        atol=0,
    )


def test_band_edges_and_tails(example):
    """Recover linear-segment integrals from the per-decade bar heights."""
    spectrum = example["diameter_spectrum"](example["radius_distribution"]())
    edges = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    expected_bins = [1.25, 1.75, 3.5, 2.5]
    np.testing.assert_array_equal(spectrum["diameter_edges_nm"], edges)
    np.testing.assert_allclose(
        spectrum["bin_number_per_m3"], expected_bins, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        spectrum["per_m3_per_decade"] * np.log10(edges[1:] / edges[:-1]),
        expected_bins,
        rtol=1e-14,
        atol=0,
    )
    parts = [
        spectrum[f"{name}_per_m3"]
        for name in ("below_band", "in_band", "above_band")
    ]
    np.testing.assert_allclose(parts, [1.25, 5.25, 2.5], rtol=1e-14, atol=0)
    np.testing.assert_allclose(sum(parts), 9.0, rtol=1e-14, atol=0)


@pytest.mark.parametrize(("value", "units"), [(2e6, "1/m^3"), (2.0, "1/cm^3")])
def test_builder_measurement(example, value, units):
    """One cubic metre contains one million cubic centimetres."""
    if units == "1/cm^3" and find_spec("pint") is None:
        with pytest.raises(ImportError, match="pint"):
            example["import_measurement"](value, units)
        return
    sample = example["import_measurement"](value, units)
    np.testing.assert_allclose(
        sample.population, [[500_000]], rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(sample.number_density, [2e6], rtol=1e-14, atol=0)


def test_replacement_and_rejection(example):
    """A rejected new sample retains its predecessor's volume and count."""
    assert example["replace_measurement"]() == {
        "imported": [0.25, 500_000],
        "current_volume": [0.25, 100],
        "new_volume": [0.5, 200],
        "after_rejection": [0.5, 200],
    }
