"""Independent representation-aware bulk particle property checks.

Mass checks use rtol=1e-12 and atol=1e-30: NumPy's default absolute
tolerance would hide errors in the roughly 1e-18 kg fixture inventories.
"""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gpu import conversion
from particula.particles.particle_data import ParticleData
from particula.particles.particle_data_builder import ParticleDataBuilder


def _particles(kind: str, volume: float = 1.0) -> ParticleData:
    return ParticleData(
        masses=np.array([[[0.5e-18, 1.5e-18], [2e-18, 3e-18], [0, 0]]]),
        concentration=np.array([[8.0, 0.5, 0.0]]),
        charge=np.zeros((1, 3)),
        density=np.array([1000.0, 1100.0]),
        volume=np.array([volume]),
        distribution_type=kind,
    )


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_resolved_number_and_species_mass_density(volume: float) -> None:
    """Resolved counts normalize once; extensive species mass is invariant."""
    data = _particles("particle_resolved", volume)
    npt.assert_allclose(data.number_density, [8.5 / volume])
    npt.assert_allclose(
        data.slot_concentration_density, [[8 / volume, 0.5 / volume, 0]]
    )
    npt.assert_allclose(
        data.species_mass_inventory,
        [[5e-18, 13.5e-18]],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        data.species_mass_density,
        [[5e-18 / volume, 13.5e-18 / volume]],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        data.total_mass[0], [2e-18, 5e-18, 0], rtol=1e-12, atol=1e-30
    )


def test_pmf_density_is_already_per_volume() -> None:
    """Fractional PMF bins add directly, without a second volume division."""
    data = _particles("discrete")
    npt.assert_allclose(data.number_density, [8.5])
    npt.assert_allclose(
        data.species_mass_density,
        [[5e-18, 13.5e-18]],
        rtol=1e-12,
        atol=1e-30,
    )


def test_pdf_nonuniform_radius_integrates_mass_times_pdf() -> None:
    """Varying species mass distinguishes product quadrature from sums."""
    data = _particles("continuous_pdf")
    data.radius_grid = np.array([1e-9, 2e-9, 4e-9])
    data.concentration[:] = [[1e9, 2e9, 1e9]]
    # First interval: (1 * 0.5 + 2 * 2) / 2 = 2.25 (in e-18 kg/m³).
    # Second interval: (2 * 2 + 1 * 0) = 4, giving 6.25 total.
    npt.assert_allclose(data.number_density, [4.5])
    npt.assert_allclose(
        data.species_mass_density,
        [[6.25e-18, 9.75e-18]],
        rtol=1e-12,
        atol=1e-30,
    )
    data.concentration[:] = 0
    data.masses[:] = 0
    npt.assert_array_equal(data.number_density, [0])
    npt.assert_array_equal(data.species_mass_density, [[0, 0]])
    npt.assert_array_equal(data.radius_grid, [1e-9, 2e-9, 4e-9])


def test_pdf_inventory_does_not_form_species_weighted_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrate node products without a full three-dimensional scratch."""
    data = _particles("continuous_pdf")
    data.radius_grid = np.array([1e-9, 2e-9, 4e-9])
    data.concentration[:] = [[1e9, 2e9, 1e9]]

    def unexpected_trapezoid(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "PDF species integration must contract node weights"
        )

    monkeypatch.setattr(np, "trapezoid", unexpected_trapezoid)
    npt.assert_allclose(
        data.species_mass_density,
        [[6.25e-18, 9.75e-18]],
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.parametrize(
    "field",
    ["masses", "concentration", "charge", "density", "volume"],
)
def test_post_mutation_wrong_dtype_rejects_read_only(field: str) -> None:
    """Tagged helpers reject schema drift before arithmetic or mutation."""
    data = _particles("particle_resolved")
    original = getattr(data, field)
    invalid = original.astype(object)
    setattr(data, field, invalid)
    with pytest.raises(ValueError, match="float64"):
        _ = data.species_mass_density
    npt.assert_array_equal(getattr(data, field), invalid)


def test_pdf_grid_wrong_dtype_rejects_read_only() -> None:
    """Grid validation reports a domain error rather than a NumPy type error."""
    data = _particles("continuous_pdf")
    data.radius_grid = np.array([1e-9, 2e-9, 4e-9], dtype=object)
    with pytest.raises(ValueError, match="radius grid"):
        _ = data.number_density
    assert data.radius_grid.dtype == object


def test_copy_and_post_mutation_validation() -> None:
    """Copies detach grid and arrays; rejected reads do not repair mutation."""
    data = _particles("continuous_pdf")
    data.radius_grid = np.array([1e-9, 2e-9, 4e-9])
    detached = data.copy()
    detached.concentration[0, 0] = -1
    before = detached.concentration.copy()
    with pytest.raises(ValueError, match="finite physical"):
        detached.validate_representation()
    with pytest.raises(ValueError, match="finite physical"):
        _ = detached.number_density
    npt.assert_array_equal(detached.concentration, before)
    data.radius_grid[1] = 3e-9
    npt.assert_array_equal(detached.radius_grid, [1e-9, 2e-9, 4e-9])


def test_builder_kind_grid_and_volume() -> None:
    """Builder checks kind, PDF grid and fixed or positive volume on creation."""
    builder = (
        ParticleDataBuilder()
        .set_masses(np.zeros((3, 1)))
        .set_density(np.array([1000.0]))
    )
    with pytest.raises(ValueError, match="distribution_type"):
        builder.build()
    builder.set_distribution_type("continuous_pdf")
    with pytest.raises(ValueError, match="radius grid"):
        builder.build()
    builder.set_radius_grid(np.array([1e-9, 2e-9, 4e-9]))
    assert builder.build().distribution_type == "continuous_pdf"
    builder.set_volume(np.array([0.25]))
    with pytest.raises(ValueError, match="exactly 1"):
        builder.build()
    builder.set_distribution_type("particle_resolved").set_radius_grid(
        np.array([])
    )
    with pytest.raises(ValueError, match="only valid"):
        builder.build()


def test_pdf_transfer_rejects_before_warp_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported PDF grid provenance must not be lost at the Warp boundary."""
    data = _particles("continuous_pdf")
    data.radius_grid = np.array([1e-9, 2e-9, 4e-9])

    def unexpected_probe() -> None:
        raise AssertionError("Warp must not be probed for tagged data")

    monkeypatch.setattr(conversion, "_ensure_warp_available", unexpected_probe)
    for copy in (True, False):
        with pytest.raises(ValueError, match="continuous_pdf.*unsupported"):
            conversion.to_warp_particle_data(data, device="cpu", copy=copy)


@pytest.mark.parametrize(
    "kind, units",
    [("continuous_pdf", "1/m^3"), ("particle_resolved", "1/m^3")],
)
def test_builder_rejects_concentration_units_for_kind(
    kind: str, units: str
) -> None:
    """Density input must not be silently reinterpreted as a PDF or count."""
    builder = (
        ParticleDataBuilder()
        .set_distribution_type(kind)
        .set_masses(np.ones((2, 1)))
        .set_density(np.array([1000.0]))
        .set_concentration(np.ones(2), units=units)
    )
    if kind == "continuous_pdf":
        builder.set_radius_grid(np.array([1e-9, 2e-9]))
    with pytest.raises(ValueError, match="units must match"):
        builder.build()


def test_one_bad_box_rejects_every_read_without_mutation() -> None:
    """A bad resolved volume cannot yield a partial bulk-density result."""
    data = _particles("particle_resolved")
    data.masses = np.repeat(data.masses, 2, axis=0)
    data.concentration = np.repeat(data.concentration, 2, axis=0)
    data.charge = np.repeat(data.charge, 2, axis=0)
    data.volume = np.array([0.25, 0.0])
    before = [
        values.copy()
        for values in (
            data.masses,
            data.concentration,
            data.charge,
            data.density,
            data.volume,
        )
    ]
    for getter in (
        lambda: data.slot_concentration_density,
        lambda: data.number_density,
        lambda: data.species_mass_density,
        lambda: data.species_mass_inventory,
    ):
        with pytest.raises(ValueError, match="volume must be positive"):
            getter()
    for original, current in zip(
        before,
        (
            data.masses,
            data.concentration,
            data.charge,
            data.density,
            data.volume,
        ),
        strict=True,
    ):
        npt.assert_array_equal(current, original)
