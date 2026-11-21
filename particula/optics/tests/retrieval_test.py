"""Recovery and non-identifiability of optical closure parameters."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp
from particula.optics.retrieval import (
    EffectiveIndexModel,
    MixingStateModel,
    OpticalGrid,
    OpticalObservations,
    dry_volume_fractions,
    fit_optical,
    profile_optical,
)


def _grid(wavelength):
    edges = np.geomspace(60e-9, 850e-9, 9)
    centres = np.sqrt(edges[:-1] * edges[1:])
    count = np.exp(-0.5 * (np.log(centres / 280e-9) / 0.52) ** 2) * 3e9
    return OpticalGrid.from_diameter_bins(
        edges, count, wavelength, measure="dN_dlog10D"
    )


def _observations(bulk, selection):
    values = np.array([bulk[group, channel] for group, channel in selection])
    channels = ["extinction", "scattering", "absorption"]
    selectors = [(group, channels[channel]) for group, channel in selection]
    sigmas = np.maximum(0.02 * values, 1e-12)
    cov = np.diag(sigmas**2) * 0.9 + 0.1 * sigmas[:, None] * sigmas
    return OpticalObservations.channels(
        len(bulk), selectors, values, covariance=cov
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fit_index_and_growth(device):
    """Recover n,k,kappa from independently generated multi-RH observations."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    wave = np.array([450, 550, 550, 550, 700]) * 1e-9
    rh = np.array([0.0, 0.0, 0.55, 0.85, 0.85])
    grid = _grid(wave)
    conditions = dict(relative_humidity=rh, sizing_relative_humidity=0.18)
    reference = EffectiveIndexModel(grid, **conditions)
    cuda = EffectiveIndexModel(grid, device=device, **conditions)
    truth = np.array([1.57, 0.035, 0.29])
    bulk = reference.predictions(truth)
    train = _observations(bulk, [(0, 0), (1, 2), (2, 1), (3, 1)])
    bounds = ([1.35, 0.005, 0.01], [1.8, 0.12, 0.8])
    starts = [[1.4, 0.012, 0.55], [1.68, 0.08, 0.18]]
    fit = fit_optical(cuda, train, bounds=bounds, starts=starts)
    fd = fit_optical(
        cuda, train, bounds=bounds, starts=starts, method="finite_difference"
    )
    assert fit.success and fd.success
    npt.assert_allclose(fit.parameters, truth, rtol=5e-8)
    npt.assert_allclose(fit.parameters, fd.parameters, rtol=5e-8)
    npt.assert_allclose(fit.sensitivity, fd.sensitivity, rtol=2e-6, atol=1e-7)
    npt.assert_allclose(
        cuda.predictions(fit.parameters)[-1], bulk[-1], rtol=5e-8
    )
    assert fit.singular_values[-1] > 0


def test_extinction_only_does_not_identify_n_and_k():
    """Quantify an extinction-only ridge with a singular Jacobian and profile."""
    grid = _grid(550e-9)
    model = EffectiveIndexModel(grid)
    bulk = model.predictions([1.55, 0.04])
    train = _observations(bulk, [(0, 0)])
    bounds = ([1.3, 0.005], [1.85, 0.12])
    fit = fit_optical(
        model,
        train,
        bounds=bounds,
        starts=[[1.4, 0.012], [1.7, 0.11]],
        method="finite_difference",
    )
    assert fit.success and fit.chi_square < 1e-12
    assert fit.singular_values[1] == 0
    coordinates = np.linspace(0.005, 0.12, 9)
    _, cost, interval = profile_optical(
        model,
        train,
        fit,
        "k",
        coordinates,
        bounds=bounds,
        method="finite_difference",
    )
    assert np.max(cost) < 1e-11
    assert interval == ((0.005, 0.12),)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mixing_fractions_preserve_dry_bin_counts_and_recover(device):
    """Use different humidity responses and size sensitivity to recover bands."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    wavelengths = np.array([450, 550, 700, 450, 550, 700]) * 1e-9
    grid = _grid(wavelengths)
    counts = grid.number_density.copy()
    # The third group is size-selected, isolating the small fraction.
    counts[2, 2 * grid.radius[2] >= 230e-9] = 0
    grid = OpticalGrid(grid.radius, counts, wavelengths)
    mass, density = [1.1e-6, 3.4e-6], [1750, 1450]
    options = dict(
        volume_fractions=dry_volume_fractions(mass, density),
        species_index=[1.82 + 0.22j, 1.47 + 0.001j],
        species_kappa=[0.01, 0.61],
        relative_humidity=[0, 0, 0, 0.83, 0.83, 0.83],
        dry_boundary=230e-9,
    )
    ref = MixingStateModel(grid, **options)
    candidate = MixingStateModel(grid, device=device, **options)
    for p in ([0, 0], [0.39, 0.62], [1, 1]):
        _, _, counts = ref._population(np.asarray(p))
        npt.assert_allclose(
            counts.reshape((*grid.radius.shape, 3)).sum(axis=-1),
            grid.number_density,
            rtol=3e-16,
        )
    truth = [0.27, 0.78]
    expected = ref.predictions(truth)
    train = _observations(expected, [(0, 2), (2, 2), (3, 1), (4, 1)])
    fit = fit_optical(
        candidate,
        train,
        bounds=([0, 0], [1, 1]),
        starts=[[0.5, 0.5]],
    )
    assert fit.success and min(fit.singular_values) > 1
    npt.assert_allclose(fit.parameters, truth, rtol=1e-7)
    npt.assert_allclose(
        candidate.predictions(fit.parameters)[5], expected[5], rtol=1e-7
    )


def test_diameter_measure_and_prescribed_growth():
    """Integrate a measured dN/dlog10D bin and use one known water ratio."""
    grid = OpticalGrid.from_diameter_bins(
        [100e-9, 180e-9, 320e-9],
        [4e8, 9e8],
        [450e-9, 670e-9],
        measure="dN_dlog10D",
    )
    expected = np.array([4e8, 9e8]) * np.log10([1.8, 320 / 180])
    npt.assert_allclose(grid.number_density, [expected, expected], rtol=1e-14)
    factors = np.array([1.05, 1.2])[:, None]
    known = EffectiveIndexModel(grid, growth_factor=factors)
    device = EffectiveIndexModel(grid, growth_factor=factors, device="cpu")
    guess = [1.47, 0.011]
    npt.assert_allclose(
        device.predictions(guess), known.predictions(guess), rtol=3e-11
    )
    with pytest.raises(ValueError, match="place of kappa"):
        EffectiveIndexModel(grid, kappa=0.2, growth_factor=factors)


def test_identical_species_yield_flat_mixing_profile():
    """Two equal optical and growth endmembers leave fractions unidentified."""
    grid = _grid([450e-9, 550e-9, 700e-9])
    model = MixingStateModel(
        grid,
        volume_fractions=[0.4, 0.6],
        species_kappa=[0.25, 0.25],
        species_index=[1.5 + 0.02j] * 2,
        relative_humidity=[0.2, 0.65, 0.88],
        dry_boundary=230e-9,
        device="cpu",
    )
    target = model.predictions([0.25, 0.75])
    obs = _observations(target, [(0, 0), (1, 1), (2, 2)])
    bounds = ([0, 0], [1, 1])
    fit = fit_optical(
        model, obs, bounds=bounds, starts=[[0.3, 0.5], [0.7, 0.8]]
    )
    assert fit.success
    assert max(fit.singular_values) < 1e-10
    _, _, limits = profile_optical(
        model,
        obs,
        fit,
        fit.names[0],
        [0, 0.25, 0.5, 0.75, 1],
        bounds=bounds,
    )
    assert limits == ((0.0, 1.0),)
