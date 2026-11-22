"""Regression tests for the Antarctic film composition inversion."""

import numpy as np
import pytest
from scripts.antarctic_ccn.model import SaltFilm, _match_fraction


@pytest.mark.parametrize("active", [True, False])
def test_low_growth_bins_reconstruct_conditioning_humidity(active):
    """Recover low-salt roots for GF 1.1 at the five field diameters."""
    film = SaltFilm("AS", 298.15)
    for diameter in (35, 50, 75, 110, 165):
        fraction = _match_fraction(film, 1.1, diameter, 0.90, active)
        assert fraction is not None and 0.07 < fraction < 0.11
        ratio = (1.1**3 - 1) * film.water_density / (film.solid * fraction)
        assert film.bounds[0] <= ratio <= film.bounds[1]
        log_rh = film.saturation(ratio, diameter, fraction, active)
        assert np.exp(log_rh) == pytest.approx(0.90, abs=1e-10)
