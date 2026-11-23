"""Regression against a measured two-liquid feed and an independent TPD mesh."""

import numpy as np
import pytest

pytest.importorskip("particula_aiomfac")

from studies.feng2012.compare import compare


def test_feng_tie_lines_and_tangent_plane():
    """Check conservation, transfer potentials and phase stability diagnostics."""
    cases = compare()["cases"]
    assert len(cases) == 13
    assert all(row["status"] in {"one_liquid", "two_liquids"} for row in cases)
    assert all(row["min_grid_tpd"] >= -1e-7 for row in cases)
    for row in cases:
        if row["status"] == "two_liquids":
            assert row["max_feed_mass_balance_error"] < 1e-12
            assert np.max(np.abs(row["log_activity_mismatches"])) < 1e-7

    # Last 303.15 K tie line, independently identified by the measured salt.
    anchor = next(
        row
        for row in cases
        if row["temperature_k"] == 303.15
        and row["measured_endpoints"][0][2] == 0.4101
    )
    assert anchor["status"] == "two_liquids"
    alpha, beta = np.asarray(anchor["predicted_endpoints"])
    assert alpha[0] > beta[0] and alpha[2] > beta[2] and beta[1] > alpha[1]
    # The chosen feed is a unit mass, so the mass lever rule is exact.
    np.testing.assert_allclose(
        np.asarray(anchor["phase_mass_fractions"]) @ [alpha, beta],
        anchor["feed_mass_fractions"],
        rtol=1e-12,
    )
