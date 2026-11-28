"""Exercise the published live and offline workflows on the NumPy owner."""

from docs.studies.examples.thermodynamics.activity.equilibrium_ccn_spectrum import (
    check_run,
    run_chamber,
)


def test_equilibrium_ccn_spectrum_example(tmp_path):
    """Compare independent cohorts over dilution and a held-out setting."""
    _, row = run_chamber("NumPy", tmp_path, record_ccn=True)
    check_run(row)
    assert row["ccn_file_bytes"] == (tmp_path / "live.ccn").stat().st_size
    assert (
        row["particle_file_bytes"]
        == (tmp_path / "particles.analysis").stat().st_size
    )
