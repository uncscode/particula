# %% [markdown]
# # How precisely could a 60 nm ocimene experiment separate the mechanisms?
#
# Start with the two observed overnight residues at 220 nm. The same initial
# diameter enters both observations. Then compare the 20-hour forecasts for a
# proposed 60 nm sample with efficient vapor removal. Repeat the comparison
# with a wider allowance for locating peaks in the published diameter spectra.

# %%
import csv
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def study_root():
    """Locate the inputs in the repository or the extracted example ZIP."""
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for directory in (start, *start.parents):
            if (directory / "docs/data/liu-terpene/digitized.zip").is_file():
                return directory
    raise FileNotFoundError("Extract the entire study download before running.")


def measurement_bounds(root, work):
    """Reconstruct Figure 4 and S4 reading bounds from the digitized inputs."""
    data = root / "docs/data/liu-terpene"
    with ZipFile(data / "digitized.zip") as archive:
        for name in (
            "sources/observations.csv",
            "sources/time_verification.csv",
            "sources/peak_check.csv",
            "uncertainty/peaks.csv",
        ):
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    estimate = runpy.run_path(
        str(root / "studies/liu-terpene/uncertainty/estimate.py")
    )
    estimate["estimate"](work, data_root=work)
    return work / "read_bounds.csv"


# %% [markdown]
# The volume observable is the cube of the final-to-initial modal aerodynamic
# diameter ratio. Each interval below spans the Figure 4 marker and the paired
# Figure S4 readings under the half-bin assumption. They share a chamber fill.


# %%
def compare():
    root = study_root()
    with TemporaryDirectory() as scratch:
        bounds = measurement_bounds(root, Path(scratch))
        with bounds.open() as source:
            for row in csv.DictReader(source):
                if (
                    row["precursor"] == "ocimene"
                    and row["resolution"] == "half_bin"
                    and int(row["spectrum_min"]) > 900
                ):
                    print(
                        f"220 nm at {row['spectrum_min']} min: "
                        f"{100 * float(row['observed']):.1f}% measured "
                        f"(reading range {100 * float(row['read_low']):.1f}–"
                        f"{100 * float(row['read_high']):.1f}%)."
                    )

    # Accepted native-size parameters were carried to 60 nm. The closest
    # edges of the two forecast ranges set the precision needed at this size.
    with (root / "docs/data/liu-terpene/figure2.csv").open() as source:
        rows = [
            r
            for r in csv.DictReader(source)
            if r["precursor"] == "ocimene" and r["diameter_nm"] == "60"
        ]
    for resolution in ("half_bin", "full_bin"):
        cases = {
            r["mechanism"]: r for r in rows if r["resolution"] == resolution
        }
        slow = cases["evolving_D"]
        pool = cases["long_lived_fraction"]
        gap = 100 * (float(pool["low"]) - float(slow["high"]))
        print(
            f"60 nm, {resolution.replace('_', '-')} peaks: "
            f"{100 * float(slow['low']):.1f}–{100 * float(slow['high']):.1f}% slower mixing, "
            f"{100 * float(pool['low']):.1f}–{100 * float(pool['high']):.1f}% lower volatility. "
            f"Gap: {gap:.1f} percentage points."
        )


if __name__ == "__main__":
    compare()

# %% [markdown]
# The gap shrinks when the original peaks can move farther between sampled
# diameters. Compare it with the resolution and repeatability of a proposed
# second-size experiment. Independent paired fills would measure that
# repeatability, which the two published overnight readings cannot establish.
