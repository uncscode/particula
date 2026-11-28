# %% [markdown]
# # Read a particle counter log and a mobility size spectrum
#
# Can we recover a broken logger record and reproduce the total number printed
# by a mobility sizer? Two short excerpts let us check both results by hand.
# The counter recorded detector counts per second on 9 July 2022. The sizer
# reported particles per cubic centimetre per decade of mobility diameter on
# 7 July. Each file samples a different day and has its own time axis.
#
# From the Particula checkout, install the reader and figure dependencies with
# `pip install ".[data]"`. Both excerpts ship in the example download.
# The counter excerpt contains the first 21 physical lines of the original file.
# The mobility excerpt contains its 24-line preamble, column headings and first
# scan. To reproduce the excerpts, take those same lines from the pinned
# [counter log](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/CPC_3010_data/CPC_3010_data_20220709_Jul.csv)
# and [mobility export](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/SMPS_data/2022-07-07_095151_SMPS.csv).
#
# The code locates the excerpts from the checkout or extracted download. Each
# named profile supplies its own delimiter, timestamp convention and units.
# The sizer prints local calendar fields without a time zone. Its built-in
# profile assumes UTC. Check the deployment clock before aligning other data.
# Use `describe_profile("cpc-3010")` from `particula.data` to inspect the
# counter profile. For a custom logger, save the description as JSON or YAML,
# edit the fields, and pass the file path as `profile=`. Calling
# `validate_profile(profile, source=source).raise_for_errors()` checks a
# profile against a sample input header before a longer import.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from particula.data import (
    import_to_parquet,
    open_measurement,
    read_measurement,
)


def excerpt_directory():
    """Find both source excerpts in a checkout or extracted study download."""
    relative = Path("docs/data/measurement-import")
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / relative / "counter.csv").is_file():
                return parent / relative
    raise FileNotFoundError(f"Extract the download containing {relative}")


# %% [markdown]
# ## Trace a count rate back to the logger
#
# Line 1 ends after the temperature and a comma. Line 2 supplies the final two
# fields. The 3010 profile can join that pair and record both line numbers.
# For a longer log, write the completed readings and their issues directly to
# Parquet. Reopen the saved dataset and select the columns needed for a plot.
# The elapsed time comes from the timestamps, so a gap in sampling stays visible.


# %%
def read_counter(source, destination):
    """Stream the logger to disk and recover the trace and repair evidence."""
    import_to_parquet(source, destination, profile="cpc-3010")
    dataset = open_measurement(destination)
    frame = dataset.lazy_frame.select(
        (pl.col("time_utc") - pl.col("time_utc").first())
        .dt.total_seconds()
        .alias("elapsed_s"),
        "count_rate",
        "temperature",
        "repaired",
    ).collect()
    repairs = (
        dataset.lazy_issues.filter(pl.col("code") == "split_record_repaired")
        .select("source_line_start", "source_line_end")
        .collect()
    )
    return {
        "elapsed_s": frame["elapsed_s"].to_list(),
        "rate_counts_s": frame["count_rate"].to_list(),
        "temperature_K": frame["temperature"].to_list(),
        "repaired": frame["repaired"].to_list(),
        "repair_lines": [list(span) for span in repairs.rows()],
    }


# %% [markdown]
# ## Recover the number concentration in one scan
#
# Each mobility-sizer row holds an entire number spectrum and the vendor's
# total concentration. Look up that row's diameter grid to find its bin centres.
# The export specifies 128 channels per decade. Multiplying each channel by its
# width in log₁₀ diameter, 1/128, gives its contribution to the total.
# Use that declared width throughout, since the printed centres are rounded.
# Convert m⁻³ to cm⁻³ for comparison with the original file and keep the
# instrument's warning tokens alongside the integrated result.


# %%
def read_spectrum(source):
    """Compare the integral of dN/dlog10(Dp) with the printed scan total."""
    dataset = read_measurement(source, profile="smps-aim")
    scan = dataset.frame.row(0, named=True)
    grid = dataset.grids[scan["grid_id"]]
    diameter_nm = np.asarray(grid["coordinates"]["diameter"]) * 1e9
    per_decade_cm3 = np.asarray(scan["number_spectrum"]) / 1e6
    return {
        "diameter_nm": diameter_nm.tolist(),
        "per_decade_cm3": per_decade_cm3.tolist(),
        "integral_cm3": float(per_decade_cm3.sum() * grid["log10_bin_width"]),
        "vendor_total_cm3": scan["total_number"] / 1e6,
        "flags": scan["vendor_flags"],
    }


# %% [markdown]
# ## Inspect the two imports
#
# Run the readers on their own files and print the quantities that can be
# checked against the source. The plot follows the count rate through the
# logger's first minute and shows how number is distributed across mobility
# diameters in the selected scan. The open circle identifies the observation
# assembled from two physical lines.


# %%
def run_example():
    """Import the two excerpts and print independently checkable readings."""
    source = excerpt_directory()
    with TemporaryDirectory() as temporary:
        counter = read_counter(source / "counter.csv", Path(temporary) / "log")
    spectrum = read_spectrum(source / "mobility.csv")
    print(
        f"{len(counter['elapsed_s'])} counter readings. "
        f"Repair spans (inclusive): {counter['repair_lines']}. "
        f"First rate: {counter['rate_counts_s'][0]:,.0f} counts/s."
    )
    print(
        f"{len(spectrum['diameter_nm'])} mobility channels. "
        f"Integrated: {spectrum['integral_cm3']:.5f} cm⁻³. "
        f"Vendor total: {spectrum['vendor_total_cm3']:.2f} cm⁻³."
    )
    print("Instrument warnings:", ", ".join(spectrum["flags"]))
    return {"counter": counter, "spectrum": spectrum}


def plot_results(report):
    """Draw the counter time series and the flagged mobility spectrum."""
    import matplotlib.pyplot as plt
    from particula import plotting

    counter = report["counter"]
    spectrum = report["spectrum"]
    seconds = np.asarray(counter["elapsed_s"])
    rate = np.asarray(counter["rate_counts_s"])
    repaired = np.asarray(counter["repaired"])
    with plt.style.context("particula.default"):
        fig, (logger, sizer) = plotting.subplots(1, 2)
        logger.plot(seconds, rate, "o-", markersize=3)
        logger.plot(
            seconds[repaired],
            rate[repaired],
            "o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=9,
            label="Repaired record",
        )
        logger.set(
            xlabel="Elapsed time (s)",
            ylabel="Detector count rate (counts/s)",
            title="Particle counter · 9 July",
        )
        plotting.legend(logger)
        sizer.semilogx(spectrum["diameter_nm"], spectrum["per_decade_cm3"])
        sizer.set(
            xlabel="Mobility diameter (nm)",
            ylabel="dN/dlog₁₀(Dp) (cm⁻³)",
            title="Mobility sizer · 7 July",
        )
        plotting.fit(fig)
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_results(run_example())
    plt.show()

# %% [markdown]
# Lines 1–2 yield a single reading of 32,660 counts/s at 290.25 K. Including
# that record gives 20 observations. Converting their count rates to number
# concentration would require the applicable sample flow. The complete
# counter log contains 33,267 observations, including 13 repaired records.
#
# The 203 channels integrate to 2,168.99816 cm⁻³, within 0.002 cm⁻³ of the
# printed 2,169.00 cm⁻³. At the export’s printed precision, summing the selected
# channels with the declared width reproduces the vendor total. The scan also
# carries both “Detector aerosol flow rate error” and “Incomplete Scan”. All
# 2,854 scans in the full export carry those
# two warnings. Excluding incomplete scans would therefore leave zero scans
# for an average. Before using this file to infer ambient particle number,
# consult the deployment record to establish what those warnings meant for
# sampling flow and scan coverage.
