"""Integrate alpha-pinene ozonolysis using the native-rate archive monitors."""

import argparse
import csv
from pathlib import Path

import numpy as np
from prepare import HERE, RECORDS, edf_rows

PRESSURE_PA = 101325.0  # No run-specific barometer measurement is archived.
BOLTZMANN = 1.380649e-23
K_298 = 9.6e-17  # cm3 molecule-1 s-1, Voliotis et al. Sect. 3.3
LOG10_RATE_ERROR = 0.15  # Cox et al. (2020), Table 1, 298 K reliability
END_S = 18000.0


def channel(path, name):
    """Return valid source times and values for a named EDF channel."""
    rows = edf_rows(path.read_bytes())
    times = np.array([row["TIME"] for row in rows])
    if name is None:  # The one VOC channel has a different source name per run.
        name = next(key for key in rows[0] if key != "TIME")
    values = np.array([row[name] for row in rows])
    valid = np.isfinite(times) & np.isfinite(values) & (values != -9999.9)
    if name == "T_a":
        valid &= values > 0  # sensor dropout is a zero T/RH pair
    times, values = times[valid], values[valid]
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"Invalid clock in {path}")
    return times, values


def budget(run, raw_dir):
    """Compare integrated ozonolysis with measured five-hour VOC depletion."""
    context, outcome = [raw_dir / str(run) / suffix for suffix in RECORDS[run]]
    t_gas, o3 = channel(next(context.glob("*_GAS.edf")), "O3")
    t_voc, fraction = channel(next(outcome.glob("*_VOC.edf")), None)
    thermo = list(context.glob("*_RHT.edf"))
    if thermo:
        t_temp, celsius = channel(thermo[0], "T_a")
        kelvin = celsius + 273.15
    else:
        t_temp, kelvin = np.array([0, END_S]), np.array([299.0, 299.0])
    knots = np.unique(np.r_[0, END_S, t_gas, t_voc, t_temp])
    knots = knots[(knots >= 0) & (knots <= END_S)]
    # Simpson quadrature on each interval resolves the product of interpolated
    # O3 and fraction even when their original sample times differ.
    grid = np.sort(np.r_[knots, (knots[1:] + knots[:-1]) / 2])
    temp = np.interp(grid, t_temp, kelvin)
    available = np.interp(grid, t_voc, fraction)
    ozone = (
        np.interp(grid, t_gas, o3)
        * 1e-9
        * PRESSURE_PA
        / (BOLTZMANN * temp)
        / 1e6
    )
    depleted = available[0] - available[-1]
    for law, rate in [
        ("published_constant", K_298),
        ("iupac_temperature", 8.22e-16 * np.exp(-640.0 / temp)),
    ]:
        flux = rate * ozone * available
        reacted = np.sum(
            np.diff(knots) / 6 * (flux[:-2:2] + 4 * flux[1::2] + flux[2::2])
        )
        ratio = reacted / depleted
        yield (
            run,
            int(END_S),
            PRESSURE_PA,
            "RHT" if thermo else "assumed_299K",
            law,
            depleted,
            reacted,
            depleted - reacted,
            ratio,
            ratio / 10**LOG10_RATE_ERROR,
            ratio * 10**LOG10_RATE_ERROR,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    args = parser.parse_args()
    with (HERE / "ozone_budget.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "run",
                "end_s",
                "pressure_pa",
                "temperature",
                "rate_law",
                "observed_depletion_fraction",
                "ozone_reacted_fraction",
                "unassigned_fraction",
                "ozone_share",
                "share_low_rate",
                "share_high_rate",
            )
        )
        for run in (1, 5, 6):
            writer.writerows(budget(run, args.raw_dir))
