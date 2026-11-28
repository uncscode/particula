"""Replay full and reduced populations against final 2023 SOAP measurements."""

import itertools

import numpy as np
from analysis import HERE, component, optical_kernel, write_rows
from numpy.polynomial.legendre import leggauss


def kernels(edges, group, scale, index, order=24, fit_case="reference"):
    """Integrate each piecewise-constant AMP density in log10 D.

    Split at scenario boundaries so no quadrature node straddles a cutoff.
    The modal assignment is frozen in the measured size coordinate.
    """
    fits = np.genfromtxt(
        HERE / "fits.csv",
        names=True,
        delimiter=",",
        dtype=None,
        encoding="utf8",
    )
    params = fits[(fits["n2o_lo_ppbv"] == group) & (fits["case"] == fit_case)]
    nodes, weights = leggauss(order)
    support = (3.80019, 950.8064)
    cuts = np.unique(np.concatenate([edges, support, [150 / scale]]))
    sub_edges = np.unique(
        np.linspace(np.log10(cuts[:-1]), np.log10(cuts[1:]), 9, axis=1)
    )
    left, right = sub_edges[:-1], sub_edges[1:]
    widths = (right - left) / 2
    d = 10 ** ((left + right)[:, None] / 2 + widths[:, None] * nodes)
    fit_modes = np.array(
        [
            component(d, row["height_cm3"], row["number_median_nm"], row["gsd"])
            for row in params
        ]
    )
    fraction = fit_modes[params["mode"] == "small"][0] / fit_modes.sum(axis=0)
    fraction[(d < support[0]) | (d > support[1])] = 0
    physical = d * scale
    area = np.pi * (physical * 1e-3) ** 2
    ext = optical_kernel(physical.ravel(), index).reshape(d.shape)
    quantities = [
        area,
        ext,
        area * fraction,
        ext * fraction,
        area * (physical < 150),
        ext * (physical < 150),
    ]
    bins = np.searchsorted(edges, d[:, 0]) - 1
    return np.array(
        [
            np.bincount(
                bins, weights=widths * (q @ weights), minlength=len(edges) - 1
            )
            for q in quantities
        ]
    )


def summarize(flight, group, measured, values, index, scale):
    """Summarize paired model errors and removed area on matched windows."""
    area, ext, mode_area, mode_ext, cutoff_area, cutoff_ext = values
    base = ext - measured
    result = dict(
        flight=flight,
        n2o_lo_ppbv=group,
        real_index=index.real,
        imaginary_index=index.imag,
        diameter_scale=scale,
        n=len(area),
        median_SOAP_STP_Mm1=np.median(measured),
        median_full_STP_Mm1=np.median(ext),
        median_full_area_STP=np.median(area),
        median_full_bias_Mm1=np.median(base),
        median_full_abs_error_Mm1=np.median(abs(base)),
        below_first_precision_anchor=int(np.sum(measured < 10)),
    )
    for label, lost_area, loss in [
        ("mode", mode_area, mode_ext),
        ("cutoff", cutoff_area, cutoff_ext),
    ]:
        reduced = base - loss
        for key, value in {
            "area_loss": np.median(lost_area),
            "ext_loss": np.median(loss),
            "area_fraction": np.median(lost_area / area),
            "ext_fraction": np.median(loss / ext),
            "bias": np.median(reduced),
            "abs_error": np.median(abs(reduced)),
            "worse_count": np.sum(abs(reduced) > abs(base)),
        }.items():
            result[label + "_" + key] = value
    return result


def _challenge_attribution(
    data, select, group, flight, measured, edges, density
):
    # Challenge the attribution and air-mass screen with matched cases, keeping
    # each comparison on the same observed windows and optical assumptions.
    tests = []
    selected_o3 = data["o3_ppbv"][select]
    for label, included, allocation in [
        ("reference", np.ones(len(group), dtype=bool), "reference"),
        (
            "alternative fit weighting",
            np.ones(len(group), dtype=bool),
            "number-weighted",
        ),
        ("ozone above 500 ppbv", selected_o3 > 500, "reference"),
        (
            "excluding March 8 and 11",
            ~np.isin(flight, [20230308, 20230311]),
            "reference",
        ),
    ]:
        values = np.empty((6, len(group)))
        for g in np.unique(group):
            take = group == g
            values[:, take] = (
                kernels(edges, g, 1.0, 1.5, fit_case=allocation)
                @ density[take].T
            )
        result = summarize(
            "pooled", 0, measured[included], values[:, included], 1.5, 1.0
        )
        result["check"] = label
        result["n_flights"] = len(np.unique(flight[included]))
        tests.append(result)
    write_rows("flight-robustness.csv", tests)


def main():
    """Score the frozen predictions on final-revision 2023 measurements."""
    data = np.load(HERE / "raw/joined.npz")
    lat, alt, n2o = (
        data[key] for key in ("latitude_deg", "altitude_m", "n2o_ppbv")
    )
    select = (
        (lat >= 60)
        & (alt >= 12000)
        & (alt <= 19000)
        & (n2o > 270)
        & (n2o < 330)
    )
    density = data["dndlog10d_cm3"][select]
    measured = data["soap_Mm1"][select]
    flight = data["flight"][select]
    group = np.where(n2o[select] < 300, 270, 300)
    edges = data["edges_nm"]
    np.testing.assert_equal(density.shape, (len(measured), len(edges) - 1))
    np.testing.assert_array_equal(np.isfinite(density) & (density >= 0), True)
    np.testing.assert_array_equal(np.isfinite(measured), True)
    summaries, scenarios, windows, accuracy_matches = [], [], [], []
    for scale, real, imag in itertools.product(
        [0.9, 1.0, 1.1], [1.4, 1.5, 1.6], [0.0, 0.01]
    ):
        index = complex(real, imag)
        nominal = (scale, real, imag) == (1.0, 1.5, 0.0)
        predictions = np.empty((6, len(group)))
        for g in np.unique(group):
            take = group == g
            kernel = kernels(edges, g, scale, index)
            predictions[:, take] = kernel @ density[take].T
            if nominal:
                refined = (
                    kernels(edges, g, scale, index, order=48) @ density[take].T
                )
                np.testing.assert_allclose(
                    refined, predictions[:, take], rtol=2e-6
                )
        for g in [0, *np.unique(group)]:
            take = np.ones(len(group), dtype=bool) if g == 0 else group == g
            scenarios.append(
                summarize(
                    "pooled",
                    g,
                    measured[take],
                    predictions[:, take],
                    index,
                    scale,
                )
            )
        if not nominal:
            continue
        for f in np.unique(flight):
            for g in np.unique(group[flight == f]):
                take = (flight == f) & (group == g)
                summaries.append(
                    summarize(
                        f, g, measured[take], predictions[:, take], index, scale
                    )
                )
        area, ext, ma, me, ca, ce = predictions
        for i, j in enumerate(np.flatnonzero(select)):
            pressure, temperature = (
                data["pressure_hpa"][j],
                data["temperature_k"][j],
            )
            density_ratio = pressure / 1013.0 * 273.15 / temperature
            allowed = 0.02 * measured[i]
            if measured[i] > 0 and abs(ext[i] - measured[i]) <= allowed:
                # Conditional bound for a positive, uniform loss. Start with
                # the actual signed model residual in this same window.
                fraction = np.clip(
                    (ext[i] - measured[i] + allowed) / me[i], 0, 1
                )
                cutoff_in_band = abs(ext[i] - ce[i] - measured[i]) <= allowed
                accuracy_matches.append(
                    dict(
                        flight=int(flight[i]),
                        start_s=data["start_s"][j],
                        measured_ext=measured[i],
                        full_ext=ext[i],
                        reduced_cutoff_ext=ext[i] - ce[i],
                        cutoff_in_accuracy_band=bool(cutoff_in_band),
                        cutoff_area_STP=ca[i],
                        cutoff_area_ambient=ca[i] * density_ratio,
                        max_uniform_mode_area_STP=fraction * ma[i],
                    )
                )
            windows.append(
                dict(
                    flight=int(flight[i]),
                    start_s=data["start_s"][j],
                    stop_s=data["stop_s"][j],
                    soap_samples=data["soap_n"][j],
                    n2o_ppbv=n2o[j],
                    latitude_deg=lat[j],
                    altitude_m=alt[j],
                    o3_ppbv=data["o3_ppbv"][j],
                    pressure_hpa=pressure,
                    temperature_k=temperature,
                    number_density_ambient_over_STP=density_ratio,
                    observed_ext=measured[i],
                    full_ext=ext[i],
                    without_mode_ext=ext[i] - me[i],
                    without_below150_ext=ext[i] - ce[i],
                    full_area=area[i],
                    removed_mode_area=ma[i],
                    removed_below150_area=ca[i],
                )
            )
    _challenge_attribution(
        data, select, group, flight, measured, edges, density
    )
    write_rows("accuracy-window-diagnostics.csv", accuracy_matches)
    write_rows("flight-windows.csv", windows)
    write_rows("flight-results.csv", summaries)
    write_rows("flight-sensitivity.csv", scenarios)


if __name__ == "__main__":
    main()
