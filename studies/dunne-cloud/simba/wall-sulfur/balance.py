"""Independent size-band balance during the dark, non-cleaning interval of run 1833."""

from pathlib import Path
import sys, json, csv
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from replay import GRID, geometry, measurements, TEMP, PRESSURE, par, KD

RADIUS = (3 * 26.1 / (4 * np.pi)) ** (1 / 3)
NATIVE = par.dynamics.get_spherical_wall_loss_coefficient_via_system_state
BANDS = [(5.0, 9.7), (9.7, 15.0), (15.0, 25.0), (25.0, 40.0)]


def write(name, rows):
    with (HERE / name).open("w") as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wall(d, eddy):
    return NATIVE(eddy, np.asarray(d) * 0.5e-9, 1350.0, TEMP, PRESSURE, RADIUS)


def normal_eddy():
    ref = np.interp(10, GRID, geometry()[:, 4])
    return brentq(lambda x: float(wall(10.0, x)) - ref, 1e-6, 500.0)


def fit_balance():
    t, d, e, spec, acid, *_ = measurements()
    h = t - t[0]
    secs = h * 3600
    eddy = normal_eddy()
    g = geometry()
    basewall = wall(d, eddy)
    mass = 1350 * np.pi / 6 * (d * 1e-9) ** 3
    kernel = (
        par.dynamics.get_brownian_kernel_via_system_state(
            d * 0.5e-9, mass, TEMP, PRESSURE
        )
        * 1e6
    )
    daughter = (d[:, None] ** 3 + d[None, :] ** 3) ** (1 / 3)
    counts = spec * np.diff(e)[:, None]
    scenarios = [
        ("primary", 6.015, 6.345, 1.15, 1.0),
        ("earlier", 5.89, 6.345, 1.15, 1.0),
        ("trimmed", 6.04, 6.31, 1.15, 1.0),
        ("no_coag", 6.015, 6.345, 1.15, 0.0),
        ("double_coag", 6.015, 6.345, 1.15, 2.0),
        ("low_acid", 6.015, 6.345, 1.05, 1.0),
        ("high_acid", 6.015, 6.345, 1.25, 1.0),
        ("cleaning_comparison", 6.42, 6.75, 1.15, 1.0),
    ]
    fits = []
    diagnostics = []
    shape_data = []
    for label, start, end, asc, coal in scenarios:
        mask = (h >= start) & (h <= end)
        time = secs[mask]
        blocks = []
        for lo, hi in BANDS:
            weights = np.maximum(
                0,
                np.minimum(e[1:], np.log10(hi))
                - np.maximum(e[:-1], np.log10(lo)),
            ) / np.diff(e)

            def edge(diam):
                y = np.array(
                    [
                        np.interp(np.log(diam), np.log(d), ss)
                        for ss in spec[:, mask].T
                    ]
                )
                return (
                    y
                    / (np.log(10) * diam)
                    * np.interp(diam, GRID, g[:, 1])
                    * acid[mask]
                    * asc
                )

            n = weights @ counts[:, mask]
            transport = edge(lo) - edge(hi)
            coeff = (weights * basewall) @ counts[:, mask]
            gain = (daughter >= lo) & (daughter < hi)
            # Parent band membership shares each source-bin width at the edges.
            change = gain.astype(float) - weights[:, None] - weights[None, :]
            nc = counts[:, mask]
            coag = (
                0.5 * np.einsum("it,ij,jt->t", nc, kernel * change, nc) * coal
            )
            rhs = (
                cumulative_trapezoid(transport + coag - KD * n, time, initial=0)
                - n
            )
            exposure = cumulative_trapezoid(coeff, time, initial=0)
            y = (rhs - rhs.mean()) / n.mean()
            x = (exposure - exposure.mean()) / n.mean()
            blocks.append((x, y))
            if label == "primary":
                by_size = cumulative_trapezoid(
                    (weights * basewall)[:, None] * nc, time, axis=1, initial=0
                )
                shape_data.append(
                    (
                        (by_size - by_size.mean(axis=1, keepdims=True))
                        / n.mean(),
                        y,
                    )
                )
            estimate = x @ y / (x @ x)
            fits.append(
                dict(
                    window=label,
                    lo_nm=lo,
                    hi_nm=hi,
                    first_hour=h[mask][0],
                    last_hour=h[mask][-1],
                    n_spectra=len(n),
                    scale=estimate,
                    number_start=n[0],
                    number_end=n[-1],
                    integrated_growth_net=np.trapezoid(transport, time),
                    integrated_coag_net=np.trapezoid(coag, time),
                    integrated_dilution=np.trapezoid(-KD * n, time),
                )
            )
        x, y = (np.concatenate(a) for a in zip(*blocks))
        diagnostics.append(
            dict(window=label, scale=x @ y / (x @ x), n_spectra=len(time))
        )
    shape_rows = []
    for leave in [-1, 0, 1, 2, 3]:
        for kind in ["native", "tilted", "constant"]:

            def solve(delta):
                factor = (
                    np.full_like(d, float(wall(10.0, eddy))) / basewall
                    if kind == "constant"
                    else (d / 10.0) ** delta
                )
                pairs = [(factor @ ex, y) for ex, y in shape_data]
                x, y = (
                    np.concatenate(a)
                    for a in zip(
                        *[b for i, b in enumerate(pairs) if i != leave]
                    )
                )
                scale = x @ y / (x @ x)
                loss = np.mean((scale * x - y) ** 2)
                test = pairs[leave] if leave >= 0 else (x, y)
                return (
                    loss,
                    scale,
                    np.mean((scale * test[0] - test[1]) ** 2) ** 0.5,
                )

            delta = (
                minimize_scalar(
                    lambda a: solve(a)[0], bounds=(-2, 2), method="bounded"
                ).x
                if kind == "tilted"
                else 0.0
            )
            loss, scale, test = solve(delta)
            shape_rows.append(
                dict(
                    leave_band=leave,
                    law=kind,
                    tilt=delta,
                    scale_at10nm=scale,
                    fit_rms=loss**0.5,
                    test_rms=test,
                )
            )
    write("wall_shape.csv", shape_rows)
    write("band_balance.csv", fits)
    write("wall_fits.csv", diagnostics)
    center = diagnostics[0]["scale"]
    assert center > 0
    # Calibrate native eddy coefficient at 10 nm, then use the native equation at every diameter.
    target = brentq(
        lambda x: float(wall(10.0, x)) - center * float(wall(10.0, eddy)),
        1e-9,
        2000.0,
    )
    profile = []
    for d0 in [2.9, 3.2, 5.0, 9.7, 10.0, 15.0, 20.0, 30.0, 40.0, 63.5]:
        old = np.interp(d0, GRID, g[:, 4])
        profile.append(
            dict(
                d_nm=d0,
                cold_law_s=0.001 / d0,
                old_nominal_s=old,
                old_fitted_s=old * 0.5,
                native_nominal_s=float(wall(d0, eddy)),
                native_dark_s=float(wall(d0, target)),
                dilution_s=KD,
            )
        )
    write("wall_profiles.csv", profile)
    meta = dict(
        equivalent_radius_m=RADIUS,
        eddy_nominal_s=eddy,
        eddy_dark_s=target,
        scale_10nm=center,
        first_cleaning_hour=6.3591671,
        band_edges_nm=BANDS,
    )
    (HERE / "calibration.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)
    print(diagnostics)


if __name__ == "__main__":
    fit_balance()
