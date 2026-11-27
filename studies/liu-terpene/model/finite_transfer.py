# %%
"""Refit each cyclic aerosol before transferring across vapor sink and loading."""

import ctypes
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from chamber import (
    CSTAR,
    CYCLIC,
    DENSITY,
    GAS_D,
    HERE,
    MFP,
    MW,
    REPORTED,
    SIZE,
    TEMP,
    Scenario,
    finite_sink,
)
from scipy.optimize import least_squares
from scipy.special import softmax


def run(out):
    out.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(HERE.parent / "sources" / "observations.csv")
    c_contig = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    )
    with tempfile.TemporaryDirectory() as build:
        lib = Path(build) / "finite.so"
        subprocess.run(
            [
                "c++",
                "-std=c++20",
                "-O2",
                "-fPIC",
                "-shared",
                str(HERE / "finite_box.cpp"),
                "-o",
                str(lib),
            ],
            check=True,
        )
        kernel = ctypes.CDLL(str(lib)).integrate
        kernel.argtypes = (
            [c_contig, c_contig, ctypes.c_int]
            + [ctypes.c_double] * 9
            + [c_contig]
        )
        kernel.restype = None

        def predict(f, name, t, diam, num, tau, step=30.0, speed=1.0):
            t, inv = np.unique(np.asarray(t, dtype="d"), return_inverse=True)
            r0 = diam * 0.5e-9
            rho = DENSITY[name]
            m0 = 4 * np.pi * r0**3 * rho / 3
            kr = 2 * 0.03 * MW / (8.314462618 * TEMP * rho)
            a = np.zeros(len(t))
            kernel(
                f,
                t,
                len(t),
                m0,
                r0,
                MFP,
                GAS_D,
                kr,
                num * 1e6,
                tau,
                step,
                speed,
                a,
            )
            return a[inv]

        transfer, bounds, checks, weights = [], [], [], []
        for (name, rh), native in data.groupby(["precursor", "rh"]):
            if name != CYCLIC[name]:
                continue
            heldname = "ocimene" if name == "pinene" else "farnesene"
            held = data[(data.precursor == heldname) & (data.rh == rh)]
            prior = REPORTED[name, rh]
            for tau in [600.0, 3600.0, 86400.0]:
                for nref in [100.0, 200.0]:

                    def resid(z):
                        fractions = softmax(np.r_[z, 0.0])
                        return np.r_[
                            predict(
                                fractions,
                                name,
                                native.time_min,
                                SIZE[name],
                                nref,
                                tau,
                            )
                            - native.vfr,
                            0.055 * (fractions - prior),
                        ]

                    opt = least_squares(
                        resid,
                        np.log(prior[:-1] / prior[-1]),
                        bounds=(-22.0, 22.0),
                        ftol=1e-8,
                        xtol=1e-8,
                        gtol=1e-8,
                        max_nfev=150,
                    )
                    f = softmax(np.r_[opt.x, 0.0])
                    weights.extend(
                        (name, rh, nref, tau, c, x)
                        for c, x in zip(CSTAR, f, strict=True)
                    )
                    control = predict(
                        f, name, native.time_min, SIZE[name], nref, tau
                    )
                    control_rmse = np.mean((control - native.vfr) ** 2) ** 0.5
                    control_max = np.max(np.abs(control - native.vfr))
                    for ngas in [100.0, 200.0]:
                        t = np.unique(np.r_[held.time_min, 200.0])
                        x = predict(f, heldname, t, SIZE[heldname], ngas, tau)
                        oracle, budget = finite_sink(
                            f,
                            heldname,
                            rh,
                            SIZE[heldname],
                            t,
                            Scenario(),
                            tau,
                            ngas,
                        )
                        fine = predict(
                            f, heldname, t, SIZE[heldname], ngas, tau, step=3.0
                        )
                        checks.append(
                            (
                                name,
                                rh,
                                nref,
                                ngas,
                                tau,
                                np.max(abs(x - fine)),
                                np.max(abs(x - oracle)),
                                budget,
                            )
                        )
                        for tm, obs in zip(held.time_min, held.vfr):
                            pred = x[t == tm][0]
                            transfer.append(
                                (
                                    name,
                                    heldname,
                                    rh,
                                    nref,
                                    ngas,
                                    tau,
                                    control_rmse,
                                    control_max,
                                    tm,
                                    obs,
                                    pred,
                                    pred - obs,
                                )
                            )
                        bounds.append(
                            (
                                name,
                                heldname,
                                rh,
                                nref,
                                ngas,
                                tau,
                                control_rmse,
                                control_max,
                                predict(
                                    f, name, [200.0], SIZE[name], nref, tau
                                )[0],
                                x[t == 200.0][0],
                            )
                        )
        for fn, cols, rows in [
            (
                "finite_weights.csv",
                "cyclic rh cyclic_number_cm3 wall_lifetime_s cstar_ug_m3 fitted_fraction",
                weights,
            ),
            (
                "finite_held.csv",
                "cyclic acyclic rh cyclic_number_cm3 acyclic_number_cm3 wall_lifetime_s cyclic_rmse cyclic_max_abs_error time_min observed_vfr cyclic_prediction signed_error",
                transfer,
            ),
            (
                "finite_200min.csv",
                "cyclic acyclic rh cyclic_number_cm3 acyclic_number_cm3 wall_lifetime_s cyclic_rmse cyclic_max_abs_error cyclic_native_vfr acyclic_prediction",
                bounds,
            ),
            (
                "finite_checks.csv",
                "cyclic rh cyclic_number_cm3 acyclic_number_cm3 wall_lifetime_s max_step_shift_vfr max_difference_from_ode_vfr max_species_budget_error",
                checks,
            ),
        ]:
            pd.DataFrame(rows, columns=cols.split()).to_csv(
                out / fn, index=False, float_format="%.10g"
            )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    run(p.parse_args().out)
