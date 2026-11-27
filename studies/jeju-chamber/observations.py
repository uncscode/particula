"""Retrieve the versioned Jeju inputs and align spectra and chamber forcing."""

import hashlib
import json
import os
import urllib.request
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np
import pandas as pd
import particula as par

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ["JEJU_RAW"])
MANIFEST = json.loads((HERE / "source-files.json").read_text())
START = {1: "15:50", 2: "11:00", 3: "16:00", 4: "13:40", 5: "16:50", 6: "15:00"}


def source(name):
    f = next(i for i in MANIFEST["files"] if i["name"] == name)
    p = RAW / f"particula-park-{f['id']}"

    def verify(path):
        with path.open("rb") as data:
            checksum = hashlib.file_digest(data, "md5").hexdigest()
        if checksum != f["computed_md5"]:
            raise ValueError(f"Figshare checksum mismatch for {name}: {path}")

    if p.exists():
        verify(p)
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=RAW, delete=False) as download:
            tmp = Path(download.name)
        try:
            urllib.request.urlretrieve(f["download_url"], tmp)
            verify(tmp)
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    return p


def sat(T):
    # Murphy & Koop (2005), equilibrium pressure in Pa, Kelvin.
    ice = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)
    water = np.exp(
        54.842763
        - 6763.22 / T
        - 4.210 * np.log(T)
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
    )
    return ice, water


CPI_EDGES = np.array([0.0, 15.0, 25.0, 40.0, 60.0, 80.0, 120.0, 200.0, np.inf])


def extract():
    runs = []
    for exp in range(1, 7):
        opc = pd.read_csv(
            source(f"OPC_exp{exp}.csv"), index_col=0, parse_dates=[0]
        )
        temp = pd.read_csv(
            source(f"Chamber_P,T_exp{exp}.csv"), index_col=0, parse_dates=[0]
        ).sort_index()
        origin = pd.Timestamp(f"{opc.index[0].date()} {START[exp]}")
        sec = (opc.index - origin).total_seconds().astype(int).to_numpy()
        diameter = opc.columns.astype(float).to_numpy()
        log_edge = np.r_[
            np.log(diameter[0]) - np.log(diameter[1] / diameter[0]) / 2,
            (np.log(diameter[:-1]) + np.log(diameter[1:])) / 2,
            np.log(diameter[-1]) + np.log(diameter[-1] / diameter[-2]) / 2,
        ]
        edge = np.exp(log_edge)
        widths = np.diff(log_edge) / np.log(10)
        rows = opc.to_numpy() * widths
        assert np.isfinite(rows).all() and rows.min() >= 0
        temp.index = (temp.index - origin).total_seconds().astype(int)
        nmin = int(sec.max() // 60 + 1)
        grid = np.arange(nmin * 60)
        valid_stop = min(temp.index.max(), sec.max())
        temp = temp.reindex(grid).interpolate(limit_direction="both")
        vapor = np.full(grid.size, np.nan)
        if exp < 6:
            gas = pd.read_csv(source(f"LICOR_H2O_exp{exp}.csv")).dropna(
                subset=["Date", "Time"]
            )
            gas.index = (
                (pd.to_datetime(gas.Date + " " + gas.Time.str[:8]) - origin)
                .dt.total_seconds()
                .astype(int)
            )
            # mmol/m³ × g/mol / 1000 = g/m³.
            vapor = (
                (gas.iloc[:, -1] * 0.01801528)
                .reindex(grid)
                .interpolate(limit=15, limit_direction="both")
                .to_numpy()
            )
            assert np.isfinite(vapor[: valid_stop + 1]).mean() > 0.94
        cpi = pd.read_csv(source(f"CPI_exp{exp}.csv")).dropna(
            subset=["Length", "Width"]
        )
        utc = cpi.iloc[:, 1].str.replace(" ", "").str.split(".").str[0]
        stamps = pd.to_datetime(
            cpi.year.astype(int).astype(str) + ":" + utc,
            format="%Y:%m:%d:%H:%M:%S",
            errors="coerce",
        ) + pd.Timedelta(hours=9)
        csec = (stamps - origin).dt.total_seconds().to_numpy()
        counts = np.array(
            [
                np.histogram(
                    cpi.Length[
                        (csec >= m * 60)
                        & (csec < (m + 1) * 60)
                        & (cpi.Length >= 10)
                    ],
                    bins=CPI_EDGES,
                )[0]
                for m in range(nmin)
            ]
        )
        # Clock-aligned, forward nonoverlapping one-minute windows.
        op = np.stack(
            [
                rows[(sec >= m * 60) & (sec < (m + 1) * 60)].mean(axis=0)
                if np.sum((sec >= m * 60) & (sec < (m + 1) * 60)) >= 55
                and (m + 1) * 60 <= valid_stop + 1
                else np.full(len(diameter), np.nan)
                for m in range(nmin)
            ]
        )
        initial = rows[(sec >= 0) & (sec < 60)].mean(axis=0)
        runs.append(
            dict(
                temp=temp[["Tair", "Tair_minimum", "Twall"]].to_numpy()
                + 273.15,
                pressure=temp.Pressure_hPa.to_numpy() * 100,
                vapor=vapor,
                observed=op,
                cpi=counts,
                initial=initial,
                valid_stop=valid_stop,
                origin=str(origin),
                diameter=diameter,
                edges=edge,
                widths=widths,
            )
        )
    return runs


def forcing(runs):
    steps = max(len(r["temp"]) for r in runs)
    all_force = np.zeros((6, steps, 13), np.float32)
    for i, r in enumerate(runs):
        mean, minimum, _ = r["temp"].T
        pressure = r["pressure"]
        n = len(mean)
        rho = pressure / (287.05 * mean)
        ie, we = sat(mean)
        ice = ie / (461.52 * mean) * 1000
        water = we / (461.52 * mean) * 1000
        v = r["vapor"].copy()
        finite = np.flatnonzero(np.isfinite(v))
        if len(finite):
            v = np.interp(np.arange(n), finite, v[finite])
        else:
            v = 0.7 * water
        diff = 2.11e-5 * (mean / 273.15) ** 1.94 * 101325 / pressure
        k = par.gas.get_thermal_conductivity(mean)
        # Diameter squared growth per unit S-1, in µm²/s. Native Particula
        # transport evaluated at unit radius with measured air properties.
        beta = []
        for e, latent, rhow in [(we, 2.50e6, 1000), (ie, 2.834e6, 917)]:
            mdot = par.dynamics.get_mass_transfer_rate(
                e, 4 * np.pi * diff, mean, 0.01801528
            ) / (4 * np.pi * rhow)
            heat = 1 + diff * e * latent / (461.52 * mean**2 * k) * (
                latent / (461.52 * mean) - 1
            )
            beta.append(8e12 * mdot / heat)
        visc = par.gas.get_dynamic_viscosity(mean)
        gravity = float(
            par.particles.get_particle_settling_velocity(
                0.5e-6, 1000.0, 1.0, 1.0, 9.80665
            )
        )
        stokes = gravity / visc
        all_force[i, :n] = np.column_stack(
            (
                mean,
                rho,
                water,
                ice,
                v,
                *beta,
                stokes,
                1e-6 * rho / visc,
                np.ones(n),
                273.15 - minimum,
                np.zeros(n),
                np.arange(n) <= r["valid_stop"],
            )
        )
    return all_force
