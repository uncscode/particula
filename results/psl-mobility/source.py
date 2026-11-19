# %% [markdown]
# # A dry PSL standard checks the diameter assigned by a mobility analyzer
#
# Kim et al. (2005) measured the electrical mobility of dry polystyrene
# latex (PSL) spheres as they varied the air pressure. Here we reconstruct
# the mobility from each published slip factor and solve for the diameter
# that Particula would assign to the same observation. The second panel
# asks how far a synthetic calibration at 100 nm transfers across sizes.
# [Read the mobility sizing guide](../../../guide/processes/psl-mobility.md).

# %% [markdown]
# ## Keep the gas and charge conventions fixed
#
# Use all 49 rows of Tables 7a–c. The diameter is in nm, pressure in kPa,
# and temperature in K. Kim's reference air has a mean free path of 67.3 nm
# at 296.15 K and 101.3 kPa. Particula's mechanical mobility takes radius
# in metres. Multiply by charge magnitude to obtain electrical mobility.

# %%
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from scipy.constants import elementary_charge
from scipy.optimize import bisect, brentq

DATA = Path("docs/data/transport/kim2005-psl-slip.csv")
TEMP_K, PRESSURE_KPA, PATH_NM = 296.15, 101.3, 67.3
slip = par.particles.get_cunningham_slip_correction


def observations():
    """Read the original table order from the checkout or extracted bundle."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / DATA).is_file():
                with (parent / DATA).open() as stream:
                    return list(csv.DictReader(stream))
    raise FileNotFoundError(f"Extract the example bundle containing {DATA}")


def electrical_mobility(diameter_nm, factor, viscosity, charges):
    """Return mobility magnitude in square metres per volt second."""
    mechanical = par.particles.get_aerodynamic_mobility(
        diameter_nm * 0.5e-9, factor, viscosity
    )
    return charges * elementary_charge * mechanical


def fitted_diameter(target, path_nm, viscosity, charges, solver=brentq):
    """Solve the mobility equation over a bracket expressed in nm."""

    def residual(d):
        predicted = electrical_mobility(
            d, slip(2 * path_nm / d), viscosity, charges
        )
        return predicted / target - 1

    return solver(residual, 1.0, 2000.0, xtol=1e-10)


# %% [markdown]
# ## Each observation gives a slip difference and a sizing difference
#
# The printed Knudsen number gives the mean free path through λ = Kn D/2.
# Hold this path fixed while solving for diameter. Independently compute
# Kn from temperature and pressure using the paper's Eq. 11. A 1% mismatch
# flags rows for the comparison that uses Particula's default gas properties.


# %%
def compare(row):
    """Evaluate one reported mobility under matched and native gas conventions."""
    d, t, p, kn, observed = (
        float(row[key])
        for key in ("diameter_nm", "temperature_k", "pressure_kpa", "kn", "c")
    )
    charges = 2 if row["table"] == "7b" else 1
    mu = 1.83245e-5 * (t / TEMP_K) ** 1.5 * (TEMP_K + 110.4) / (t + 110.4)
    path = d * kn / 2
    target = electrical_mobility(d, observed, mu, charges)
    recovered = fitted_diameter(target, path, mu, charges)
    check = fitted_diameter(target, path, mu, charges, bisect)
    roundtrip = electrical_mobility(
        recovered, slip(2 * path / recovered), mu, charges
    )
    np.testing.assert_allclose(check, recovered, rtol=0, atol=1e-6)
    np.testing.assert_allclose(roundtrip / target, 1.0, rtol=1e-9, atol=0)
    paper_path = (
        PATH_NM
        * (t / TEMP_K)
        * (PRESSURE_KPA / p)
        * (1 + 110.4 / TEMP_K)
        / (1 + 110.4 / t)
    )
    consistent = abs(paper_path / path - 1) <= 0.01
    native_nm = None
    if consistent:
        native_path = 1e9 * par.gas.get_molecule_mean_free_path(
            temperature=t, pressure=p * 1000
        )
        native_nm = fitted_diameter(
            target, native_path, par.gas.get_dynamic_viscosity(t), charges
        )
    return dict(
        table=row["table"],
        row=int(row["row"]),
        diameter_nm=d,
        kn=kn,
        slip_error_pct=100 * (slip(kn) / observed - 1),
        recovered_nm=recovered,
        native_gas_nm=native_nm,
        pressure_consistent=consistent,
    )


# %% [markdown]
# ## One calibration point fixes the mobility scale at 100 nm
#
# Generate a second, synthetic set of mobilities from the authors' fitted
# slip law at the reference gas conditions. Compare two diameter inversions:
# the original Particula scale and a single scale factor matched at 100 nm.


# %%
def kim_slip(kn):
    """Return the dimensionless slip factor from Kim et al.'s fit."""
    return 1 + kn * (1.165 + 0.483 * np.exp(-0.997 / kn))


def simulate():
    """Return per-row inversions and a separate single-point calibration scan."""
    measured = [compare(row) for row in observations()]
    kn100 = 2 * PATH_NM / 100
    scale = kim_slip(kn100) / slip(kn100)
    scan = []
    for d in np.unique(
        np.r_[np.geomspace(20, 300, 181), 20, 50, 100, 200, 269, 300]
    ):
        target = electrical_mobility(
            d, kim_slip(2 * PATH_NM / d), 1.83245e-5, 1
        )
        initial = fitted_diameter(target, PATH_NM, 1.83245e-5, 1)
        calibrated = fitted_diameter(target / scale, PATH_NM, 1.83245e-5, 1)
        scan.append(
            dict(
                diameter_nm=d,
                size_error_pct=100 * (initial / d - 1),
                calibrated_size_error_pct=100 * (calibrated / d - 1),
            )
        )
    return dict(measured=measured, scan=scan, mobility_scale=scale)


# %% [markdown]
# ## Read the two panels at different levels
#
# Each point on the left compares one published observation with Particula.
# On the right the calibration curve crosses zero by construction at 100 nm.
# The shaded interval extends below the smallest measured Knudsen number.


# %%
@plt.style.context("particula.default")
def plot_results(result):
    """Show measured slip residuals and the transfer of a 100 nm calibration."""
    from particula import plotting

    fig, (left, right) = plotting.subplots(1, 2)
    for d, color, marker in [
        (19.9, plotting.COLORS["teal"], "o"),
        (100.7, plotting.COLORS["indigo"], "s"),
        (269, plotting.COLORS["rust"], "^"),
    ]:
        group = [r for r in result["measured"] if r["diameter_nm"] == d]
        left.scatter(
            [r["kn"] for r in group],
            [r["slip_error_pct"] for r in group],
            color=color,
            marker=marker,
            alpha=0.8,
            label=f"{d:g} nm",
        )
    left.set(
        xscale="log",
        xlabel="Published Knudsen number",
        ylabel="Slip factor difference (%)",
        title="Measured PSL slip",
    )
    plotting.legend(left, placement="inside", title="Diameter")
    scan = result["scan"]
    for key, label, color in [
        (
            "size_error_pct",
            "Original mobility scale",
            plotting.COLORS["indigo"],
        ),
        (
            "calibrated_size_error_pct",
            "Matched at 100 nm",
            plotting.COLORS["rust"],
        ),
    ]:
        right.plot(
            [r["diameter_nm"] for r in scan],
            [r[key] for r in scan],
            label=label,
            color=color,
            lw=2.3,
        )
    right.axvspan(
        2 * PATH_NM / min(r["kn"] for r in result["measured"]),
        300,
        color=".92",
        zorder=0,
    )
    right.plot(100, 0, "o", color=plotting.COLORS["rust"], ms=4)
    right.set(
        xscale="log",
        xlabel="Reference diameter (nm)",
        ylabel="Retrieved diameter difference (%)",
        title="Sizing at 101.3 kPa",
    )
    right.set_xticks(
        [20, 50, 100, 200, 300], labels=["20", "50", "100", "200", "300"]
    )
    plotting.legend(right)
    for ax in (left, right):
        ax.axhline(0, color=".4", lw=0.7)
    plotting.fit(fig)
    return fig


def run_example():
    """Run the two comparisons and show the reference-size result."""
    result = simulate()
    near_atmospheric = [
        r["recovered_nm"]
        for r in result["measured"]
        if r["table"] == "7a" and r["row"] in (1, 2)
    ]
    print(
        "100.7 nm reference at near-atmospheric pressure (nm):",
        np.round(near_atmospheric, 2),
    )
    plot_results(result)
    plt.show()
    return result


if __name__ == "__main__":
    run_example()

# %% [markdown]
# Particula assigns 102.02 and 102.03 nm to the two near-atmospheric
# 100.7 nm observations. In the synthetic transfer test, matching at 100 nm
# leaves a −0.46% shift at 20 nm. A second independently sized standard at
# the working pressure can test whether a calibration transfers. The
# [validation record](../../../contribute/validation/transport/psl-mobility.md)
# explains how to interpret the two pressure-inconsistent rows.
