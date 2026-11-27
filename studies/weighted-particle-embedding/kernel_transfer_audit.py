"""Test the unchanged Brownian network against independent Particula laws."""

import argparse
from pathlib import Path

import numpy as np
import torch
from kernel_transfer import (
    AIR_DENSITY,
    DISSIPATION,
    MATERIAL_DENSITY,
    MECHANISMS,
    KernelBox,
    fit_rates,
    radii_from_quantiles,
)
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.coagulation.charged_dimensional_kernel import (
    get_coulomb_kernel_gopalakrishnan2012_via_system_state,
)
from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state,
)
from particula.dynamics.coagulation.turbulent_shear_kernel import (
    get_turbulent_shear_kernel_st1956_via_system_state,
)
from stage3_evidence import write


def targets(r, m, z):
    """Sample matching entries from independent complete CPU kernel matrices."""
    output = {name: [] for name in MECHANISMS}
    for a in range(0, len(r), 64):
        rad, mass, charge = (p[a : a + 64] for p in (r, m, z))
        i, j = np.arange(0, len(rad), 2), np.arange(1, len(rad), 2)
        brown = get_brownian_kernel_via_system_state(
            rad, mass, 293.0, 101325.0
        )[i, j]
        charged = get_coulomb_kernel_gopalakrishnan2012_via_system_state(
            rad, mass, charge, 293.0, 101325.0
        )[i, j]
        settle = get_sedimentation_kernel_sp2016_via_system_state(
            rad, np.full_like(rad, MATERIAL_DENSITY), 293.0, 101325.0, False
        )[i, j]
        shear = get_turbulent_shear_kernel_st1956_via_system_state(
            rad, DISSIPATION, 293.0, AIR_DENSITY
        )[i, j]
        for name, rate in zip(
            MECHANISMS,
            (brown, charged, settle, shear, brown + settle + shear),
            strict=True,
        ):
            output[name].extend(rate)
    return {name: np.asarray(values) for name, values in output.items()}


def audit(folder):
    """Hold the population and sampled pairs fixed across every comparison."""
    torch.set_num_threads(1)
    fits = fit_rates(folder, "cpu")
    rng = np.random.default_rng(87031)
    u = rng.random(8192)
    radii = radii_from_quantiles(u)
    masses = 4 * np.pi / 3 * MATERIAL_DENSITY * radii**3
    charges = rng.choice([-1, 1], size=len(u))
    ref = targets(radii, masses, charges)
    r, m, z = map(torch.as_tensor, (radii, masses, charges))
    i = torch.arange(0, len(r), 2)
    rows = []
    with torch.inference_mode():
        for law in MECHANISMS:
            for method in [
                "Physical formula",
                "Unchanged Brownian net",
                "Kernel-specific neural",
            ]:
                actual = (
                    "Brownian" if method == "Unchanged Brownian net" else law
                )
                charge = "bipolar1" if actual == "Charged" else "neutral"
                use_net = method != "Physical formula"
                model = KernelBox(
                    2,
                    32,
                    1e12,
                    charge,
                    "cpu",
                    307,
                    1,
                    "learned"
                    if use_net and actual in ("Brownian", "Charged", "Combined")
                    else "physical",
                )
                model.law = actual
                model.fit = fits.get(actual) if use_net else None
                pred = model.rates(r, m, z, (i, i + 1)).numpy()
                error = 100 * np.abs(pred - ref[law]).sum() / ref[law].sum()
                if method == "Physical formula" and error > 1e-6:
                    raise ValueError(
                        f"{law} differs from native kernel by {error}%"
                    )
                rows.append(
                    dict(
                        kernel=law,
                        method=method,
                        pairs=len(i),
                        rate_error_pct=error,
                    )
                )
    write(folder / "rate-transfer.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "results/kernel-transfer",
    )
    folder = parser.parse_args().folder
    folder.mkdir(parents=True, exist_ok=True)
    audit(folder)
