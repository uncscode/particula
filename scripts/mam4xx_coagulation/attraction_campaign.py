"""Compare modal closure, size sections and stochastic particle histories."""

import time
from functools import lru_cache

import numpy as np
from scripts.mam4xx_coagulation.attraction_particles import ensemble
from scripts.mam4xx_coagulation.modal_attraction import (
    HAMAKER,
    HIST_EDGES,
    HOURS,
    RHO,
    TEMP,
    P,
    distribution,
    kernel_grid,
    solve_modes,
)
from scripts.mam4xx_coagulation.neutral_geometry import neutral_geometry
from scripts.mam4xx_coagulation.regime_box import evolve


@lru_cache(maxsize=16)
def laws(resolution, material):
    """Join native supported rates to the qualified neutral geometry tail."""
    edges, bare, supported, outside = kernel_grid(resolution, material)
    diameter = np.sqrt(edges[:-1] * edges[1:])
    independent = neutral_geometry(diameter, material, TEMP, P, RHO)
    discrepancy = np.max(
        np.abs(independent[~outside] / supported[~outside] - 1)
    )
    if discrepancy > 0.015:
        raise ValueError(
            f"Independent supported-domain check failed: {discrepancy}"
        )
    return (
        edges,
        outside,
        dict(
            bare=bare,
            supported=supported,
            complete=np.where(outside, independent, supported),
        ),
        float(discrepancy),
    )


def sections(regime, widths, coarse, law, resolution, material):
    """Run the same collision experiment on the independently evolving grid."""
    modes = distribution(regime, widths, coarse)

    def combined(diameter):
        from scripts.mam4xx_coagulation.regime_rates import tables

        _, _, _, _, _, _, _, full, support, *_ = tables(
            resolution, 0, TEMP, material, 5.9
        )
        direct = neutral_geometry(diameter, material, TEMP, P, RHO)
        return np.where(support, full, direct)

    r = evolve(
        regime,
        "neutral" if law == "bare" else "dispersion",
        per_decade=resolution,
        cap=0,
        hamaker=material,
        modes=modes,
        neutral_kernel=combined if law == "complete" else None,
    )
    return {
        k: r[k]
        for k in (
            "uf_cm3",
            "above_cm3",
            "volume_above_100_um3_cm3",
            "budget",
            "diameter_nm",
            "spectrum_cm3",
        )
    }


def deterministic(report, regime, widths, coarse, modes):
    """Run numerical and material controls for one initial spectrum."""
    for material in (HAMAKER, HAMAKER / 2, 2 * HAMAKER):
        for resolution in (24, 48, 96) if material == HAMAKER else (48,):
            edges, outside, models, overlap = laws(resolution, material)
            for law, kernel in models.items():
                if material != HAMAKER and law != "complete":
                    continue
                row = solve_modes(modes, kernel, edges, outside)
                row.update(
                    regime=regime,
                    widths=widths,
                    coarse=coarse,
                    law=law,
                    resolution=resolution,
                    hamaker=material,
                    inputs=modes.tolist(),
                    overlap_error=overlap,
                )
                report["modal"].append(row)
        for resolution in (8, 16, 32) if material == HAMAKER else (16,):
            for law in ("bare", "supported", "complete"):
                if material != HAMAKER and law != "complete":
                    continue
                row = sections(
                    regime,
                    widths,
                    coarse,
                    law,
                    resolution,
                    material,
                )
                row.update(
                    regime=regime,
                    widths=widths,
                    coarse=coarse,
                    law=law,
                    resolution=resolution,
                    hamaker=material,
                )
                report["sectional"].append(row)


def run():
    """Run polluted first, then marine, including paired material controls."""
    ed, _, zero, err = laws(24, 0)
    np.testing.assert_allclose(zero["complete"], zero["bare"], rtol=2e-5)
    _, _, _, overlap = laws(24, HAMAKER)
    d = np.sqrt(ed[:-1] * ed[1:])
    ref = neutral_geometry(d, HAMAKER, TEMP, P, RHO, 96)
    original = neutral_geometry(d, HAMAKER, TEMP, P, RHO)
    refine = float(np.max(abs(original / ref - 1)))
    if refine > 0.01:
        raise ValueError(
            "Neutral geometry radial/energy quadrature failed refinement"
        )
    report = dict(
        hours=HOURS.tolist(),
        edges_nm=HIST_EDGES.tolist(),
        nominal_hamaker_J=HAMAKER,
        temperature_K=TEMP,
        pressure_Pa=P,
        zero_gpu_relative_error=err,
        overlap_relative_error=overlap,
        geometry_refinement_relative_error=refine,
        modal=[],
        sectional=[],
        monte_carlo=[],
    )
    for regime in ("polluted", "marine"):
        start = time.monotonic()
        for widths in ("host", "field"):
            for coarse in (False, True):
                modes = distribution(regime, widths, coarse)
                deterministic(report, regime, widths, coarse, modes)
                # A 5 cm³ polluted box contains exactly 27 coarse particles.
                # All fine and coarse counts are integral at both box sizes.
                base_box = 10.0
                if regime == "polluted":
                    base_box = 5.0 if coarse else 0.05
                for resolution, nruns, scale, first_seed in [
                    (24, 24, 1, 1401),
                    (48, 16, 2, 1901),
                ]:
                    edges, _, models, _ = laws(resolution, HAMAKER)
                    for law in ("bare", "complete"):
                        box_cm3 = base_box * scale
                        row = ensemble(
                            modes,
                            edges,
                            models[law],
                            box_cm3,
                            range(first_seed, first_seed + nruns),
                        )
                        row.update(
                            regime=regime,
                            widths=widths,
                            coarse=coarse,
                            law=law,
                            resolution=resolution,
                        )
                        report["monte_carlo"].append(row)
        print(
            f"{regime} completed in {time.monotonic() - start:.1f} s",
            flush=True,
        )
    return report
