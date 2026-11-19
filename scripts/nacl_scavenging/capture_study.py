"""Resolve dispersion and polarization within the NaCl chamber retrieval.

Run with native CUDA and OPENBLAS_NUM_THREADS=1. Rates in the chamber solve
are cm³/s. The pair evaluator uses SI. Both particle sizes are diameters.
"""

import json
from functools import lru_cache

import numpy as np
import warp as wp
from instrument import HERE, STATES, TARGETS, save
from particula import particles
from particula.dynamics import get_van_der_waals_kernel_via_system_state
from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    MAX_RADIUS_RATIO,
)
from particula.gpu.dynamics.van_der_waals import prepare_capture
from scripts.coagulation_validation.cuda_accuracy import _evaluate
from steady import evaluate
from study import DENSITY, PRESSURE, TEMP, archive_table, revision

HAMAKER = 8.9e-20
EPSILON = 5.9
MATERIALS = {
    "bare": (0.0, 1.0),
    "dispersion_only": (HAMAKER, 1.0),
    "polarization_only": (0.0, EPSILON),
    "full": (HAMAKER, EPSILON),
    "half_hamaker": (HAMAKER / 2, EPSILON),
    "double_hamaker": (HAMAKER * 2, EPSILON),
}


@lru_cache(maxsize=len(MATERIALS))
def material(a, eps):
    """Prepare each constant material table once per campaign."""
    return prepare_capture(a, eps, "cuda:0")


def rate_models(small, large, tracer_states, collectors, native, extra):
    """Average actual signed pair rates over the steady collector inventory."""
    results = {
        name: [] for name in (*MATERIALS, "full_zero_tail", "full_legacy_tail")
    }
    metrics = {
        key: []
        for key in (
            "excluded_pairs",
            "excluded_size",
            "excluded_unlike",
            "tail_bare_cm3_s",
            "full_neutral_neutral_cm3_s",
            "full_tracer_neutral_cm3_s",
            "full_collector_neutral_cm3_s",
            "full_both_charged_cm3_s",
        )
    }
    r = small[:, None, None] * 0.5e-9
    s = large[None, :, None] * 0.5e-9
    diff = particles.get_diffusion_coefficient_via_system_state(
        r, TEMP, PRESSURE
    )
    diff = diff + particles.get_diffusion_coefficient_via_system_state(
        s, TEMP, PRESSURE
    )
    mass_factor = 4 * np.pi / 3 * DENSITY
    coupling = COULOMB_LENGTH_T / (TEMP * (r + s))
    z = STATES[None, None, :]
    size_limit = np.maximum(r / s, s / r) > MAX_RADIUS_RATIO

    def weighted(values):
        return (values * collectors[None, :, :]).sum(axis=(1, 2))

    for charge in tracer_states:
        unlike_limit = coupling * charge * z < -0.5
        outside = (
            size_limit
            | unlike_limit
            | (coupling * np.maximum(charge**2, z**2) > 60)
        )
        # Apply the same support and same completion rule in all four controls.
        tail = np.stack([native[int(charge * b)] for b in STATES], axis=-1)
        for key, values in [
            ("excluded_pairs", outside),
            ("excluded_size", size_limit),
            ("excluded_unlike", unlike_limit),
            ("tail_bare_cm3_s", outside * tail),
        ]:
            metrics[key].append(weighted(values))
        cols = np.broadcast_arrays(
            r, s, mass_factor * r**3, mass_factor * s**3, charge, z, diff, TEMP
        )
        inputs = np.column_stack([c[~outside] for c in cols])
        device = wp.array(inputs, dtype=wp.float64, device="cuda:0")
        out = wp.empty(len(inputs), dtype=wp.float64, device="cuda:0")
        for name, (a, eps) in MATERIALS.items():
            wp.launch(
                _evaluate,
                len(inputs),
                inputs=[material(a, eps), device, out],
                device="cuda:0",
            )
            values = out.numpy() * 1e6
            if not np.all(np.isfinite(values) & (values >= 0)):
                raise ValueError("capture evaluator rejected a scoped pair")
            rates = tail.copy()
            rates[~outside] = values
            results[name].append(weighted(rates))
            if name == "full":
                for key, mask in [
                    ("full_neutral_neutral_cm3_s", (charge == 0) & (z == 0)),
                    ("full_tracer_neutral_cm3_s", (charge == 0) & (z != 0)),
                    ("full_collector_neutral_cm3_s", (charge != 0) & (z == 0)),
                    ("full_both_charged_cm3_s", (charge != 0) & (z != 0)),
                ]:
                    metrics[key].append(
                        weighted(np.where(~outside & mask, rates, 0))
                    )
                results["full_zero_tail"].append(
                    weighted(np.where(outside, 0.0, rates))
                )
                results["full_legacy_tail"].append(
                    weighted(rates + outside * extra[:, :, None])
                )
    return (
        [(key, np.column_stack(cols)) for key, cols in results.items()],
        {key: np.column_stack(cols) for key, cols in metrics.items()},
    )


def pair_controls():
    """Independent public CPU pair evaluation at the five nominal diameters."""
    rows = []
    for small in TARGETS:
        radii = np.array([small, 100.0]) * 0.5e-9
        masses = 4 * np.pi / 3 * DENSITY * radii**3
        for charges in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            row = dict(
                target_nm=small,
                tracer_charge=charges[0],
                collector_charge=charges[1],
            )
            for name, (a, eps) in MATERIALS.items():
                row[name] = (
                    1e6
                    * get_van_der_waals_kernel_via_system_state(
                        radii,
                        masses,
                        charges,
                        TEMP,
                        PRESSURE,
                        a,
                        eps,
                        exclude_self_pairs=True,
                    )[0, 1]
                )
            rows.append(row)
    return rows


def main():
    """Save matched retrievals and out-of-domain contributions."""
    if not wp.is_cuda_available():
        raise RuntimeError("The native pair-rate sweep requires CUDA")
    measured = archive_table("Fig6_CoagCoef")
    observed = dict(measured[np.isfinite(measured[:, 0]), :2])
    scan = dict(
        analyzer=0.3, channel_centers=(3.59, 3.72, 3.85, 4.0, 4.14, 4.29, 4.45)
    )
    cases = [
        ("reference", TARGETS, {}),
        ("refined", TARGETS, dict(parts=128, stride=4)),
        ("archived_4nm_scan", (4.0,), scan),
        ("single_size", TARGETS, dict(ideal=True)),
        ("ions_elevated", (4.0,), dict(ion=100)),
        ("tube_transmission", (4.0,), dict(tube=1.0)),
    ]
    rows = []
    for case, sizes, options in cases:
        for d in sizes:
            records = evaluate(d, case, pair_models=rate_models, **options)
            for row in records:
                if row["model"] == "full":
                    total = sum(
                        row[k]
                        for k in row
                        if k.startswith("full_") and k.endswith("_cm3_s")
                    )
                    np.testing.assert_allclose(
                        total + row["tail_bare_cm3_s"],
                        row["true_capture_cm3_s"],
                        rtol=1e-11,
                        atol=0,
                    )
                if row["model"] != "full":
                    for key in row:
                        if key.startswith("full_") or key == "tail_bare_cm3_s":
                            row[key] = None
                rows.append(
                    {
                        key: row[key]
                        for key in (
                            "case",
                            "model",
                            "target_nm",
                            "apparent_cm3_s",
                            "true_capture_cm3_s",
                            "excluded_pairs",
                            "excluded_size",
                            "excluded_unlike",
                            "tail_bare_cm3_s",
                            "full_neutral_neutral_cm3_s",
                            "full_tracer_neutral_cm3_s",
                            "full_collector_neutral_cm3_s",
                            "full_both_charged_cm3_s",
                        )
                    }
                    | {"measured_cm3_s": observed[d]}
                )
    save("capture-steady.csv", rows)
    save("capture-pairs.csv", pair_controls())
    metadata = dict(
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        revision=revision(),
        temperature_K=TEMP,
        pressure_Pa=PRESSURE,
        density_kg_m3=DENSITY,
        hamaker_J=HAMAKER,
        relative_permittivity=EPSILON,
        rows=len(rows),
        pair_kernel_unit="cm3/s",
        collector_number_cm3=1e4,
    )
    (HERE / "capture-run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
