"""Screen collision mechanisms in the steady NaCl chamber retrieval."""

import json

import numpy as np
import warp as wp
from capture_study import rate_models
from instrument import HERE, STATES, TARGETS, save
from particula import dynamics as dyn
from particula.dynamics import (
    get_coulomb_kernel_gopalakrishnan2012_via_system_state as gopalakrishnan,
)
from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state,
)
from steady import evaluate
from study import DENSITY, PRESSURE, TEMP, archive_table, revision

# Dry-air ideal gas density at the study's T and P.
AIR = PRESSURE / (287.05 * TEMP)
DISSIPATION = (0.001, 0.1, 1.0, 10.0)
CHARGED = {
    "hard_sphere": dyn.get_hard_sphere_kernel_via_system_state,
    "dyachkov": dyn.get_coulomb_kernel_dyachkov2007_via_system_state,
    "gatti": dyn.get_coulomb_kernel_gatti2008_via_system_state,
    "gopalakrishnan": gopalakrishnan,
    "chahl": dyn.get_coulomb_kernel_chahl2019_via_system_state,
}


def mechanical(radii):
    """Return geometric kernels in m³/s, using unit collection efficiency."""
    rho = np.full_like(radii, DENSITY)
    kernels = {
        "settling": get_sedimentation_kernel_sp2016_via_system_state(
            radii, rho, TEMP, PRESSURE, calculate_collision_efficiency=False
        )
    }
    for eps in DISSIPATION:
        kernels[f"shear_{eps:g}"] = (
            dyn.get_turbulent_shear_kernel_st1956_via_system_state(
                radii, eps, TEMP, AIR
            )
        )
    # Ayala et al. included settling in their turbulent collision kernel.
    # This illustrative cloud-model extrapolation uses zero imposed slip.
    kernels["ao2008_0.01"] = (
        dyn.get_turbulent_dns_kernel_ao2008_via_system_state(
            radii,
            rho,
            AIR,
            TEMP,
            re_lambda=100.0,
            relative_velocity=0.0,
            turbulent_dissipation=0.01,
        )
    )
    np.testing.assert_allclose(
        kernels["shear_10"], 100 * kernels["shear_0.001"], rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(kernels["settling"].diagonal(), 0.0)
    return kernels


def charged_rates(law, radii, masses, split, tracer_states, collectors):
    """Only the signed charge product enters the alternative Coulomb fits."""
    rates = {}
    for a in tracer_states:
        for b in STATES:
            product = int(a * b)
            if product in rates:
                continue
            z = np.r_[np.full(split, a), np.full(len(radii) - split, b)]
            if product == 0:
                z.fill(0)
            matrix = law(radii, masses, z, TEMP, PRESSURE)
            values = matrix[:split, split:] * 1e6
            if not np.all(np.isfinite(values) & (values >= 0)):
                raise ValueError(f"Invalid {law.__name__} cross-pair rate")
            rates[product] = values
    mixed = np.column_stack(
        [
            sum(
                rates[int(a * b)] @ collectors[:, j]
                for j, b in enumerate(STATES)
            )
            for a in tracer_states
        ]
    )
    neutral = rates[0] @ collectors.sum(axis=1)
    return mixed, neutral[:, None]


def chamber_models(small, large, tracer_states, collectors, native, extra):
    """Apply alternative thermal fits or additive mechanical rates."""
    baseline, _ = rate_models(
        small, large, tracer_states, collectors, native, extra
    )
    full = dict(baseline)["full"]
    results = [("full", full)]
    radii = np.r_[small, large] * 0.5e-9
    masses = 4 * np.pi / 3 * DENSITY * radii**3
    n = len(small)

    def average(matrix):
        return (matrix[:n, n:] @ collectors.sum(axis=1))[:, None] * 1e6

    additions = {
        key: average(value) for key, value in mechanical(radii).items()
    }
    for key, value in additions.items():
        results.append(("full_" + key, full + value))
    results.append(
        (
            "full_settling_shear_10",
            full + additions["settling"] + additions["shear_10"],
        )
    )
    results.append(
        (
            "fuchs_neutral",
            average(
                dyn.get_brownian_kernel_via_system_state(
                    radii, masses, TEMP, PRESSURE
                )
            ),
        )
    )
    for key, law in CHARGED.items():
        mixed, neutral = charged_rates(
            law, radii, masses, n, tracer_states, collectors
        )
        results.extend([(key, mixed), (key + "_neutral", neutral)])
    return results, {}


def main():
    """Record apparent changes and nominal additive rates separately."""
    if not wp.is_cuda_available():
        raise RuntimeError("Native CUDA is required for the capture baseline")
    measured = archive_table("Fig6_CoagCoef")
    obs = dict(measured[np.isfinite(measured[:, 0]), :2])
    rows, pairs = [], []
    for d in TARGETS:
        r = np.array([d, 100.0]) * 0.5e-9
        pairs.append(
            {
                "target_nm": d,
                **{k: m[0, 1] * 1e6 for k, m in mechanical(r).items()},
            }
        )
    for case, sizes, options in [
        ("reference", TARGETS, {}),
        ("refined", TARGETS, dict(parts=128, stride=4)),
        (
            "archived_4nm_scan",
            (4.0,),
            dict(
                analyzer=0.3,
                channel_centers=(3.59, 3.72, 3.85, 4.0, 4.14, 4.29, 4.45),
            ),
        ),
    ]:
        for d in sizes:
            records = evaluate(d, case, pair_models=chamber_models, **options)
            base = next(
                row["apparent_cm3_s"]
                for row in records
                if row["model"] == "full"
            )
            for row in records:
                value = row["apparent_cm3_s"]
                rows.append(
                    dict(
                        case=case,
                        target_nm=d,
                        model=row["model"],
                        apparent_cm3_s=value,
                        true_capture_cm3_s=row["true_capture_cm3_s"],
                        measured_cm3_s=obs[d],
                        difference_from_full_cm3_s=value - base,
                    )
                )
    save("other-kernels-steady.csv", rows)
    save("other-kernels-pairs.csv", pairs)
    (HERE / "other-kernels-run.json").write_text(
        json.dumps(
            dict(
                revision=revision(),
                device=wp.get_device("cuda:0").name,
                air_density_kg_m3=AIR,
                dissipation_m2_s3=DISSIPATION,
                ao2008_taylor_reynolds=100,
                ao2008_imposed_slip_m_s=0,
                temperature_K=TEMP,
                pressure_Pa=PRESSURE,
                rows=len(rows),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
