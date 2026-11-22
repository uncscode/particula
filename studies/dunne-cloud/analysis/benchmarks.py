"""Particula empirical baselines, digitized ceiling and S3 reproduction audit."""

import csv
import json
import numpy as np
from particula.dynamics import (
    ActivationNucleationStrategy,
    KineticNucleationStrategy,
)
from particula.dynamics.nucleation.nucleation_strategies import (
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    NucleationValidityDomain,
)
from reference import *


def main():
    domain = NucleationValidityDomain(
        ClosedInterval(1, 1e18), ClosedInterval(190, 310)
    )
    baseline = []
    group = np.stack((BAND, np.rint(np.log10(NH3) / 0.15)), axis=1)
    for name, exclude in SPLITS.items():
        test = WINDOW & (Q == 0) & exclude
        train = WINDOW & (Q == 0) & ~exclude
        for scheme, local in [("global", False), ("local", True)]:
            selected = train.copy()
            if local:
                selected &= np.isin(BAND, BAND[test])
            if not np.any(selected):
                continue
            _, inv, c = np.unique(
                group[selected], axis=0, return_inverse=True, return_counts=True
            )
            for power, cls in [
                (1, ActivationNucleationStrategy),
                (2, KineticNucleationStrategy),
            ]:
                # Fit log coefficient in SI with same condition weighting.
                C = ACID[selected] * 1e12
                coefficient = np.exp(
                    np.average(
                        np.log(J[selected] * 1e6) - power * np.log(C),
                        weights=1 / c[inv],
                    )
                )
                law = cls(
                    coefficient,
                    domain,
                    InjectionComposition((1,)),
                    FormationMetadata(1.7e-9),
                    coefficient_provenance="weighted log fit to designated training rows",
                )
                for i in np.where(test)[0]:
                    mass = ACID[i] * 1e12 * 0.098079 / 6.02214076e23
                    pred = law.potential_rate(mass, 0.098079, T[i]) / 1e6
                    assert np.isclose(
                        pred,
                        coefficient * (ACID[i] * 1e12) ** power / 1e6,
                        rtol=1e-6,
                    )
                    baseline.append(
                        dict(
                            line=i + 2,
                            holdout=name,
                            scheme=scheme,
                            power=power,
                            estimated_J=pred,
                            observed_J=J[i],
                            coefficient_SI=coefficient,
                        )
                    )
    dump_csv(OUT / "particula_baselines.csv", baseline)
    tab = np.loadtxt(
        ROOT / "rate_sources/kurten_fig7_kinetic_limit.csv",
        delimiter=",",
        skiprows=1,
    )
    ok = (ACID * 1e6 >= min(tab[:, 0])) & (ACID * 1e6 <= max(tab[:, 0]))
    cap = np.exp(
        np.interp(np.log(ACID * 1e6), np.log(tab[:, 0]), np.log(tab[:, 1]))
    )
    masks = list(
        csv.DictReader(
            (ROOT / "rate_sources/figure_s3_archive_membership.csv").open()
        )
    )
    predicted = channels("S1_precise").sum(axis=0)
    coverage = []
    for i in range(len(J)):
        coverage.append(
            dict(
                line=i + 2,
                figure_membership=masks[i]["status"],
                J_observed=J[i],
                J_curve=predicted[i],
                kinetic_limit=cap[i] if ok[i] else None,
                J_with_reference_ceiling=min(predicted[i], cap[i])
                if ok[i]
                else None,
            )
        )
    dump_csv(OUT / "reference_ceiling.csv", coverage)
    print(
        "cap support",
        sum(ok),
        "pred above",
        sum(ok & (predicted > cap)),
        "status labels",
        set(row["status"] for row in masks),
    )


if __name__ == "__main__":
    main()
