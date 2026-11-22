"""Fixed published laws, empirical baselines and digitized kinetic ceiling under joint calibration."""

import numpy as np
import pandas as pd
from formation import (
    BAND,
    DRAWS,
    HERE,
    SPLITS,
    WINDOW,
    J,
    Q,
    dump_csv,
    inputs,
    weight,
)
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
from reference import T, ROOT, channels, scores


def main():
    maskinfo = pd.read_csv(
        ROOT / "rate_sources/figure_s3_archive_membership.csv"
    )
    included = maskinfo.status.str.startswith("included").to_numpy()
    lines = np.arange(len(T)) + 2
    sets = dict(
        all=T > 0,
        primary=WINDOW,
        neutral=WINDOW & (Q == 0),
        ionized=WINDOW & (Q > 0),
        figure45=included | (lines == 45),
        figure93=included | (lines == 93),
    )
    table = np.loadtxt(
        ROOT / "rate_sources/kurten_fig7_kinetic_limit.csv",
        delimiter=",",
        skiprows=1,
    )
    x = np.log(table[:, 0])
    measurements = []
    metrics = []
    estimates = []
    baselines = []
    for row in DRAWS:
        ac, base = inputs(row)
        concentration = ac * 1e6
        observed = J * row["jscale"]
        support = (concentration >= table[0, 0]) & (
            concentration <= table[-1, 0]
        )
        ceiling = np.exp(
            np.interp(np.log(concentration), x, np.log(table[:, 1]))
        )
        for law in ("S1_precise", "S1", "S2"):
            ch = channels(law, acid=ac, ammonia=base)
            total = ch.sum(axis=0)
            assert np.all(ch >= 0) and np.all(ch[2:].sum(axis=0) <= Q + 1e-9)
            for key, mask in sets.items():
                metrics.append(
                    dict(
                        id=row["id"],
                        block=row["block"],
                        law=law,
                        subset=key,
                        **scores(observed[mask], total[mask]),
                    )
                )
            if law == "S1_precise":
                estimates.append(total)
                measurements.append(
                    dict(
                        id=row["id"],
                        n_supported=int(support.sum()),
                        n_estimate_exceeds=int(
                            (support & (total > ceiling)).sum()
                        ),
                        n_observed_exceeds=int(
                            (support & (observed > ceiling)).sum()
                        ),
                        max_observed_ratio=(observed / ceiling)[support].max(),
                    )
                )
        for split, held in SPLITS.items():
            test = WINDOW & (Q == 0) & held
            if not test.any():
                continue
            train = WINDOW & (Q == 0) & ~held
            for label, mask in [
                ("global", train),
                ("local", train & np.isin(BAND, BAND[test])),
            ]:
                if not mask.any():
                    continue
                for power, cls in [
                    (1, ActivationNucleationStrategy),
                    (2, KineticNucleationStrategy),
                ]:
                    c = np.exp(
                        np.average(
                            np.log(observed[mask] * 1e6)
                            - power * np.log(ac[mask] * 1e12),
                            weights=weight(row, mask) ** 2,
                        )
                    )
                    native = cls(
                        c,
                        NucleationValidityDomain(
                            ClosedInterval(1, 1e18), ClosedInterval(190, 310)
                        ),
                        InjectionComposition((1,)),
                        FormationMetadata(1.7e-9),
                        coefficient_provenance="condition-weighted joint calibration draw",
                    )
                    pred = c * (ac[test] * 1e12) ** power / 1e6
                    idx = np.flatnonzero(test)[0]
                    assert np.isclose(
                        native.potential_rate(
                            ac[idx] * 1e12 * 0.098079 / 6.02214076e23,
                            0.098079,
                            T[idx],
                        )
                        / 1e6,
                        pred[0],
                        rtol=2e-7,
                    )
                    for i, val in zip(np.flatnonzero(test), pred):
                        baselines.append(
                            dict(
                                id=row["id"],
                                block=row["block"],
                                split=split,
                                scheme=label,
                                power=power,
                                line=i + 2,
                                predicted_J=val,
                                adjusted_J=observed[i],
                            )
                        )
    dump_csv(HERE / "archive-scores.csv", metrics)
    dump_csv(HERE / "archive-ceiling.csv", measurements)
    dump_csv(HERE / "baselines.csv", baselines)
    np.savez_compressed(
        HERE / "archive-rates.npz",
        id=[r["id"] for r in DRAWS],
        line=lines,
        predicted_J=np.array(estimates),
    )


if __name__ == "__main__":
    main()
