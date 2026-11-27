"""Check the dependence of cross-salt transfer on training weights and AS observations."""

import numpy as np
import pandas as pd
from transfer import HERE, prediction


def main():
    keys = ["tail", "B1", "B2", "C2"]
    profiles = pd.concat(
        [
            pd.read_csv(HERE / f)
            for f in [
                "refined-calibration.csv",
                "extended-calibration.csv",
                "joint-calibration.csv",
            ]
        ]
    ).drop_duplicates(keys + ["NH4_delta"])
    status = pd.read_csv(HERE / "binary-stability.csv")
    profiles = profiles.merge(status[status.binary_stable][keys], on=keys)
    data = pd.read_csv(HERE / "phase-data.csv").query("salt=='AS'").set_index("ovf_pct")
    cases = {
        "training_SD_plus_2_RH_in_quadrature": [50, 70, 90],
        "omit_AS50": [70, 90],
        "omit_AS70": [50, 90],
        "omit_AS90": [50, 70],
    }
    rows = []
    for name, vs in cases.items():
        score = profiles.binary_loss.copy()
        for v in vs:
            sigma = np.hypot(
                data.loc[v].srh_sd_pct, 2.0 if name.startswith("training_SD") else 0.0
            )
            score += ((profiles[f"AS{v}"] - data.loc[v].srh_pct) / sigma) ** 2
        shortlist = (
            profiles.assign(score=score).dropna(subset=["score"]).nsmallest(4, "score")
        )
        options = []
        for _, r in shortlist.iterrows():
            costs = r.binary_loss
            for v in vs:
                sigma = np.hypot(
                    data.loc[v].srh_sd_pct,
                    2.0 if name.startswith("training_SD") else 0.0,
                )
                costs += ((prediction("AS", v, r) - data.loc[v].srh_pct) / sigma) ** 2
            if np.isfinite(costs):
                options.append((costs, r))
        if not options:
            continue
        cost, r = min(options, key=lambda z: z[0])
        rows.append(
            [
                name,
                *[r[k] for k in keys],
                r.NH4_delta,
                cost,
                *[prediction("NaCl", v, r) for v in [50, 70, 90]],
            ]
        )
    pd.DataFrame(
        rows,
        columns=[
            "training_treatment",
            *keys,
            "NH4_delta",
            "score",
            "NaCl50",
            "NaCl70",
            "NaCl90",
        ],
    ).to_csv(HERE / "training-sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
