# %% Check the pinned Fortran cases and measured pure salts.
"""Check the pinned Fortran cases and measured pure salts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import particula_aiomfac
from solve import HERE, AIOMFACModel, MixtureSpec


def main():
    """Compare with the upstream activities and measured endmembers."""
    package = Path(particula_aiomfac.__file__).resolve().parents[2]
    result = []
    for name, component, ids in [
        ("nacl", "NaCl", {202: 1, 242: 1}),
        ("sulfate", "NH4HSO4", {204: 1, 248: 1}),
    ]:
        case = json.loads((package / "reference/cases" / f"{name}.json").read_text())
        expected = json.loads(
            (package / "reference/expected" / f"{name}.json").read_text()
        )
        p = AIOMFACModel.prepare(MixtureSpec({"water": {16: 1}, component: ids}))
        r = p.evaluate(case["fractions"], case["temperature_K"], diagnostics=True)
        result.append(
            (
                name,
                "water activity",
                expected["neutrals"]["water"]["activity"],
                r.neutral_activities[0],
            )
        )
        for i, ion in enumerate(r.ion_ids):
            expected_activity = expected["species"][str(ion)]["activity"]
            actual = r.ion_molalities[i] * np.exp(r.ion_log_coefficients[i])
            result.append((name, str(ion) + " activity", expected_activity, actual))
    frame = pd.DataFrame(result, columns=["case", "observable", "fortran", "cpu"])
    frame["abs_diff"] = abs(frame.cpu - frame.fortran)
    assert frame.abs_diff.max() < 1e-9
    frame.to_csv(HERE / "fortran-check.csv", index=False)
    measured = pd.read_csv(HERE / "../../xiong/endmembers.csv")
    tab = pd.read_csv(HERE / "halite.csv")
    ll = pd.read_csv(HERE / "liquid.csv")
    points = []
    for f, col in [(0, "nacl"), (1, "bisulfate")]:
        sel = tab[tab.f.eq(f)].merge(measured, on="rh_pct")
        sel = sel.merge(
            ll[ll.f.eq(f)][["rh_pct", "gf"]],
            on="rh_pct",
            suffixes=["_halite", "_liquid"],
        )
        for _, row in sel.iterrows():
            obs = row[col + "_gf"]
            half = row[col + "_spread"] + 0.01
            points.append(
                (
                    col,
                    row.rh_pct,
                    obs,
                    half,
                    row.gf_halite,
                    row.gf_liquid,
                    (obs / row.gf_halite) ** 3,
                )
            )
    pd.DataFrame(
        points,
        columns=[
            "salt",
            "rh_pct",
            "observed_gf",
            "printed_spread_plus_digitization",
            "gf_halite",
            "gf_liquid",
            "observed_over_additive_wet_volume",
        ],
    ).to_csv(HERE / "observed-endmembers.csv", index=False)
    print(frame.to_string(index=False))
    print(
        "parameters sha256",
        hashlib.sha256(
            (package / "src/particula_aiomfac/data/parameters.npz").read_bytes()
        ).hexdigest(),
    )


if __name__ == "__main__":
    main()
