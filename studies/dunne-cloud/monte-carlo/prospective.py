"""Pointwise rival predictions at the original candidate settings, after refitting."""

import json

import numpy as np
import pandas as pd
from formation import DRAWS, HERE, kernel
from reference import ROOT


def main():
    grid = pd.read_csv(ROOT / "results/precision_design.csv")
    fits = pd.read_csv(HERE / "solutions-extension.csv").query(
        "split=='complete'"
    )
    summary = []
    for scenario, g in grid.groupby("precision", sort=False):
        g = g.reset_index(drop=True)
        pred = {}
        for (topo, shape), rows in fits.groupby(["topology", "shape"]):
            vals = []
            for fit in rows.sort_values("id").itertuples():
                row = DRAWS[fit.id]
                if scenario == "archived_gas_envelope":
                    af, bf = row["acid"], row["base0"]
                elif scenario == "ammonia20pct":
                    af, bf = row["acid"], 0.8 + 0.4 * row["u7"]
                else:
                    af, bf = 0.9 + 0.2 * row["u6"], 0.9 + 0.2 * row["u7"]
                temp = g.T_K.to_numpy()
                d = np.stack(
                    [
                        (temp - 248) / 50,
                        np.log(g.acid_cm3 * af / 1e6),
                        np.log(
                            g.NH3_pptv
                            * bf
                            * 1.005
                            * 101325
                            / (1.380649e-23 * temp)
                            * 1e-24
                        ),
                    ]
                )
                vals.append(
                    kernel(
                        np.array(json.loads(fit.parameters)), d, topo, shape
                    )[0]
                )
            pred[topo, shape] = np.exp(np.quantile(vals, [0.05, 0.95], axis=0))
        for topo in ["assisted", "full"]:
            g[topo + "_p05"] = np.minimum(
                pred[topo, "curved"][0], pred[topo, "linear"][0]
            )
            g[topo + "_p95"] = np.maximum(
                pred[topo, "curved"][1], pred[topo, "linear"][1]
            )
        g["gap"] = np.maximum(
            g.full_p05 / g.assisted_p95, g.assisted_p05 / g.full_p95
        )
        g["eligible"] = (
            g.within_seen_ammonia
            & (g[["assisted_p05", "full_p05"]].min(axis=1) >= 0.001)
            & (g[["assisted_p95", "full_p95"]].max(axis=1) <= 100)
        )
        summary.append(
            g[
                [
                    "T_K",
                    "acid_cm3",
                    "NH3_pptv",
                    "precision",
                    "full_p05",
                    "full_p95",
                    "assisted_p05",
                    "assisted_p95",
                    "gap",
                    "eligible",
                ]
            ]
        )
    pd.concat(summary).to_csv(HERE / "prospective-settings.csv", index=False)


if __name__ == "__main__":
    main()
