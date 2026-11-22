"""Prospective high/low acid rate ratios at the two 223 K replication settings."""

from itertools import product
import numpy as np
import torch
from design import covariates
from fit import FITDIR, log_rates
from fraction_profile import read_best
from reference import OUT, dump_csv


def main():
    rows = []
    for training, topology, precision in product(
        ("complete", "223_low"),
        ("full", "assisted"),
        ("reported_gases", "shared_10pct", "separate_acid_10pct"),
    ):
        pairs = []
        factors = (
            [1.0]
            if precision == "reported_gases"
            else np.linspace(0.9, 1.1, 21)
        )
        for base_factor, acid_low in product(factors, factors):
            high_factors = (
                factors if precision == "separate_acid_10pct" else [acid_low]
            )
            pairs.extend((base_factor, acid_low, high) for high in high_factors)
        base_factor, acid_low, acid_high = np.array(pairs).T
        inputs = [
            covariates(223.0, acid * scale, 0.06 * base_factor)
            for acid, scale in [(6.64, acid_low), (13.05, acid_high)]
        ]
        ratios = []
        for shape in ("linear", "curved"):
            solutions = read_best(
                FITDIR
                / f"{training}_{topology}_{shape}_nominal_neutral_group.json"
            )
            for run in solutions:
                if run["cost"] > 1.02 * solutions[0]["cost"]:
                    continue
                params = torch.tensor(run["parameters"]).expand(len(pairs), -1)
                low, high = [
                    log_rates(params, x, topology, shape)[0] for x in inputs
                ]
                ratios.extend((high - low).exp().tolist())
        rows.append(
            dict(
                training=training,
                topology=topology,
                gas_scenario=precision,
                ratio_min=min(ratios),
                ratio_max=max(ratios),
                observed_ratio=15.81 / 0.32,
                gas_pairs_per_start=len(pairs),
            )
        )
    dump_csv(OUT / "paired_scan.csv", rows)


if __name__ == "__main__":
    main()
