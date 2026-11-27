# %%
"""Map a hypothetical increase in aged density through a paired closed parcel."""

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "parcel"))
from parcel import PAIRS, Case, load_survival, simulate, write_csv


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    tables = load_survival(ROOT / "model/results/ageing.csv")
    with (ROOT / "model/results/density_sensitivity.csv").open() as f:
        ratios = {
            (r["precursor"], r["rh"]): float(
                r["density_ratio_at_fixed_geometric_volume"]
            )
            for r in csv.DictReader(f)
            if r["source"] == "interpolated"
        }
    rows = []
    start = time.monotonic()
    for pair, (c, a) in PAIRS.items():
        k = 0.1 if pair == "monoterpenes" else 0.05
        for rh in ("dry", "wet"):
            ds, vs = tables[a, rh, "primary"]
            for speed in (0.2, 1.0):
                for name, label, ratio in [
                    (c, "cyclic", 1.0),
                    (a, "uncorrected", 1.0),
                    (a, "matched-excess", ratios[a, rh]),
                    (a, "ten-percent", 1.10),
                ]:
                    local = tables.copy()
                    if name == a:
                        local[a, rh, "primary"] = (ds, vs / ratio**3)
                    cc = Case(precursor=name, rh=rh, w=speed, kappa=k)
                    first = simulate(cc, local, "cuda:0")
                    cutoff = float(first["original_cutoff_nm"])
                    fine = replace(cc, n=96, probes=64, rtol=1.0e-6, dz=0.5)
                    second = simulate(
                        fine,
                        local,
                        "cuda:0",
                        np.linspace(cutoff * 0.96, cutoff * 1.04, 101),
                    )
                    assert second["max_threshold_interval_pp"] < 0.15
                    rows.append(
                        dict(
                            pair=pair,
                            precursor=name,
                            rh=rh,
                            w=speed,
                            treatment=label,
                            kappa=k,
                            density_ratio=ratio,
                            coarse_fraction_pct=first["fraction_pct"],
                            **second,
                        )
                    )
    write_csv(out / "cloud_density.csv", rows)
    (out / "runtime.json").write_text(
        json.dumps(
            dict(
                device=wp.get_device("cuda:0").name,
                runs=2 * len(rows),
                elapsed_s=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(parser.parse_args().out)
