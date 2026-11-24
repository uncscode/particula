"""Aggregate coincident PM1 inventories and evaluate the original charge-closure baseline."""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parents[1] / "extension"
SPECIES = ["organics", "nitrate", "sulfate", "ammonium", "chloride"]


def main():
    raw = pd.read_csv(EXTENSION / "chemistry/ams.csv.gz").dropna(
        subset=[s + "_fraction" for s in SPECIES]
    )
    grouped = raw.groupby(raw.time_unix // 21600)
    df = grouped[[s + "_ug_m3" for s in SPECIES]].mean().loc[grouped.size() >= 24]
    with np.load(EXTENSION / "blocks-final.npz") as archive:
        blocks, n, env = archive["keys"][:, 0], archive["n"], archive["env"]
    df = df[df.index.isin(blocks)].copy()
    df.insert(0, "utc", df.index * 21600)
    df["ams_scans"] = grouped.size()
    df["temperature_K"] = [
        np.average(env[blocks == k, 1], weights=n[blocks == k]) for k in df.index
    ]
    anions = df[["nitrate_ug_m3", "sulfate_ug_m3", "chloride_ug_m3"]].to_numpy()
    equivalents = (anions / [62.004, 96.063, 35.453]) @ [1, 2, 1]
    ammonium = df.ammonium_ug_m3 / 18.038
    sodium = (equivalents - ammonium) * 22.99
    if np.any(sodium < 0):
        raise ValueError("Unresolved negative charge requires a different baseline")
    df["measured_nh4_anion_equiv_ratio"] = ammonium / equivalents
    df["inferred_sodium_ug_m3"] = sodium
    df["original_dry_volume_um3_cm3"] = (
        df.organics_ug_m3 / 1.50
        + (anions.sum(axis=1) + df.ammonium_ug_m3 + sodium) / 1.76
    )
    df.to_csv(HERE / "composition.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
