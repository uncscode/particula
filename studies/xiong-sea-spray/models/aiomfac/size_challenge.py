# %% Test the air-volume ranking against the shared size-law assumptions.
import numpy as np
import pandas as pd
from fit import ROOT


def main():
    spectrum = pd.read_csv(ROOT.parent / "rosati/spectra.csv").set_index("diameter_nm")
    rows = []
    for tag in ["peak150", "peak250", "size", "challenge", "retctrl", "eq8", "blank"]:
        runs = pd.read_csv(ROOT / tag / "inventories.csv").query("time_h==3")
        for (network, variant), g in runs.groupby(["network", "variant"]):
            z = g.set_index("diameter_nm")
            for shape in ["initial", "five_hour", "mean"]:
                base = (
                    spectrum[shape + "_cm3_decade"]
                    * np.pi
                    / 6
                    * (spectrum.index * 1e-9) ** 3
                    * 2165
                    * 1e15
                    * 96.06
                    / 58.443
                )
                mass = base * z.total_svi
                rows.append(
                    (
                        tag,
                        network,
                        variant,
                        shape,
                        mass[200],
                        mass[300],
                        mass[300] / mass[200],
                    )
                )
    result = pd.DataFrame(
        rows, columns="stage network variant spectrum mass200 mass300 ratio".split()
    )
    result.to_csv(ROOT / "results/size-challenges.csv", index=False)


if __name__ == "__main__":
    main()
