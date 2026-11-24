"""Exercise the fitted optical transfer and a direct, independent June 10 AIOMFAC case."""

from pathlib import Path
import numpy as np
import pandas as pd

from models import run as fit_optical
from thermo import run as simulate_recipe


def main():
    root = Path(__file__).resolve().parent
    optical = root / "mechanisms/droplets"
    fit_optical(optical)
    scores = (
        pd.read_csv(optical / "scores.csv")
        .query(
            "source=='selected' and split=='test' and floor==0.5 and model=='uniform_amplitude'"
        )
        .set_index("metric")
    )
    for metric, expected in {
        "observed_ratio": 0.77591834,
        "obs_mean_shift": -0.01289983,
        "model_ratio": 0.99913780,
        "model_mean_shift": -1.54117639,
    }.items():
        value = scores.at[metric, "mean"]
        np.testing.assert_allclose(value, expected, atol=3e-6, rtol=0)
        print(f"{metric}: {value:.8f}")
    assert scores.at["observed_ratio", "days"] == 7
    simulate_recipe("pilot", "cpu")
    pilot = pd.read_csv(root / "field-mixtures/chem/pilot-cpu.csv.gz")
    assert len(pilot) == 6
    center = pilot.query("recipe=='low' and model=='full'").iloc[0]
    np.testing.assert_allclose(
        center[["gf85_30", "gf85_60", "gf85_90"]].to_numpy(float),
        [1.461378, 1.4988514, 1.5125292],
        atol=2e-6,
        rtol=0,
    )
    np.testing.assert_allclose(center.dcrit08_nm, 36.582260, atol=2e-5)
    print("June 10 low-solubility pilot at 60 nm:", center.gf85_60)


if __name__ == "__main__":
    main()
