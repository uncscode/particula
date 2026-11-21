"""Compare evaporation and chemical loss against bulk aerosol measurements."""

import numpy as np
import pandas as pd
from finish import read, sized
from fit import OBJECTIVES
from reactions import CANDIDATES
from smoke import DOSES, OUT

from gpu import Model


def main():
    """Replay the two losses at the same effective rate."""
    m = Model()
    cards = read("mechanism.json")
    index = {r.key: i for i, r in enumerate(CANDIDATES)}
    keys = [c["key"] for c in cards["reactions"]]
    records = []
    for n, photo, evap in [
        (3, "a_photo", "a_evap"),
        (4, "i_photo", "i_evap"),
        (20, "a_photo", "a_evap"),
        (20, "i_photo", "i_evap"),
    ]:
        rec = read(f"round_{n:02}.json")
        for obj in OBJECTIVES:
            p = rec["fits"][obj]["p"]
            k = sized(rec["fits"][obj]["k"])
            ids = rec["active_indices"]
            if n == 20:
                p = cards["baseline_parameters"][obj]
                ids = [index[key] for key in keys]
                k = np.zeros(len(CANDIDATES))
                k[ids] = [r[f"{obj}_fit_rate"] for r in cards["reactions"]]
            h = index[photo]
            e = index[evap]
            score, out, doses, _ = m.run(p, k, ids, DOSES, obj, sub=4)
            kp = k.copy()
            kp[e] = kp[h]
            kp[h] = 0.0
            alt = [e if a == h else a for a in ids]
            v, obs, _, _ = m.run(p, kp, alt, doses[0, :, None], obj, sub=4)
            records.append(
                dict(
                    stage=n,
                    objective=obj,
                    channel=photo,
                    alternative=evap,
                    initial_rate=k[h],
                    substituted_rate=kp[e],
                    score=score[0],
                    altered=v[0],
                    max_error_difference=np.abs(
                        out[0, :, :2] - obs[0, :, :2]
                    ).max(),
                    max_peak_difference_ug_m3=np.abs(
                        out[0, :, 3] - obs[0, :, 3]
                    ).max(),
                )
            )
            if n < 5:
                np.testing.assert_allclose(
                    out[0, :, :2], obs[0, :, :2], atol=1e-10
                )
                np.testing.assert_allclose(
                    out[0, :, 3:6], obs[0, :, 3:6], atol=1e-8
                )
            if n == 20 and photo == "a_photo":
                kp = k.copy()
                kp[h] = -3 * np.log1p(-0.0056) / 24.0
                v, obs, _, _ = m.run(p, kp, ids, doses[0, :, None], obj, sub=4)
                records.append(
                    dict(
                        stage=n,
                        objective=obj,
                        channel=photo,
                        alternative="solar_bulk_diameter_scale",
                        initial_rate=k[h],
                        substituted_rate=kp[h],
                        score=score[0],
                        altered=v[0],
                        max_error_difference=np.abs(
                            out[0, :, :2] - obs[0, :, :2]
                        ).max(),
                        max_peak_difference_ug_m3=np.abs(
                            out[0, :, 3] - obs[0, :, 3]
                        ).max(),
                    )
                )
    pd.DataFrame(records).to_csv(OUT / "physical_alternatives.csv", index=False)


if __name__ == "__main__":
    main()
