"""Hold run 5's selected full-fit parameters fixed across four physical alternatives."""

import numpy as np
import pandas as pd


def perturb(exp, heat, grid, material, settings, obs, steps, out):
    from replay import calculate

    rows = []
    r = 4
    late = obs["valid"][r] & (np.arange(exp.nmin) >= obs["cut"][r])
    assert late.sum() == 10
    for dt in steps:
        for archived_fill in (False, True):
            for uniform in (False, True):
                result = calculate(
                    exp,
                    heat,
                    grid,
                    material,
                    settings,
                    "R5-full",
                    dt,
                    obs,
                    archived_fill=archived_fill,
                    uniform_vapor=uniform,
                )
                book = result["budget"][0]
                sc = result["scores"][0]
                row = dict(
                    source="archived" if archived_fill else "A2",
                    contrast="removed" if uniform else "fitted",
                    dt_s=dt,
                    optical_full_decades=sc[0, 0],
                    optical_late_decades=sc[2, 0],
                    vapor_late_rmse_g_m3=sc[2, 2],
                    imposed_vapor_contrast=result["p"][57],
                    water_closure_g_m3=book[0] - book[1],
                    number_closure_cm3=book[2] - book[3],
                )
                for lo, hi in [(2, 5), (5, 10), (10, 20)]:
                    mask = (exp.d >= lo) & (exp.d < hi)
                    for label, arr in [
                        ("measured", obs["y"][r]),
                        ("simulated", result["y"][0]),
                    ]:
                        row[f"{label}_{lo}_{hi}_cm3"] = (
                            arr[late][:, mask].sum(axis=1).mean()
                        )
                assert abs(row["water_closure_g_m3"]) < 0.026
                assert abs(row["number_closure_cm3"]) < 0.5
                rows.append(row)
                print(
                    row["source"],
                    row["contrast"],
                    dt,
                    "s",
                    row["optical_full_decades"],
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
