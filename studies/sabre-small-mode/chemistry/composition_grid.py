"""Recompute wet surface at nine dry organic fractions for AD interpolation."""

import numpy as np
import pandas as pd
from hydration import (
    HERE,
    MODEL,
    SIGMA,
    STUDY,
    WATER_MW,
    WATER_RHO,
    ActivityTable,
    R,
    size_quadrature,
    water_equilibrium,
)


def main():
    """Generate composition-dependent areas and check equilibrium directly."""
    flights = pd.read_csv(STUDY / "flight-windows.csv")
    humidity = pd.read_csv(HERE / "flight-chemistry.csv")
    frame = flights.merge(
        humidity[["flight", "start_s", "rhw_pct"]],
        on=["flight", "start_s"],
        validate="one_to_one",
    )
    joined = np.load(STUDY / "raw/joined.npz")
    keys = dict(
        zip(
            zip(joined["flight"], joined["start_s"], strict=True),
            joined["dndlog10d_cm3"],
            strict=True,
        )
    )
    sections = {g: size_quadrature(joined["edges_nm"], g) for g in [270, 300]}
    # The rest of the population remains pure acid in all nine cases.
    baseline = pd.read_csv(HERE / "hydration.csv").query("case=='aiomfac'")
    idx = ["flight", "start_s"]
    frame = frame.merge(
        baseline[[*idx, "wet_other_area", "dry_total_area"]],
        on=idx,
        validate="one_to_one",
    )
    rows = []
    max_error = 0.0
    for organic in np.linspace(0.25, 0.75, 9):
        table = ActivityTable(organic, frame.temperature_k, 140)
        for i, row in enumerate(frame.itertuples()):
            d, width, small, bins = sections[270 if row.n2o_ppbv < 300 else 300]
            concentration = keys[(row.flight, row.start_s)][bins]
            area = (
                np.pi
                * (d / 1000) ** 2
                * width
                * concentration
                * row.number_density_ambient_over_STP
            )
            rh = row.rhw_pct / 100
            w, factor = water_equilibrium(
                table, row.temperature_k, rh, d, SIGMA, False
            )
            wet_small_area = np.sum(area * small * factor)
            if i % 20 == 0:
                take = np.arange(0, len(d), 32)
                z = np.column_stack(
                    (
                        w[take],
                        np.full(len(take), organic),
                        np.full(len(take), 1 - organic),
                    )
                )
                activity = MODEL.evaluate(
                    z, row.temperature_k
                ).neutral_activities[:, 0]
                exponent = (
                    4
                    * SIGMA
                    * WATER_MW
                    / (
                        R
                        * row.temperature_k
                        * WATER_RHO
                        * d[take]
                        * 1e-9
                        * np.sqrt(factor[take])
                    )
                )
                max_error = max(
                    max_error, np.max(abs(activity * np.exp(exponent) / rh - 1))
                )
            rows.append(
                dict(
                    flight=row.flight,
                    start_s=row.start_s,
                    f_org=organic,
                    dry_total_area=row.dry_total_area,
                    wet_small_area=wet_small_area,
                    wet_other_area=row.wet_other_area,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "composition-grid.csv", index=False, float_format="%.10g")
    print(f"{len(out)} rows, max direct relative RH residual {max_error:.7f}")


if __name__ == "__main__":
    main()
