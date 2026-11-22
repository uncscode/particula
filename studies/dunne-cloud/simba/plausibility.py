"""Compare fitted interfacial sulfate production to a dilute aqueous kinetic scenario."""

import json
import numpy as np
from replay import HERE, measurements
from search import write
from bulk import water_fraction, aqueous


def main():
    t, _, _, _, acid, of, pf, obs, hp, valid, so = measurements()
    par = json.loads((HERE / "run.json").read_text())["model_params"]
    rows = []
    for d in (3.0, 5.0, 10.0, 15.0):
        area = np.pi / 4 * (d * 1.0e-9) ** 2
        total_liters = np.pi / 6 * (d * 1e-9) ** 3 * 1e3
        water_volume_fraction = water_fraction(d)
        liters = total_liters * water_volume_fraction
        for ph in (2, 4, 6):
            ozone, peroxide = aqueous(ph)
            for kind, rate, flux, j in [
                ("ozone", ozone, of, 3),
                ("peroxide", peroxide, pf, 4),
            ]:
                fitted = area * flux * par["original_kernel_" + kind][j]
                predicted = liters * 6.02214076e23 * rate
                mask = t - t[0] < 3.8
                rows.append(
                    dict(
                        physical_diameter_nm=d,
                        pH=ph,
                        oxidant=kind,
                        water_volume_fraction=water_volume_fraction,
                        fitted_events_s=fitted[mask].mean(),
                        aqueous_events_s=predicted[mask].mean(),
                        fitted_over_aqueous=fitted[mask].mean()
                        / predicted[mask].mean(),
                    )
                )
    write("aqueous_rates.csv", rows)


if __name__ == "__main__":
    main()
