"""Compare the retained CLOUD port to laboratory curves at common compositions."""
import csv
import json

import numpy as np

from extract import HERE, SALT_KG_MOL, write_csv

activity = np.genfromtxt(HERE.parent / "activity.csv", delimiter=",", names=True)
m_port = (1 / activity["water_mass_fraction"] - 1) / SALT_KG_MOL
aw_port = activity["aw_target"]
assert np.all(np.diff(m_port) < 0)
with (HERE / "edb.csv").open() as f:
    observed = list(csv.DictReader(f))
paired = []
for row in observed:
    aw = float(row["aw_observed"])
    if not 0.26 <= aw <= 0.56:
        continue
    for field in ("original_mol_kg", "standardized_mol_kg"):
        if not row[field]:
            continue
        molality = float(row[field])
        if m_port[-1] <= molality <= m_port[0]:
            estimate = np.interp(molality, m_port[::-1], aw_port[::-1])
            paired.append([row["study"], field, molality, aw, estimate,
                           estimate - aw, row["fit_weight"]])
write_csv("edb-port-residuals.csv", ["study", "molality_basis", "mol_kg", "aw_observed",
          "aw_port_interp", "port_minus_observed", "archive_fit_weight"], paired)

band = np.genfromtxt(HERE / "cheng-F2a-band.csv", delimiter=",", names=True)
assert min(m_port) >= min(band["mol_kg"]) and max(m_port) <= max(band["mol_kg"])
lower = np.interp(m_port, band["mol_kg"], band["aw_lower_graphic"])
upper = np.interp(m_port, band["mol_kg"], band["aw_upper_graphic"])
outside = (aw_port < lower - 0.003) | (aw_port > upper + 0.003)
summary = {}
for study, basis in sorted(set((r[0], r[1]) for r in paired)):
    errors = np.array([r[5] for r in paired if (r[0], r[1]) == (study, basis)])
    summary[f"{study}_{basis}"] = {
        "n": len(errors), "mean_absolute_error_aw": float(np.abs(errors).mean()),
        "median_signed_error_aw": float(np.median(errors)),
    }
summary["Cheng_band_reading_sensitivity"] = {
    "max_outside_cloud_diameter_nm": float(max(activity["diameter_nm"][outside])),
    "min_inside_cloud_diameter_nm": float(min(activity["diameter_nm"][~outside])),
    "points_outside_by_more_than_0p003": int(outside.sum()),
    "computed_grid_points": len(activity),
}
(HERE / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
