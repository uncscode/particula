"""Extract 298 K ammonium sulfate observations and the two published DKA fits."""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
SALT_KG_MOL = 0.13214
HTML = HERE / "observations-source.html"
EXPECTED = "de2c36ccf3fff983f5d1400b62dc17f1b680fc9beb324dd86d55cd4284ba24df"


def cheng(m):
    """Cheng et al. (2015), supplement Table 1; m in mol per kg of water."""
    x = (m - 13.02) / 17.15
    return (46.29 + 26.42 * x + 12.51 * x**2) / (
        80.77 + 103.6 * x + 55.09 * x**2 + x**3
    )


def mikhailov(solute_fraction):
    """Mikhailov et al. (2024), supplement Table S1, valid X=0.15–0.81."""
    return np.polynomial.polynomial.polyval(
        solute_fraction, [1.0, -0.38778, 1.12310, -4.34221, 3.25940, -0.56537]
    )


def observations():
    buf = HTML.read_bytes()
    assert hashlib.sha256(buf).hexdigest() == EXPECTED
    blocks = re.findall(r"<pre>(.*?)</pre>", buf.decode("latin-1"), re.S)
    rows = []
    for line in blocks[1].strip().splitlines()[1:]:
        standardized, _, aw, wt, original, temp, ref, weight = line.split()
        assert temp == "25" and ref == "a"
        rows.append(["Tang1994", "2a", "", 298.15, aw, wt, original, standardized, weight])
    sources = {"a": "Clegg1995", "b": "Chan1992", "c": "Wyslouzil_unpublished"}
    for line in blocks[2].strip().splitlines()[1:]:
        fields = line.split()
        assert len(fields) in (7, 14)
        for j in range(0, len(fields), 7):
            number, aw, _, m, temp, ref, weight = fields[j:j + 7]
            if temp == "25" and sources[ref] != "Wyslouzil_unpublished":
                rows.append([sources[ref], "2b", number, 298.15, aw, "", "", m, weight])
    return rows


def write_csv(name, header, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    edb = observations()
    write_csv("edb.csv", ["study", "table", "particle_id", "T_K", "aw_observed",
              "original_wt_percent", "original_mol_kg", "standardized_mol_kg", "fit_weight"], edb)
    solve_lo = mikhailov(0.81)
    solve_hi = mikhailov(0.15)
    targets = np.unique(np.r_[np.linspace(0.26, 0.56, 61), solve_lo])
    rows = []
    for aw in targets:
        m = brentq(lambda salt_m: cheng(salt_m) - aw, 0, 160)
        wd = 1 / (1 + m * SALT_KG_MOL)
        wf = 1 - brentq(lambda s: mikhailov(s) - aw, 0.15, 0.81) if solve_lo <= aw <= solve_hi else ""
        rows.append([aw, m, wd, wf])
    write_csv("dka-inverse.csv", ["aw", "Cheng_mol_kg", "Cheng_water_mass_fraction",
              "Mikhailov_water_mass_fraction_in_range"], rows)
    print("298.15 K EDB records:", len(edb))
    for study in sorted(set(row[0] for row in edb)):
        subset = [r for r in edb if r[0] == study and 0.26 <= float(r[4]) <= 0.56]
        print(study, len(subset), "in range", min(float(r[4]) for r in subset), max(float(r[4]) for r in subset))
    print("Mikhailov minimum supported water activity:", solve_lo)


if __name__ == "__main__":
    main()
