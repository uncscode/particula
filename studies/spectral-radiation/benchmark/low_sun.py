"""Compare low-Sun plane-parallel and pseudo-spherical DISORT."""

import argparse
import tempfile
from pathlib import Path

from disort import HERE, run_uvspec, write_csv


def main(prefix):
    """Record matched plane-parallel and pseudo-spherical spectra."""
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for nm in (350, 550, 850):
            for angle in (30, 45, 60, 65, 70, 75, 80, 85, 87, 89):
                flux = []
                for geometry in ("", "pseudospherical"):
                    out = run_uvspec(
                        prefix,
                        temp,
                        f"""source solar
mol_abs_param reptran coarse
rte_solver disort
number_of_streams 16
{geometry}
albedo 0.2
sza {angle}
wavelength {nm} {nm}
zout 0 toa
output_user lambda zout edir edn eup""",
                    )
                    flux.append(out[0, 2:4] * 1e-3)
                plane, curved = flux
                rows.append(
                    dict(
                        wavelength_nm=nm,
                        sza_deg=angle,
                        plane_direct_W_m2_nm=plane[0],
                        curved_direct_W_m2_nm=curved[0],
                        plane_total_W_m2_nm=sum(plane),
                        curved_total_W_m2_nm=sum(curved),
                        relative_total_pct=100 * (sum(plane) / sum(curved) - 1),
                    )
                )
    write_csv(HERE / "low_sun.csv", rows)
    for angle in set(r["sza_deg"] for r in rows):
        print(
            angle,
            max(
                abs(r["relative_total_pct"])
                for r in rows
                if r["sza_deg"] == angle
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    main(parser.parse_args().prefix.resolve())
