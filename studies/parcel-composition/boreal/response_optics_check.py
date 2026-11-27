"""Validate interpolated population scattering against direct layered Mie."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from redraw_checkpoint import draw_box, fit_box
from response_course import ASCENT, CASES, SHORT
from response_optics import Scattering, exact


def check(root):
    """Compare area sums on identical size-stratified samples."""
    lut = Scattering(root / "response-scattering.npz")
    rows = []
    for case in CASES:
        for t in (*SHORT, ASCENT):
            boxes, _, _ = read(root / "response-course" / case.key / f"{t}.npz")
            rng = np.random.default_rng([20261127, case.seed, t, 0, 1])
            redraw = [draw_box(fit_box(b), rng) for b in boxes]
            for method, population in (("full", boxes), ("redraw", redraw)):
                for i, b in enumerate(population):
                    v, bc = b.dry.sum(1), b.dry[:, 3]
                    for mode, water in (("dry", 0.0), ("ambient", b.water)):
                        d, r, s = lut.coordinates(v, bc, water)
                        # Stratify by size and presence of BC.
                        labels = 2 * np.floor(np.log10(d) * 10).astype(int) + (
                            r > 0
                        )
                        ii, ww = [], []
                        for g in np.unique(labels):
                            j = np.flatnonzero(labels == g)
                            j = j[np.argsort(d[j])]
                            for subset in np.array_split(j, min(32, len(j))):
                                ii.append(subset[len(subset) // 2])
                                ww.append(len(subset))
                        d, r, s = d[ii], r[ii], s[ii]
                        direct = np.dot(ww, exact(d, r, s) * np.pi * d * d / 4)
                        approx = np.dot(ww, lut.area(d, r, s))
                        rows.append(
                            dict(
                                seed=case.seed,
                                method=method,
                                start=t,
                                parcel=i,
                                mode=mode,
                                samples=len(d),
                                particles=len(v),
                                exact_area_nm2=direct,
                                relative_error_pct=100 * (approx / direct - 1),
                            )
                        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    result = check(a.root)
    if not (result.relative_error_pct.abs() <= 1).all():
        raise ValueError("Scattering table misses the 1% sample check")
    result.to_csv(a.out, index=False, float_format="%.10g")
    print(result.groupby("mode").relative_error_pct.agg(["min", "max"]))
