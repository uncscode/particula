"""Refresh the original-bin curves in the gallery artwork from a whole-record replay."""

import argparse
from pathlib import Path
import re

import numpy as np

TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "docs/studies/literature/jeju-chamber/toc.svg.in"
)


def refresh(results):
    with (
        np.load(results / "observations.npz") as obs,
        np.load(results / "pooled-full-1s.npz") as fit,
    ):
        valid = obs["valid"][0]
        edges = obs["edges"]
        widths = np.diff(np.log10(edges))
        assert fit["runs"][0] == 0
        curves = [
            a[0, valid].mean(axis=0, dtype=float) / widths
            for a in (obs["y"], fit["y"])
        ]
        peak = obs["d"][np.argmax(curves[0])]
        assert round(peak, 1) == 14.9 and np.argmax(curves[0]) == np.argmax(
            curves[1]
        )

    def x(diameter):
        return 51 + 231 * np.log10(diameter / edges[0]) / np.log10(
            edges[-1] / edges[0]
        )

    svg = TEMPLATE.read_text()
    for key, density in zip(("measured", "shared"), curves):
        assert (
            np.isfinite(density).all()
            and density.min() >= 0
            and density.max() <= 600
        )
        y = 238 - density * 114 / 600
        path = f"M51 {y[0]:.3f}" + "".join(
            f"V{height:.3f}H{x(right):.3f}"
            for height, right in zip(y, edges[1:])
        )
        svg, n = re.subn(
            rf'(<path id="{key}"[^>]*? d=")[^"]*', rf"\g<1>{path}", svg
        )
        assert n == 1
    ticks = []
    for d in (1, 3, 10, 30, 100):
        pos = x(d)
        ticks.append(
            f'<path d="M{pos:.3f} 238v4" stroke="#617679"/><text x="{pos:.3f}" y="256">{d}</text>'
        )
    svg, n = re.subn(
        r'(<g id="diameter-ticks"[^>]*>).*?(</g>)',
        lambda m: m[1] + "\n      " + "\n      ".join(ticks) + "\n    " + m[2],
        svg,
        flags=re.S,
    )
    assert n == 1
    TEMPLATE.write_text(svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    refresh(parser.parse_args().results)
