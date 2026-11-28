"""Refresh the number-conserving, reduced-resolution curves in the gallery artwork from a whole-record replay."""

import argparse
from pathlib import Path
import re

import numpy as np
from particula.particles import get_number_per_decade

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
        assert fit["runs"][0] == 0
        mean_counts = [
            a[0, valid].mean(axis=0, dtype=float) for a in (obs["y"], fit["y"])
        ]
        curves = (
            get_number_per_decade(np.array(mean_counts) * 1e6, edges * 1e-6)
            * 1e-6
        )
        peak = obs["d"][np.argmax(curves[0])]
        assert round(peak, 1) == 14.9 and np.argmax(curves[0]) == np.argmax(
            curves[1]
        )

    # Sum number before converting to density, including the last partial group.
    starts = np.arange(0, len(edges) - 1, 4)
    display_edges = np.r_[edges[starts], edges[-1]]
    display_widths = np.diff(np.log10(display_edges))
    display_curves = (
        get_number_per_decade(
            np.add.reduceat(mean_counts, starts, axis=1) * 1e6,
            display_edges * 1e-6,
        )
        * 1e-6
    )
    np.testing.assert_allclose(
        (display_curves * display_widths).sum(axis=1),
        np.sum(mean_counts, axis=1),
        rtol=1e-12,
    )

    min_diameter, max_diameter = 2, 40
    visible = (display_edges[:-1] < max_diameter) & (
        display_edges[1:] > min_diameter
    )
    visible_right = np.minimum(display_edges[1:][visible], max_diameter)

    def x(diameter):
        return 42 + 238 * np.log10(diameter / min_diameter) / np.log10(
            max_diameter / min_diameter
        )

    svg = TEMPLATE.read_text()
    for key, density in zip(("measured", "shared"), display_curves):
        assert (
            np.isfinite(density).all()
            and density.min() >= 0
            and density.max() <= 600
        )
        y = 241 - density[visible] * 123 / 600
        path = f"M{x(min_diameter):.3f} {y[0]:.3f}" + "".join(
            f"V{height:.3f}H{x(right):.3f}"
            for height, right in zip(y, visible_right)
        )
        svg, n = re.subn(
            rf'(<path id="{key}"[^>]*? d=")[^"]*', rf"\g<1>{path}", svg
        )
        assert n == 1
    ticks = []
    for d in (2, 5, 10, 20, 40):
        pos = x(d)
        ticks.append(
            f'<path d="M{pos:.3f} 241v4" stroke="#617679"/><text x="{pos:.3f}" y="260">{d}</text>'
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
