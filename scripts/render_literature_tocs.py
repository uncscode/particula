"""Render all literature TOCs or selected study IDs from website/topics.json.

Run from the repository root with python -m scripts.render_literature_tocs.
"""

import argparse
import base64
import json
import runpy
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs/images"


def dots_svg():
    """Embed the transparent vector logo with styles scoped to its own SVG."""
    logo = (IMAGES / "icon_light_clear.svg").read_bytes()
    encoded = base64.b64encode(logo).decode("ascii")
    return (
        '<image width="22.5" height="17.1" aria-hidden="true" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xlink:href="data:image/svg+xml;base64,{encoded}"/>'
    )


def is_literature_study(resource):
    """Include primary and secondary literature studies."""
    return "study" in resource and (
        resource["source"].startswith("docs/studies/literature/")
        or resource["study"].get("secondary_category") == "literature"
    )


def render_svg(template, output, **values):
    """Fill an editable study illustration and embed the official dots."""
    drawing = Template(template.read_text()).substitute(
        dots=dots_svg(), **values
    )
    output.write_text(drawing)


def main():
    """Render the requested study artwork without running simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_ids", nargs="*", help="Default: all literature TOCs"
    )
    args = parser.parse_args()
    resources = json.loads((ROOT / "website/topics.json").read_text())[
        "resources"
    ]
    gallery = {
        r["study"]["id"]: r
        for r in resources
        if "tocFigure" in r and is_literature_study(r)
    }
    unknown = set(args.study_ids) - gallery.keys()
    if unknown:
        parser.error("Unknown literature study: " + ", ".join(sorted(unknown)))
    for study_id in args.study_ids or gallery:
        study = gallery[study_id]
        source = Path(study["tocSource"])
        if source.suffix == ".py":
            module = ".".join(source.with_suffix("").parts)
            runpy.run_module(module, run_name="__main__")
        else:
            render_svg(ROOT / source, IMAGES / study["tocFigure"])
        print(f"{study_id}: {study['tocFigure']}", flush=True)


if __name__ == "__main__":
    main()
