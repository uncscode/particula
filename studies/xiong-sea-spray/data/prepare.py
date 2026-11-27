"""Restore the study's observed inputs and optionally replay graphical readings."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from binary_tables import binary_tables
from growth_tables import relative_changes, table_s1, table_s2
from phase_table import phase_data

CODE = Path(__file__).resolve().parent
ARCHIVE = CODE.parents[2] / "docs/data/xiong-sea-spray/observations.zip"
SOURCES = json.loads((CODE / "publishers.json").read_text())
DIGESTS = json.loads((CODE / "observations.json").read_text())
FIGURES_ONLY = {"figure2.png", "xiong-2023-figure6.png"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def obtain(name, cache):
    source = SOURCES[name]
    destination = cache / name
    if destination.exists():
        if digest(destination) != source["sha256"]:
            raise ValueError(f"Checksum mismatch for cached {destination}")
        return
    request = Request(
        source["url"], headers={"User-Agent": "Particula-literature-study/1.0"}
    )
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        partial = Path(directory) / name
        with (
            urlopen(request, timeout=90) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        if digest(partial) != source["sha256"]:
            raise ValueError(
                f"Publisher checksum mismatch for {name}: {source['url']}"
            )
        partial.replace(destination)


def check_phase_figure(image, observations, output):
    """Independently read the two orange NaCl square/error bars in 2023 Fig. 6."""
    from PIL import Image, ImageDraw

    canvas = Image.open(image).convert("RGB")
    if canvas.size != (2067, 1031):
        raise ValueError("Unexpected phase diagram dimensions")
    draw = ImageDraw.Draw(canvas)
    with observations.open() as stream:
        for row, (x, top, bottom) in zip(
            csv.DictReader(stream),
            [(502, 246, 339), (1098, 133, 233)],
            strict=True,
        ):
            # 100% tick at y=85.5, 80% at y=343.4. Orange cap positions
            # are read at the outer edge of the square/circle overlap.
            for y in (top, bottom):
                r, g, b = canvas.getpixel((x, y))
                if not (r > 220 and 130 < g < 240 and b < 150):
                    raise ValueError(
                        f"Phase diagram cap missing at pixel {(x, y)}"
                    )
                draw.ellipse(
                    (x - 8, y - 8, x + 8, y + 8), outline="purple", width=2
                )
            center = 100 - ((top + bottom) / 2 - 85.5) * 20 / (343.4 - 85.5)
            sd = (bottom - top) * 10 / (343.4 - 85.5)
            if abs(center - float(row["onset_rh_pct"])) > 0.25:
                raise ValueError("Chun Xiong 2023 graphical onset differs")
            if abs(sd - float(row["error_upper_rh_points"])) > 0.5:
                raise ValueError("Chun Xiong 2023 graphical bar differs")
    canvas.thumbnail((1034, 516))
    canvas.save(output)


def prepare(output, cache, figures, digitizations_only):
    cache = cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        work = Path(temporary).resolve()
        with ZipFile(ARCHIVE) as archive:
            members = archive.namelist()
            for name in members:
                if name not in DIGESTS:
                    raise ValueError(f"Unrecognized digitization {name}")
                path = work / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
                if digest(path) != DIGESTS[name]:
                    raise ValueError(
                        f"Digitization checksum mismatch for {name}"
                    )

        if not digitizations_only:
            cache.mkdir(parents=True, exist_ok=True)
            for name in SOURCES:
                if figures or name not in FIGURES_ONLY:
                    obtain(name, cache)
            main = cache / "xiong-main.pdf"
            supplement = cache / "xiong-supp.pdf"
            coat = work / "aiomfac/coating"
            fit = coat / "parameter-fit"
            fit.mkdir(exist_ok=True)
            table_s1(supplement, work / "xiong")
            table_s2(supplement, coat)
            relative_changes(main, work / "xiong")
            phase_data(cache / "xiong-zenodo-8079001.xlsx", fit)
            binary_tables(cache, fit)
            if figures:
                for script, args, folder in [
                    ("xiong_figures.py", [main, supplement], work / "xiong"),
                    ("coating_figures.py", [main], coat),
                    (
                        "low_dose_figure.py",
                        [supplement],
                        work / "aiomfac/low_so2",
                    ),
                    (
                        "rosati_figure.py",
                        [cache / "figure2.png"],
                        work / "rosati",
                    ),
                ]:
                    subprocess.run(
                        [sys.executable, CODE / script, *args],
                        cwd=folder,
                        check=True,
                    )
                check_phase_figure(
                    cache / "xiong-2023-figure6.png",
                    coat / "onset-observations.csv",
                    fit / "phase-marker-check.png",
                )

        records = members if digitizations_only else DIGESTS
        for name in records:
            if digest(work / name) != DIGESTS[name]:
                raise ValueError(
                    f"Reconstructed observations differ for {name}"
                )
        products = [work / name for name in records]
        if figures:
            products.extend(work.rglob("*.png"))
        for path in products:
            target = output / path.relative_to(work)
            if target.exists() and digest(target) != digest(path):
                raise FileExistsError(
                    f"Different existing result: {target}; choose an empty output directory"
                )
        for path in products:
            target = output / path.relative_to(work)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        print(f"Verified {len(records)} observation CSVs in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination for observed input tables",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache/particula/xiong-sea-spray",
        help="Reusable checksum-verified publisher downloads",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--verify-sources",
        action="store_true",
        help="Also reread publisher plots and create coordinate overlays",
    )
    modes.add_argument(
        "--digitizations-only",
        action="store_true",
        help="Extract just the offline graphical readings",
    )
    args = parser.parse_args()
    prepare(
        args.output, args.cache, args.verify_sources, args.digitizations_only
    )


if __name__ == "__main__":
    main()
