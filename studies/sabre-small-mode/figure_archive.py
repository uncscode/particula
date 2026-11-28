"""Read and write the downloadable SABRE plot and digitization archive."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ARCHIVE = (
    HERE.parents[1]
    / "website/public/study-data/sabre-small-mode/figure-data.zip"
)
DIGITIZATIONS = {
    "jensen_spectra.csv": Path("jensen_spectra.csv"),
    "surface_markers.csv": Path("surface_markers.csv"),
    "decker-tracer-marks.csv": Path(
        "chemistry/evidence/decker-tracer-marks.csv"
    ),
}


def restore_digitizations():
    """Restore original figure coordinates to the replay working directory."""
    with ZipFile(ARCHIVE) as archive:
        for name, path in DIGITIZATIONS.items():
            (HERE / path).write_bytes(archive.read(name))


def write_archive(source, plotted):
    """Preserve digitizations verbatim and pack plotted columns reproducibly."""
    entries = {
        name: (source / path).read_bytes()
        for name, path in DIGITIZATIONS.items()
    }
    entries["figure-values.json"] = json.dumps(
        plotted, allow_nan=False, separators=(",", ":")
    ).encode()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".tmp")
    with ZipFile(temporary, "w") as archive:
        for name, content in entries.items():
            member = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.external_attr = 0o644 << 16
            archive.writestr(
                member, content, compress_type=ZIP_DEFLATED, compresslevel=9
            )
    temporary.replace(ARCHIVE)
