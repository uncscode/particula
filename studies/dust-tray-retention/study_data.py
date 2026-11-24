"""Read saved dust-tray data for plotting, including local replay outputs."""

from pathlib import Path
from zipfile import ZipFile


def result_text(name: str) -> str:
    """Prefer a local replay result, otherwise read its archived counterpart."""
    root = Path(__file__).resolve().parent
    try:
        return (root / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        with ZipFile(root / "study-results.zip") as archive:
            return archive.read(name).decode("utf-8")
