"""Retrieve the second high-loading α-pinene chamber run from EUROCHAMP."""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
UTILITY = HERE.parents[1] / "data" / "prepare.py"


def get_utility():
    """Load the primary study’s archive utility."""
    spec = importlib.util.spec_from_file_location("voliotis_edf", UTILITY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate the study data utility at {UTILITY}")
    edf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edf)
    return edf


def prepare(cache):
    """Check raw EDF checksums and export experiment 2 to its own directory."""
    edf = get_utility()
    edf.RECORDS = {2: ("3549-H353", "GXV6-3646")}
    edf.HERE = HERE
    edf.build(cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    prepare(parser.parse_args().raw_dir)
