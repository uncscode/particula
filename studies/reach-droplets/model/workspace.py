"""Archived measurements and configurable storage for local GPU trajectories."""

import json
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
# A fitted history is a function of the sectional remap as well as its controls.
WORK = Path(os.environ.get("REACH_OUTPUT", "reach-output")).resolve() / "volume-remap-v1"
ARCHIVE = STUDY.parents[1] / "website/public/study-data/reach-droplets/digitizations.zip"


def measurement(name):
    with ZipFile(ARCHIVE) as archive, archive.open(name) as stream:
        return pd.read_csv(stream)


def background(run):
    return json.loads((STUDY / "early-background.json").read_text())[run]
