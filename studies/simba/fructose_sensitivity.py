"""Test the archived and published conditions for the fourth HMF experiment."""

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from published_rankings import refit
from reference_inputs import CASES, REVISION, fructose
from scipy.integrate import solve_ivp


def evaluate(archive):
    """Change one initial condition or kinetic bound across matched refits."""
    times = np.linspace(0.0, 90.0, 30)
    published = [4.0, 0.0, 1.0]
    rng = np.random.RandomState(1998)
    noise = rng.normal(0.0, CASES["fruc_HMF2"][2], (5, 3, 30))[3]
    regenerated = np.maximum(
        0.0, solve_ivp(fructose, (0.0, 90.0), published, t_eval=times).y + noise
    ).T
    settings = {
        "archived": {},
        "archived_default_code_bound": {"upper": 3.0},
        "paper_initials_and_regenerated_data": {
            "initial_override": published,
            "data_override": regenerated,
        },
        "paper_initials_archived_data": {"initial_override": published},
    }
    scores = {}
    for key, kwargs in settings.items():
        scores[key] = [
            refit((archive, "fruc_HMF2", i, False), **kwargs)["aic"]
            for i in (1, 2, 3)
        ]
    with tarfile.open(archive) as source:
        log = (
            source.extractfile(
                f"auto_react_mech_construct-{REVISION}/output_fruc_to_hmf2.log"
            )
            .read()
            .decode()
        )
    scores["authors_log"] = [
        float(s) for s in re.findall(r"Current AIC value: ([-.\d]+)", log)
    ]
    return scores


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(evaluate(sys.argv[1]), indent=2) + "\n"
    )
