"""Check the relative ranking of a marginal pair at additional timesteps."""

import argparse
import json

import numpy as np
from ad_shortlist import errors
from baseline_replay import HERE, Model, save
from growth import BANK


def main():
    """Refine each training-selected fit with fixed doses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_file", help="Saved matched fine-fit filename")
    args = parser.parse_args()
    archive = json.loads((HERE / args.fit_file).read_text())
    model = Model(BANK)
    records = []
    for row in archive["results"]:
        fit = min(row["starts"], key=lambda s: s["score"])
        points = []
        for sub in (32, 64, 128):
            for reverse in (False, True):
                residual = errors(
                    model,
                    np.asarray(fit["x"]),
                    row["active"],
                    fit["dose"],
                    sub,
                    reverse,
                )
                run, obs = np.unravel_index(residual.argmax(), residual.shape)
                points.append(
                    dict(
                        step_s=60.0 / sub,
                        order="reverse" if reverse else "forward",
                        score=float(residual[run, obs]),
                        run=int(model.runs[run][0]),
                        observable=("mass", "count")[obs],
                    )
                )
        records.append(
            dict(extra=row["extra"], training_start=fit["start"], points=points)
        )
    save(
        "ad_refine_" + args.fit_file.removeprefix("ad_"),
        dict(
            source=args.fit_file,
            dose="Frozen continuous fit dose for each run",
            results=records,
        ),
    )


if __name__ == "__main__":
    main()
