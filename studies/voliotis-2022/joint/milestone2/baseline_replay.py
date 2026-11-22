"""Reproduce Manchester baselines and compare dose-profiled campaign fits."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent
sys.path.insert(0, str(JOINT / "chains"))
from bank import BANK, DOSES, OUT  # noqa: E402
from bank import (  # noqa: E402
    FAMILIES as FAMILIES,
)
from bank import (  # noqa: E402
    PARENTS as PARENTS,
)
from bank import (  # noqa: E402
    TRIALS as TRIALS,
)
from bank import (  # noqa: E402
    Channel as Channel,
)  # noqa: E402
from reactions import CANDIDATES as CANDIDATES  # noqa: E402

from gpu import Model  # noqa: E402


def save(name, value):
    """Write an atomic JSON result."""
    path = HERE / name
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, allow_nan=False, indent=1) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run(model, x, ids, sub=4, doses=DOSES, reverse=False, diagnostics=False):
    """Replay a batch and profile one admissible dose per experiment."""
    x = np.atleast_2d(x)
    return model.run(
        x[:, :16],
        x[:, 16:],
        ids,
        doses,
        "joint",
        sub=sub,
        reverse=reverse,
        diagnostics=diagnostics,
    )


def packed(bank, x, ids, doses):
    """Serialize selected channels in execution order."""
    return dict(
        channels=[dict(asdict(bank[i]), area=bank[i].area_scaled) for i in ids],
        x=np.r_[x[:16], x[16 + np.asarray(ids)]].tolist(),
        dose=list(doses),
    )


def run_rows(model, name, err, doses):
    """Pair predictions with the observed mass peak and last sample."""
    rows = []
    for (r, _, _, y, _), e, dose in zip(model.runs, err, doses, strict=True):
        rows.append(
            dict(
                network=name,
                run=r,
                dose=dose,
                mass=e[0],
                count=e[1],
                closure_ug_m3=e[2],
                observed_drop=max(y) - y[-1],
                drop=e[3] - e[4],
                observed_peak=max(y),
                observed_last=y[-1],
                peak=e[3],
                last=e[4],
            )
        )
    return rows


def baseline():
    """Verify archived joint fits at the original and fine resolutions."""
    records = json.loads((OUT / "continued.json").read_text())
    archived = pd.read_csv(OUT / "sensitivity.csv")
    model = Model(BANK)
    summary, rows = {}, []
    for name in ("backbone", "family_evidence", "all"):
        record = records[name]["joint"]
        ids = record["active"]
        x = np.asarray(record["fit"]["x"])
        summary[name] = {}
        for label, sub, reverse in (
            ("reference", 4, False),
            ("1.875s_forward", 32, False),
            ("1.875s_reverse", 32, True),
        ):
            start = perf_counter()
            score, err, dose, _ = run(model, x, ids, sub=sub, reverse=reverse)
            elapsed = perf_counter() - start
            expected = (
                archived.query(
                    "network == @name and objective == 'joint' "
                    "and test == @label"
                )
                .iloc[0]
                .score
            )
            np.testing.assert_allclose(score[0], expected, atol=1e-9, rtol=0)
            rows.extend(
                dict(**r, replay=label)
                for r in run_rows(model, name, err[0], dose[0])
            )
            summary[name][label] = dict(
                score=score[0],
                seconds=elapsed,
                archive_delta=score[0] - expected,
            )
        save("baselines.json", summary)
    pd.DataFrame(rows).to_csv(HERE / "baselines_by_run.csv", index=False)
    best = min(
        summary,
        key=lambda k: max(
            summary[k][q]["score"] for q in ("1.875s_forward", "1.875s_reverse")
        ),
    )
    rec = records[best]["joint"]
    archived_runs = pd.read_csv(OUT / "per_run.csv").query("objective=='joint'")
    for name in summary:
        replay = (
            pd.DataFrame(rows)
            .query("network==@name and replay=='reference'")
            .set_index("run")
            .sort_index()
        )
        reference = (
            archived_runs.query("network==@name").set_index("run").sort_index()
        )
        columns = ["mass", "count", "dose", "drop"]
        if not replay.index.equals(reference.index):
            raise ValueError("Run identities differ from the archive")
        np.testing.assert_allclose(
            replay[columns], reference[columns], rtol=0, atol=1e-9
        )
        summary[name]["max_per_run_difference"] = {
            col: float(abs(replay[col] - reference[col]).max())
            for col in columns
        }
    save("baselines.json", summary)
    print(best, len(rec["active"]), summary, flush=True)


if __name__ == "__main__":
    baseline()
