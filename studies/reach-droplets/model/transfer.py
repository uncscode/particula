"""Replay center-calibrated controls unchanged with finer layers and reversed circulation."""

import tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import HERE, simulate, VIEWS


from placement import half

rows = []
for mesh, flow in [(2304, "jets"), (1944, "reverse"), (1944, "jets")]:
    control = HERE / "runs" / f"{mesh}-{flow}-2-0.05-control.npz"
    if not control.exists():
        simulate(mesh, flow, 2, 0.05, "control", "cuda:0")
    z = np.load(control)
    for ce in ("import_above", "local_mergers"):
        par = json.loads(next((HERE / "matched").glob("*-" + ce + ".json")).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            simulate(
                mesh,
                flow,
                2,
                0.05,
                ce,
                "cuda:0",
                par["strength"],
                par["start_s"],
                par["end_s"],
                out,
            )
            v = np.load(next(out.glob("*.npz")))
            t = v["t"]
            y = (v["tail"] - z["tail"]).reshape(len(t), 3, 7)
            for k, shift in enumerate(z["registration_m"]):
                for i, view in enumerate(VIEWS):
                    a, b = y[:, k, 0], y[:, k, i]
                    rows.append(
                        (
                            mesh,
                            flow,
                            ce,
                            shift,
                            view,
                            a.max(),
                            b.max(),
                            t[np.argmax(a)],
                            t[np.argmax(b)],
                            half(t, a) - half(t, b),
                        )
                    )
pd.DataFrame(
    rows,
    columns=[
        "mesh",
        "flow",
        "case",
        "registration_m",
        "view",
        "A_peak_cm3",
        "B_peak_cm3",
        "A_peak_s",
        "B_peak_s",
        "B_lead_s",
    ],
).to_csv(HERE / "matched-transfer.csv", index=False, float_format="%.7g")
