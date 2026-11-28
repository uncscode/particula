"""Match two hypotheses to the same center-camera design amplitude and peak quarter."""

import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.optimize import brentq
from experiment import HERE, simulate

target = 0.08
mesh = 1536
folder = HERE / "matched"
folder.mkdir(exist_ok=True)
base = np.load(HERE / "runs" / f"{mesh}-jets-2-0.05-control.npz")["tail"][:, 0]
trials = []
for case, bracket in [("import_above", (0.01, 1.0)), ("local_mergers", (1.0, 60000.0))]:
    with tempfile.TemporaryDirectory() as tmp:
        start, end = (4.5, 5.125) if case == "import_above" else (4.375, 5.125)

        def response(strength, destination):
            y, t = simulate(
                mesh, "jets", 2, 0.05, case, "cuda:0", strength, start, end, destination
            )
            dy = y - base
            peak = np.argmax(dy)
            return float(dy[peak]), float(t[peak])

        def residual(v):
            amp, pt = response(v, Path(tmp))
            trials.append([case, start, v, amp, pt])
            return amp - target

        # Shift the source clock using A alone before fitting how many droplets appear there.
        amp, pt = response(0.5 if case == "import_above" else 8000.0, Path(tmp))
        correction = round((5.125 - pt) / 0.05) * 0.05
        start += correction
        end += correction
        strength = brentq(residual, *bracket, xtol=0.0001)
        amp, pt = response(strength, folder)
        assert abs(amp - target) < 0.0001 and 5.0 <= pt < 5.25
        print(case, strength, amp, pt, flush=True)
(HERE / "matched-trials.json").write_text(json.dumps(trials, indent=2) + "\n")
