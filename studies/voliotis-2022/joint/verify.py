# %%
"""Verify the saved study, including analytic oracles and archived GPU fits."""

import hashlib
import json

import numpy as np
import pandas as pd
from campaign import STEP_S, basis
from engine import ORDER, RESULT, Engine
from optimize import FAMILIES, Score, definition
from scipy.linalg import expm


def verify_campaign():
    """Check matrix exponentials, sample counts and saved campaign scores."""
    times = np.arange(0, 721, STEP_S)
    mass = np.arange(len(times) - 1)[:, None].astype(float)
    dt = STEP_S / 3600
    for k, w in ((0.25, 0.2500000002), (2.0, 0.4), (0.8, 0.8)):
        native = basis((0, times, times, None, mass), 1 / k, w)[:, 0]
        matrix = np.array([[-k, 0, 1], [k, -w, 0], [0, 0, 0]])
        step = expm(dt * matrix)
        state = np.zeros(3)
        ref = [0.0]
        for source in mass[:, 0]:
            state[-1] = source / dt
            state = step @ state
            ref.append(state[1])
        np.testing.assert_allclose(native, ref, atol=1e-11, rtol=1e-12)
    curves = pd.read_csv(RESULT / "campaign_curves.csv")
    scores = pd.read_csv(RESULT / "campaign_scores.csv")
    for _, row in scores.iterrows():
        subset = curves[(curves.run == row.run) & (curves.family == row.family)]
        np.testing.assert_equal(len(subset), row.n)
        rmse = np.mean((subset.measured - subset.modeled) ** 2) ** 0.5
        np.testing.assert_allclose(rmse, row.rmse, atol=1e-12)
    np.testing.assert_equal(scores[scores.family == "additive"].n.sum(), 1258)
    doses = pd.read_csv(RESULT / "campaign_dose_scores.csv")
    for _, group in doses.groupby(["family", "stage"]):
        low = group.lower_worst_nrmse.iloc[0]
        high = group.upper_worst_nrmse.iloc[0]
        if not (
            low <= group.nrmse.max() + 1e-9 <= high + 2e-9 and high - low < 6e-6
        ):
            raise ArithmeticError("Incorrect campaign dose bounds")
        if not group.dose_multiplier.between(0.85 - 1e-10, 1.15 + 1e-10).all():
            raise ValueError("Campaign dose outside the injection interval")


def verify_chambers():
    """Re-score the fit witnesses on CUDA and check native budget records."""
    scorer = Score()
    engine = Engine(dt=3)
    for f in FAMILIES:
        rec = json.loads((RESULT / (f + ".json")).read_text())
        _, counts, n = definition(f)
        computed = scorer.objective(engine(rec["physical"], counts), n)[0]
        np.testing.assert_allclose(computed, rec["objective"], atol=1e-12)
    for row in pd.read_csv(RESULT / "direct_fine_audit.csv").itertuples():
        trace = engine(json.loads(row.physical_vector))
        value = scorer.objective(trace, 4)[0]
        np.testing.assert_allclose(value, row.worst_nrmse, atol=1e-12)
    for row in pd.read_csv(RESULT / "joint_gate_search.csv").itertuples():
        _, counts, n = definition(row.family)
        trace = engine(json.loads(row.physical), counts)
        peaks = np.array([y.max() for _, _, y in scorer.data])
        peak = np.max(abs(trace.max(2)[0, :n] / peaks[:n] - 1))
        ratio = max(scorer.objective(trace, n)[0] / 0.1, peak / 0.15)
        np.testing.assert_allclose(ratio, row.gate_ratio, atol=1e-11)
    for f in RESULT.glob("*_cuda0_*s_18000_split*.json"):
        meta = json.loads(f.read_text())
        family = f.name.split("_cuda0_")[0]
        rec = json.loads((RESULT / (family + ".json")).read_text())
        sha = hashlib.sha256(np.asarray(rec["physical"], dtype="<f8").tobytes())
        np.testing.assert_equal(sha.hexdigest(), meta["fit_hash"])
        if max(meta["mass_rel"], meta["carbon_rel"]) >= 2e-7:
            raise ArithmeticError("Native budget failed")
    equ = pd.read_csv(RESULT / "equivalent_sources.csv")
    for _, g in equ.groupby("solution"):
        p = json.loads((RESULT / "single_nox.json").read_text())["physical"]
        reference = engine(p)[0, :4]
        g = g.set_index("run").loc[list(ORDER[:4])]
        p[0], p[16] = g.intercept_at_30ppb.iloc[0], g.slope.iloc[0]
        p[9:13] = g.dose_multiplier.tolist()
        np.testing.assert_allclose(engine(p)[0, :4], reference, atol=1e-10)


if __name__ == "__main__":
    verify_campaign()
    verify_chambers()
    print("Campaign oracle, fit scores, budgets and source equivalence passed.")
