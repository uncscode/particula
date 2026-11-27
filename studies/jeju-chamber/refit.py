"""Repeat the bounded individual or pooled search from the published starting design."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from gradient import gradient
from replay import HERE, calculate, observations
from scipy.special import expit
from scipy.stats import qmc
from scoring import WINDOWS


def group_run(group):
    return int(group[1]) - 1


def members(group):
    return list(range(6)) if group == "pooled" else [group_run(group)]


def source(group):
    return -1 if group == "R5archive" else 0


def base_design(study):
    rng = qmc.Sobol(d=36, seed=26419, scramble=True)
    u = rng.random_base2(6)
    # Broader alternate transport settings plus smaller perturbations at the inherited prior.
    org = study.model.origin.copy()
    zz = (u - 0.5) * 5
    zz[:32] = org + zz[:32]
    zz[32:] = 0.45 * org + zz[32:]
    # Across each run, tune the effective total nuclei once, not two aliased factors.
    zz[:, 24:30] = org[24:30]
    return np.vstack([org, study.base["C"], study.base["D"], zz])


def active(group):
    idx = np.ones(36, bool)
    idx[24:30] = False
    if group != "pooled":
        r = group_run(group)
        idx[30:] = False
        idx[30 + r] = True
        if r != 5:
            idx[7] = False
        if r == 0:
            idx[[12, 13, 15, 16, 18]] = False
    return idx


def objective(scores, group, wi):
    return scores[:, members(group), wi, -1].mean(axis=1)


def screen(study):
    z = base_design(study)
    for src in (0,):
        sc = study.evaluate(z, src)
        np.savez_compressed(OUT / f"screen{src}.npz", z=z, scores=sc)
        print("screen", src, len(z), flush=True)


def train(study, group, window):
    wi = WINDOWS.index(window)
    src = source(group)
    data = np.load(OUT / f"screen{src}.npz")
    zz = data["z"]
    scores = objective(data["scores"], group, wi)
    allowed = np.arange(len(zz))
    if window == "prefix":
        allowed = allowed[~np.isin(allowed, [1, 2])]
    rank = allowed[np.argsort(scores[allowed])]
    # Two separated starting regions, independently ranked by this run/window.
    best = rank[0]
    mask = active(group)
    far = [
        i for i in rank[1:] if np.linalg.norm((zz[i] - zz[best])[mask]) > 2.5
    ]
    starts = [best, far[0] if far else rank[1]]
    records_z = list(zz[allowed])
    records_s = list(scores[allowed])
    for start, idx in enumerate(starts):
        z = zz[idx].copy()
        score = scores[idx]
        rng = np.random.default_rng(17021 + start)
        for step in range(3):
            sigma = 0.65 * 0.80**step
            eps = rng.normal(size=(4, 36)) * mask
            sample = z + sigma * np.vstack([eps, -eps])
            vals = objective(study.evaluate(sample, src), group, wi)
            g = np.concatenate(
                [
                    gradient(z[k : k + 9], sample[:, k : k + 9], vals, sigma)
                    for k in range(0, 36, 9)
                ]
            )
            g *= mask
            proposed = z - np.array([0.7, 1.8])[:, None] * g / max(
                np.linalg.norm(g), 1.0e-15
            )
            prop_loss = objective(study.evaluate(proposed, src), group, wi)
            candidates = np.vstack([z, sample, proposed])
            values = np.r_[score, vals, prop_loss]
            records_z.extend(candidates[1:])
            records_s.extend(values[1:])
            v = values.argmin()
            z = candidates[v]
            score = values[v]
    # Standard-resolution reranking of distinct best candidates, including the fixed shared control for full fits.
    rank = np.argsort(records_s)
    finalists = []
    for i in rank:
        if all(
            np.linalg.norm((records_z[i] - q)[mask]) > 0.45 for q in finalists
        ):
            finalists.append(records_z[i])
        if len(finalists) == 5:
            break
    if window == "full":
        finalists.append(study.base["C" if src < 0 else "D"])
    candidates = np.array(finalists)
    actual = study.evaluate(candidates, src, dt=2)
    j = objective(actual, group, wi).argmin()
    np.savez_compressed(
        OUT / f"fit-{group}-{window}.npz",
        z=candidates[j],
        finalists=candidates,
        scores=actual,
        selected=j,
        trials_z=np.array(records_z),
        trials_score=records_s,
        starts=starts,
        cut=study.cut,
    )
    print(group, window, "score", objective(actual, group, wi)[j], flush=True)


class Search:
    """Evaluate the original 36 search coordinates with the portable physical solver."""

    def __init__(self, device, cache):
        from dust import source_arrays
        from grid import ChamberGrid, ChamberHeat
        from solver import Circulation
        from thermal import archives

        self.exp = Circulation(cache, device=device)
        self.heat = ChamberHeat(archives()[0])
        self.grid = ChamberGrid(6, 1)
        self.material = source_arrays()
        self.obs = observations(self.exp)
        self.cut = self.obs["cut"]
        self.settings = json.loads((HERE / "settings.json").read_text())
        design = json.loads((HERE / "search-design.json").read_text())
        self.base = {k: np.array(v) for k, v in design["seeds"].items()}
        self.model = SimpleNamespace(origin=self.base["A"])
        self.bounds = np.asarray(design["bounds"])
        self.reference_p = np.asarray(design["reference_p"])
        self.group = "pooled"

    def evaluate(self, samples, source_id, dt=5):
        if source_id != 0:
            raise ValueError(
                "This refit uses the published A2-constrained comparison."
            )
        indices = [
            45,
            46,
            57,
            59,
            11,
            12,
            15,
            17,
            19,
            21,
            25,
            26,
            40,
            41,
            42,
            43,
            *range(28, 40),
        ]
        out = []
        for z in samples:
            v = self.bounds[:, 0] + np.diff(self.bounds).ravel() * expit(z)
            p = self.reference_p.copy()
            p[[0, 1, 4, 7]] = v[4:8]
            p[indices] = np.where(
                z[8:] == self.model.origin[8:], self.reference_p[indices], v[8:]
            )
            p[62] = 1.0
            key = self.group + "-search"
            self.settings["cases"][key] = {
                "microphysics": p,
                "flow_thermal": v[:4],
            }
            result = calculate(
                self.exp,
                self.heat,
                self.grid,
                self.material,
                self.settings,
                key,
                dt,
                self.obs,
            )
            score = np.full((6, 3, 6), np.nan)
            score[result["runs"]] = result["scores"]
            out.append(score)
        return np.stack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "screen", "optimize"])
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run",
        choices=["pooled", *[f"R{i}" for i in range(1, 7)]],
        default="pooled",
    )
    parser.add_argument("--window", choices=["full", "prefix"], default="full")
    args = parser.parse_args()
    global OUT
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["JEJU_RAW"] = str(args.raw.resolve())
    wp.config.kernel_cache_dir = str(args.cache / "warp")
    study = Search(args.device, args.cache)
    if args.stage == "pilot":
        np.save(OUT / "pilot.npy", study.evaluate([study.model.origin], 0))
    elif args.stage == "screen":
        screen(study)
    else:
        study.group = args.run
        train(study, args.run, args.window)


if __name__ == "__main__":
    main()
