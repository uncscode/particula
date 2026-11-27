"""Compare coefficient approximations and conservative GPU exchange."""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from condensation_fusion import fused_step
from process_cost import forecast
from process_scale import DT, SPECIES, STEPS, ObservedBox, score, write_csv
from stage2_conductance import Conductance, examples, fit_features

GRID = 257
LO, HI = math.log(0.25e-9), math.log(5e-6)
METHODS = (
    "physical",
    "constants",
    "physical-grid",
    "net8",
    "net24-grid",
    "reuse",
    "fused",
    "fused-8",
    "richardson",
    "learned-step",
)


def train_head(width, train):
    """Refit the output of a fixed tanh layer to native conductance labels."""
    head = Conductance("features", width)
    fit_features(head, *train, seed=5431)
    return head


def fit_heads(folder):
    """Fit two residual networks on synthetic native transition factors."""
    train = examples(1703, 24000, "organic")
    test = examples(2751, 12000, "organic")
    heads, rows = {}, []
    for width in (8, 24):
        start = perf_counter()
        head = train_head(width, train)
        elapsed = perf_counter() - start
        x, a, truth = test
        pred = head.feature_factor(x, a)[0]
        error = (pred / truth - 1).abs().numpy() * 100
        rows.append(
            dict(
                width=width,
                parameters=sum(p.numel() for p in head.parameters()),
                train=24000,
                holdout=12000,
                fit_s=elapsed,
                median_error_pct=np.median(error),
                p99_error_pct=np.percentile(error, 99),
                max_error_pct=error.max(),
            )
        )
        heads[width] = head.cuda().eval()
    write_csv(folder / "fits.csv", rows)
    return heads


class TrialBox(ObservedBox):
    """Change vapor computation while retaining the event clocks and ledgers."""

    def __init__(self, *args, method, heads):
        """Prepare the constant factors or radius table for this method."""
        super().__init__(*args)
        self.method = method
        self.coeff = 4 * math.pi * self.diff
        self.linear = (1 + 0.283 * self.alpha) * self.path / (0.75 * self.alpha)
        self.quad = self.path.square() / (0.75 * self.alpha)
        self.cache = {}
        self.heads = heads
        if method.endswith("-grid"):
            radius = torch.linspace(
                LO, HI, GRID, device=self.device, dtype=self.dtype
            ).exp()
            if method == "physical-grid":
                k = super().conductance(radius, slice(None))
            else:
                k = self.network(radius, slice(None), 24)
            self.log_table = k.log()
            self.lanes = torch.arange(SPECIES, device=self.device)

    def network(self, radius, j, width):
        """Evaluate the positive baseline times its learned residual."""
        kn = self.path[j] / radius[:, None]
        beta = self.heads[width].feature_factor(kn, self.alpha[j])[0]
        return self.coeff[j] * radius[:, None] * beta

    def geometry(self):
        """Share the radius interpolation indices across all species tiles."""
        r, m = super().geometry()
        if self.method.endswith("-grid"):
            t = (r.log() - LO) * ((GRID - 1) / (HI - LO))
            # Fail visibly if the specified radius domain is ever exceeded.
            if bool(((t < 0) | (t > GRID - 1)).any()):
                raise ArithmeticError("Lookup radius outside tabulated domain")
            self.lo = t.long().clamp_max(GRID - 2)
            self.frac = (t - self.lo)[:, None]
        return r, m

    def conductance(self, radius, j):
        """Return size-dependent conductances in cubic metres per second."""
        if self.method == "constants":
            r = radius[:, None]
            return (
                self.coeff[j]
                * r.square()
                * (r + self.path[j])
                / (r * (r + self.linear[j]) + self.quad[j])
            )
        if self.method.endswith("-grid"):
            lanes = self.lanes[j]
            lo = self.log_table[self.lo[:, None], lanes]
            hi = self.log_table[self.lo[:, None] + 1, lanes]
            return torch.lerp(lo, hi, self.frac).exp()
        if self.method == "net8":
            return self.network(radius, j, 8)
        if self.method == "reuse" and self.vsteps:
            # After events (odd half steps), compute and retain new geometry.
            # Immediately following them (even half steps), rows still match.
            if self.vsteps % 2 == 0:
                return self.cache.pop(j.start)
            self.cache[j.start] = super().conductance(radius, j)
            return self.cache[j.start]
        return super().conductance(radius, j)

    def proposals(self, duration):
        """Return a coarse end state and keep the two-substep prediction."""
        initial_m, initial_g = self.mass.clone(), self.gas.clone()
        fused_step(self, duration)
        coarse_m, coarse_g = self.mass, self.gas
        self.mass, self.gas = initial_m, initial_g
        fused_step(self, duration / 2)
        fused_step(self, duration / 2)
        return coarse_m, coarse_g

    def extrapolate(self, duration, weight):
        """Correct the conservative two-step prediction with one residual."""
        coarse_m, coarse_g = self.proposals(duration)
        m = self.mass[:, 1:]
        proposed_m = m + weight * (m - coarse_m[:, 1:])
        proposed_g = self.gas + weight * (self.gas - coarse_g)
        good = (proposed_m.amin(0) >= 0) & (proposed_g >= 0)
        m.copy_(torch.where(good, proposed_m, m))
        self.gas.copy_(torch.where(good, proposed_g, self.gas))

    def vapor(self, duration, vmax, tile=4):
        """Execute the reference or fused conservative exchange."""
        if self.method in ("richardson", "learned-step"):
            self.extrapolate(
                duration,
                self.heads["step"] if self.method == "learned-step" else 1.0,
            )
        elif self.method.startswith("fused"):
            count = int(self.method.split("-")[1]) if "-" in self.method else 1
            for _ in range(count):
                fused_step(self, duration / count)
        else:
            super().vapor(duration, vmax, tile)


def fit_step(heads):
    """Fit a zero-intercept residual weight using separate physical rollouts."""
    numerator = denominator = 0.0
    for seed in (41171, 41219):
        box = TrialBox(
            "physical",
            seed,
            None,
            None,
            2048,
            SPECIES,
            method="fused",
            heads=heads,
        )
        scale = box.gas.clone()
        for half in range(2 * STEPS):
            start_m, start_g = box.mass.clone(), box.gas.clone()
            _, coarse = box.proposals(DT / 2)
            fine = box.gas
            box.mass, box.gas = start_m, start_g
            for _ in range(32):
                fused_step(box, DT / 64)
            x = (fine - coarse) / scale
            y = (box.gas - fine) / scale
            numerator += float((x * y).sum())
            denominator += float(x.square().sum())
            if half % 2 == 0:
                box.inject(DT)
                box.collide(DT)
    return numerator / denominator


def profile(box, repeats=9):
    """Partition warmed physical half steps with CUDA event timestamps."""
    rows = []
    for repeat in range(repeats + 1):
        events = []

        def mark(label, events=events):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append((label, event))

        mark("start")
        radius, total = box.geometry()
        mark("geometry")
        for start in range(0, SPECIES, 4):
            j = slice(start, start + 4)
            m = box.mass[:, start + 1 : start + 5]
            kdt = box.conductance(radius, j) * (DT / 2)
            mark("conductance")
            inv = (1 + kdt * box.cstar[j] / total[:, None]).reciprocal()
            w = kdt * inv
            released = (w * box.cstar[j] * (m / total[:, None])).sum(0)
            gas = (box.gas[j] + released / box.volume) / (
                1 + w.sum(0) / box.volume
            )
            mark("gas_solve")
            m.copy_(m * inv + w * gas)
            box.gas[j] = gas
            mark("commit")
        events[-1][1].synchronize()
        times = {}
        for (_, before), (key, after) in zip(events, events[1:], strict=False):
            times[key] = times.get(key, 0) + before.elapsed_time(after)
        if repeat:
            rows.append(dict(repeat=repeat, **times))
    return rows


@torch.inference_mode()
def main():
    """Save matched rollouts, time-step sensitivity and kernel timings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--hosts", type=int, default=1000000)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[70101, 70102, 70103]
    )
    args = p.parse_args()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    heads = fit_heads(folder)
    heads["step"] = fit_step(heads)

    def run(method, seed, steps):
        factory = partial(TrialBox, method=method, heads=heads)
        return forecast(
            "physical",
            seed,
            (None, None),
            box_type=factory,
            hosts=args.hosts,
            species=SPECIES,
            steps=steps,
            dt=DT,
        )

    box = TrialBox(
        "physical",
        70100,
        None,
        None,
        args.hosts,
        SPECIES,
        method="physical",
        heads=heads,
    )
    write_csv(folder / "profile.csv", profile(box))
    del box
    methods = (*METHODS, "fused-16")
    for method in methods:
        run(method, 70100, 2)
    for seed in args.seeds:
        trials = {}
        for method in np.random.default_rng(seed).permutation(methods):
            timings, obs = run(method, seed, args.steps)
            trials[method] = (timings, obs[:, :-1])
            print(
                seed, method, round(timings["total_ms"] / 1000, 3), flush=True
            )
        ref = trials["physical"][1]
        fine = trials["fused-16"][1]
        rows = []
        for method, (timings, obs) in trials.items():
            same = score(obs, ref, DT)
            refined = score(obs, fine, DT)
            rows.append(
                dict(
                    method=method,
                    seed=seed,
                    hosts=args.hosts,
                    steps=args.steps,
                    **timings,
                    **same,
                    refined_phase_pct=refined["phase_error_pct"],
                    refined_number_pct=refined["number_error_pct"],
                )
            )
        write_csv(folder / f"trials-{seed}.csv", rows)
    (folder / "settings.json").write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                torch=torch.__version__,
                dt_s=DT,
                hosts=args.hosts,
                steps=args.steps,
                seeds=args.seeds,
                grid=GRID,
                learned_step_weight=heads["step"],
                radius_min_m=math.exp(LO),
                radius_max_m=math.exp(HI),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
