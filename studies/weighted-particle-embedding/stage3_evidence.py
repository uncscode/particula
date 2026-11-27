"""Reduce completed one-box inputs to measured costs and paired error bounds."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
from scipy.stats import t


def write(p, rows):
    """Replace a completed result table atomically."""
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    with tmp.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)


def clock_markers(cache, filenames):
    """Read GPU handover times from the study's own window logs."""
    stamps = []
    for name in filenames:
        path = cache / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                try:
                    dt = datetime.fromisoformat(line.strip())
                except ValueError:
                    dt = parsedate_to_datetime(line.strip())
            except (ValueError, TypeError):
                continue
            if dt is not None and dt.tzinfo is not None:
                stamps.append(dt.astimezone(timezone.utc).isoformat())
    return stamps


def crossed_marker(row, stamps):
    """Exclude elapsed times spanning externally suspended GPU work."""
    if "start" not in row or "end" not in row:
        return False
    return any(row["start"] < t < row["end"] for t in stamps)


def contention_windows(cache, panel):
    """Read intervals with confirmed concurrent study integration work."""
    path = cache / "contention-windows.csv"
    if not path.exists():
        return []
    with path.open() as file:
        return [
            (r["start_utc"], r["end_utc"])
            for r in csv.DictReader(file)
            if r["panel"] == panel
        ]


def overlaps(row, windows):
    """Test whether any part of this timed trial was contended."""
    if "start" not in row or "end" not in row:
        return False
    return any(
        row["start"] < end and start < row["end"] for start, end in windows
    )


def obs(row):
    """Extract number and physical inventories at every observation."""
    return np.array(
        [
            [
                h["n"],
                *h["gas"],
                *h["particle"],
                sum(h["particle"]),
                sum(h["particle"]) + sum(h["gas"]),
            ]
            for h in row["history"]
        ],
        float,
    )


class Paired:
    """Accumulate reference uncertainty and paired differences using Welford."""

    def __init__(self, a):
        """Allocate paired sufficient statistics by time and field."""
        self.n = 0
        self.mean = np.zeros_like(a)
        self.m2 = np.zeros_like(a)
        self.ref = np.zeros_like(a)
        self.ref2 = np.zeros_like(a)
        self.max_member = 0.0

    def add(self, x, y):
        """Include one matched seed and its reference trajectory."""
        self.n += 1
        diff = x - y
        q = diff - self.mean
        self.mean += q / self.n
        self.m2 += q * (diff - self.mean)
        q = y - self.ref
        self.ref += q / self.n
        self.ref2 += q * (y - self.ref)
        self.max_member = max(
            self.max_member, float(abs(diff[:, 0] / y[:, 0]).max())
        )

    def score(self):
        """Calculate simultaneous upper estimates over measured fields."""
        floor = np.full(self.mean.shape, 1e-11)
        floor[:, 0] = 1
        cutoff = np.where(abs(self.ref) > floor, abs(self.ref), floor)
        raw = abs(self.mean) / cutoff
        upper = np.full_like(raw, np.inf)
        if self.n > 1:
            q = t.ppf(1 - 0.05 / (2 * self.mean.size * 2), self.n - 1)
            se = np.sqrt(np.maximum(0, self.m2) / (self.n * (self.n - 1)))
            sy = np.sqrt(np.maximum(0, self.ref2) / (self.n * (self.n - 1)))
            denom = np.where(
                abs(self.ref) > floor, abs(self.ref) - q * sy, floor
            )
            np.divide(
                abs(self.mean) + q * se, denom, out=upper, where=denom > 0
            )
        number = 100 * upper[:, 0].max()
        mass = 100 * upper[:, 1:].max()
        omission = 100 * raw[:, 0].max()
        observed_mass = 100 * raw[:, 1:].max()
        return dict(
            seeds=self.n,
            number_mean_pct=omission,
            number_upper_pct=number,
            mass_mean_pct=observed_mass,
            mass_upper_pct=mass,
            worst_member_number_pct=100 * self.max_member,
            decision="passes tested bounds"
            if number <= 1 and mass < 0.1
            else "exceeds target in mean"
            if omission > 1 or observed_mass >= 0.1
            else "unresolved",
        )


def run(folder, out):
    """Collect timing, rejection and refinement evidence by setting."""
    timing = defaultdict(list)
    pairs = {}
    failures = defaultdict(list)
    for f in folder.glob("*.json"):
        if f.name in ("closed.json", "device.json"):
            continue
        with f.open() as stream:
            r = json.load(stream)
        if "panel" not in r:
            continue
        key = (
            r["panel"],
            r["n"],
            r["species"],
            r["density"],
            r["charge"],
            r["kernel"],
            r["steps"],
            r["vmax"],
            r["horizon"],
        )
        if r["status"] != "completed":
            failures[key].append(r["why"])
            continue
        timing[key].append(
            (
                r["total_s"],
                r["setup_s"],
                r["advance_and_observe_s"],
                r["cuda_peak_allocated"],
                r["cuda_peak_reserved"],
                r["final_n"],
                r["vapor_steps"],
                r["collision_steps"],
                r["max_probability"],
                max(h["ledger"] for h in r["history"]),
            )
        )
        if r["panel"] not in ("crossover", "largest-refinement"):
            continue
        rk, rv = (4, 0.003) if r["panel"] == "crossover" else (4, 0.008)
        name = (
            f"{r['panel']}-{r['n']}-{r['species']}-{r['density']:g}-"
            f"{r['charge']}-{r['family']}-{r['seed']}-physical-{rk}-{rv:g}.json"
        )
        ref = folder / name
        if not ref.exists() or ref == f:
            continue
        target = json.loads(ref.read_text())
        if target["status"] != "completed":
            continue
        y = obs(target)
        x = obs(r)
        group = key + (r["family"],)
        if group not in pairs:
            pairs[group] = Paired(y)
        pairs[group].add(x, y)
    names = (
        "panel n species density charge kernel collision_steps_per_observation "
        "max_radius_change horizon_s"
    ).split()
    rows = []
    for key in sorted(timing):
        arr = np.array(timing[key])
        rows.append(
            dict(zip(names, key, strict=True))
            | dict(
                runs=len(arr),
                median_total_s=np.median(arr[:, 0]),
                q10_s=np.quantile(arr[:, 0], 0.1),
                q90_s=np.quantile(arr[:, 0], 0.9),
                median_setup_s=np.median(arr[:, 1]),
                median_advance_s=np.median(arr[:, 2]),
                max_allocated_bytes=int(arr[:, 3].max()),
                max_reserved_bytes=int(arr[:, 4].max()),
                mean_final_particles=arr[:, 5].mean(),
                mean_vapor_steps=arr[:, 6].mean(),
                mean_collision_steps=arr[:, 7].mean(),
                max_pair_probability=arr[:, 8].max(),
                max_relative_ledger=arr[:, 9].max(),
            )
        )
    write(out / "cuda-scaling.csv", rows)
    write(
        out / "cuda-refinement.csv",
        [
            dict(zip(names + ["family"], key, strict=True)) | v.score()
            for key, v in sorted(pairs.items())
        ],
    )
    write(
        out / "cuda-failures.csv",
        [
            dict(zip(names, key, strict=True))
            | dict(failed=len(v), first_reason=v[0])
            for key, v in sorted(failures.items())
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.folder, a.out)
