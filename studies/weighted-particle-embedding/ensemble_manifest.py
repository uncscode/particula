"""Freeze a common stochastic reference before comparing model architectures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def checksum(path):
    """Identify an immutable native input or one seeded realization."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(root, out, slots, seeds, coagulation_steps, coupled_steps):
    """Select the same seeds for all particle resolutions and both processes."""
    file_sets = []
    inputs = {}
    for k in slots:
        input_name = f"stochastic-inputs/k{k}.npz"
        inputs[input_name] = checksum(root / input_name)
        with np.load(root / input_name) as d:
            expected = {
                (str(f), str(c), str(m), leg)
                for f, c, m in zip(
                    d["ids"], d["density_multiplier"], d["mix"], strict=True
                )
                for leg in ("wet_end", "dry_end")
            }
        for mode, n in (
            ("coagulation", coagulation_steps),
            ("coupled", coupled_steps),
        ):
            pattern = f"{mode}-k{k}-s{n}-seed"
            available = {}
            for path in (root / "stochastic").glob(pattern + "*.csv"):
                with path.open() as stream:
                    keys = [
                        (
                            r["family"],
                            r["density_multiplier"],
                            r["mix"],
                            r["leg"],
                        )
                        for r in csv.DictReader(stream)
                    ]
                if len(keys) == len(expected) and set(keys) == expected:
                    available[int(path.stem.removeprefix(pattern))] = path
            file_sets.append(available)
    common = sorted(set.intersection(*(set(a) for a in file_sets)))[:seeds]
    if len(common) != seeds:
        raise ValueError(f"Only {len(common)} complete common seeds")
    files = {
        str(group[i].relative_to(root)): checksum(group[i])
        for group in file_sets
        for i in common
    }
    result = dict(
        slots=slots,
        seeds=common,
        inputs=inputs,
        files=files,
        coagulation_steps=coagulation_steps,
        coupled_steps=coupled_steps,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != payload:
        raise ValueError(f"Existing reference manifest differs at {out}")
    out.write_text(payload)


def load_frozen(root, path, slots):
    """Verify all source hashes before reading the selected native CSV rows."""
    from stochastic_report import read

    data = json.loads(path.read_text())
    if sorted(slots) != sorted(data["slots"]):
        raise ValueError("Resolution set differs from the manifest")
    for name, sha in (data["inputs"] | data["files"]).items():
        if checksum(root / name) != sha:
            raise ValueError(f"Native reference changed at {name}")
    return read(
        root / "stochastic", files=[root / name for name in data["files"]]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--slots", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--coagulation-steps", type=int, default=32)
    p.add_argument("--coupled-steps", type=int, default=1024)
    create(**vars(p.parse_args()))
