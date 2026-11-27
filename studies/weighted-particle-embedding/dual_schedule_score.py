"""Compare joint forecasts under partial resets and conditional organic mass."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_experiment import load_files, reopen, reporting, save_population
from dual_stages import DualStages, build
from dual_transfer import DualExchange
from node_transfer import write_table
from projection_study import digest


def references(root, split, geometry, blocks, families, bdf):
    """Match the native resolution pair and optional stiff reference."""
    folder = root / split / geometry
    d = load_files(folder / f"b{blocks}-s128-k16", families)
    c = load_files(folder / f"b{blocks}-s64-k16", families)
    for key in ("ids", "m", "n", "time", "reset_mask", "levels"):
        np.testing.assert_array_equal(d[key], c[key])
    error = abs(c["bulk"] - d["bulk"])
    panels = [("native", d["bulk"], error[::2] + error[1::2])]
    if bdf:
        # Validate the exact prefix against the higher-duration input before
        # using a long BDF trajectory for a shorter pulse panel.
        long = load_files(folder / "b3-s32-k16", families)
        for key in ("ids", "m", "n"):
            np.testing.assert_array_equal(long[key], d[key])
        for key in ("reset_mask", "levels"):
            np.testing.assert_array_equal(
                long[key][:, : d[key].shape[1]], d[key]
            )
        np.testing.assert_array_equal(long["time"][: len(d["time"])], d["time"])
        hi, lo = [
            bdf_values(root, geometry, long, tol, len(d["time"]))
            for tol in (1e-9, 1e-7)
        ]
        error = abs(hi - lo)
        panels.append(("BDF", hi, error[::2] + error[1::2]))
    return d, panels


def bdf_values(root, geometry, data, tol, frames):
    """Read native-source-verified stiff trajectories."""
    folder = root / "bdf" / geometry / "b3"
    parts = []
    for stem, sha in data["sources"]:
        with np.load(folder / f"{stem}-r{tol:.0e}-coupled.npz") as a:
            if str(a["source_sha256"]) != sha or a["tolerance"] != tol:
                raise ValueError("BDF provenance differs")
            np.testing.assert_array_equal(data["time"], a["time"])
            parts.append(a["water"][:, :frames])
    return np.concatenate(parts)[: len(data["m"])]


def models(paths, ablate, stages):
    """Reload each checkpoint and optionally isolate individual experts."""
    yield "physical", DualExchange(), 0, 0, "", "physical"
    for entry in stages:
        order, it = map(int, entry.split("x"))
        yield (
            f"rk{order}i{it}-physical",
            DualStages(order, it),
            0,
            0,
            "",
            "physical",
        )
    for path in paths:
        saved = torch.load(path, weights_only=True, map_location="cpu")
        net = build(saved["config"])
        net.load_state_dict(saved["model"])
        routes = ["learned"]
        if ablate and net.kind == "spatiotemporal":
            routes += ["uniform", "physical", "ff", "gru", "attention"]
        for route in routes:
            net.routing = route
            yield (
                path.stem,
                net.eval(),
                path.stat().st_size,
                saved["seed"],
                digest(path),
                route,
            )


def decoded(file, data, nodes, profile, moment_dtype=np.float32):
    """Validate the stored record before forecasting either vapor."""
    save_population(
        file, data, nodes, organic_profile=profile, moment_dtype=moment_dtype
    )
    m, n, levels, time, volume = reopen(file)
    if not np.isfinite(m).all() or np.any((n > 0) & (m[..., -2:].sum(-1) <= 0)):
        raise ValueError(
            "Each occupied decoded particle needs finite positive dry mass"
        )
    with np.load(file) as a:
        reset = a["reset_mask"].repeat(2, axis=0)
    np.testing.assert_array_equal(reset, data["reset_mask"])
    np.testing.assert_array_equal(levels, data["levels"])
    np.testing.assert_array_equal(time, data["time"])
    np.testing.assert_allclose(n.sum(1), data["n"].sum(1), rtol=1e-13)
    np.testing.assert_allclose(
        (m * n[..., None]).sum(1),
        (data["m"] * data["n"][..., None]).sum(1),
        atol=0,
        rtol=2e-13,
    )
    return tuple(map(torch.from_numpy, (m, n / volume, levels, reset)))


def append_scores(rows, data, panels, pred, gas, case, file, factors):
    """Score paired films, then score each independent reset ordering."""
    label, size, seed, geometry, steps = case
    # Three distinct reset orders and two organic-composition directions.
    groups = [("all", np.ones(len(data["ids"]), dtype=bool))]
    if factors["stratified"]:
        groups += [(str(i), data["ids"] % 6 == i) for i in range(6)]
    for name, group in groups:
        members = group.repeat(2)
        local = dict(
            time=data["time"],
            bootstrap=data.get("bootstrap", False),
            ids=data["ids"][group],
            levels=data["levels"][members],
            reset_mask=data["reset_mask"][members],
        )
        for ref, target, bounds in panels:
            before = len(rows)
            reporting(
                label,
                size,
                pred[members],
                gas[members],
                local,
                target[members],
                bounds[group],
                ref,
                file,
                geometry,
                steps,
                seed,
                rows,
            )
            for row in rows[before:]:
                row.update(factors, reset_order_and_geometry_sign=name)


@torch.inference_mode()
def run(
    root,
    out,
    split,
    geometries,
    blocks,
    families,
    nodes,
    substeps,
    checkpoints,
    stages,
    bdf,
    ablate,
    stratified,
    bootstrap=False,
):
    """Score single-state, free-running trajectories against both reservoirs."""
    rows = []
    for geometry in geometries:
        d, panels = references(root, split, geometry, blocks, families, bdf)
        d["bootstrap"] = bootstrap
        folder = root / "archives" / split / geometry
        folder.mkdir(parents=True, exist_ok=True)
        for n in nodes:
            for profile in [False, True] if n else [False]:
                file = (
                    folder / f"b{blocks}-f{families}-n{n}-p{int(profile)}.npz"
                )
                m, number, levels, mask = decoded(file, d, n, profile)
                for label, net, size, seed, sha, route in models(
                    checkpoints, ablate, stages
                ):
                    for s in substeps:
                        wet, gas = net.rollout(
                            m, number, levels, d["time"], s, mask
                        )
                        bulk = (wet * number[:, None, :, None]).sum(2)
                        case = label, size, seed, geometry, s
                        metadata = dict(
                            nodes=n,
                            organic_profile=profile,
                            plateaus=4 * blocks,
                            routing=route,
                            checkpoint_sha256=sha,
                            stratified=stratified,
                        )
                        append_scores(
                            rows,
                            d,
                            panels,
                            bulk.numpy(),
                            gas.numpy(),
                            case,
                            file,
                            metadata,
                        )
            write_table(rows, out)
            print(geometry, n, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--split",
        choices=("validation", "test", "challenge"),
        default="validation",
    )
    p.add_argument(
        "--geometries", nargs="+", default=["conditioned", "proportional"]
    )
    for key, default in (("blocks", 1), ("families", 64)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--nodes", nargs="+", type=int, default=[0, 3, 5, 8])
    p.add_argument("--substeps", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--stages", nargs="*", default=[])
    p.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    for flag in ("bdf", "ablate", "stratified", "bootstrap"):
        p.add_argument("--" + flag, action="store_true")
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
