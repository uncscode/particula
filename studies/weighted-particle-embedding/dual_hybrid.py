"""Use implicit transfer steps after a change in a vapor reservoir."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_schedule_score import append_scores, decoded, references
from dual_stages import DualStages, build
from dual_transfer import DualExchange
from node_transfer import write_table
from projection_study import digest


def guarded_rollout(
    model,
    reference,
    m,
    n,
    levels,
    mask,
    t,
    guard,
    steps,
    warm,
    nodes=False,
    guard_mask=None,
):
    """Use the known elapsed time since the last prescribed gas change."""
    ctx = model.static(m, n)
    q = m[..., :2] / ctx[0][..., None]
    gas = levels[:, 0]
    hidden = model.start(n)
    physical = reference.start(n)

    def observe(q):
        return (
            q * ctx[0][..., None] if nodes else (q * ctx[3][..., None]).sum(1)
        )

    wet = [observe(q)]
    gases = [gas]
    leg, age, guarded, learned = 0, 0, 0, 0
    for i, dt in enumerate(np.diff(t)):
        if dt == 0:
            leg += 1
            gas = torch.where(mask[:, leg], levels[:, leg], gas)
            age = 0
        else:
            h = gas.new_full((len(gas),), dt)
            protected = age < guard if guard_mask is None else guard_mask[i]
            if protected:
                if warm:
                    _, _, hidden = model.step(q, gas, ctx, h, hidden)
                    learned += 1
                for _ in range(steps):
                    q, gas, physical = reference.step(
                        q, gas, ctx, h / steps, physical
                    )
                guarded += 1
            else:
                q, gas, hidden = model.step(q, gas, ctx, h, hidden)
                learned += 1
            age += 1
        wet.append(observe(q))
        gases.append(gas)
    return torch.stack(wet, 1), torch.stack(gases, 1), guarded, learned


def heads(files):
    """Include the uncorrected single-step solver at identical work rules."""
    yield "uncorrected", DualExchange(), 0, 0, ""
    for p in files:
        state = torch.load(p, weights_only=True, map_location="cpu")
        m = build(state["config"])
        m.load_state_dict(state["model"])
        yield p.stem, m, p.stat().st_size, state["seed"], digest(p)


@torch.inference_mode()
def run(
    root,
    out,
    files,
    split,
    geometries,
    blocks,
    families,
    nodes,
    guards,
    physical_steps,
    history,
    bdf,
    bootstrap,
):
    """Score complete free forecasts and count the expensive advances."""
    rows = []
    ref = DualStages(3, 3)
    for geom in geometries:
        d, panels = references(root, split, geom, blocks, families, bdf)
        d["bootstrap"] = bootstrap
        dst = root / "hybrid-records" / split / geom
        dst.mkdir(parents=True, exist_ok=True)
        file = dst / f"b{blocks}-f{families}-n{nodes}.npz"
        m, n, levels, mask = decoded(file, d, nodes, True)
        for name, net, size, seed, sha in heads(files):
            for guard in guards:
                for strategy in history if net.cell and guard else ["hold"]:
                    w, g, exact, steps = guarded_rollout(
                        net,
                        ref,
                        m,
                        n,
                        levels,
                        mask,
                        d["time"],
                        guard,
                        physical_steps,
                        strategy == "warm",
                    )
                    case = (name, size, seed, geom, 1)
                    meta = dict(
                        nodes=nodes,
                        organic_profile=True,
                        plateaus=4 * blocks,
                        routing="reset",
                        checkpoint_sha256=sha,
                        stratified=True,
                        guard_frames=guard,
                        hidden_at_guard=strategy,
                        implicit_frames=exact,
                        implicit_substeps=exact * physical_steps,
                        cheap_model_calls=steps,
                        advancing_frames=int((np.diff(d["time"]) > 0).sum()),
                    )
                    append_scores(
                        rows, d, panels, w.numpy(), g.numpy(), case, file, meta
                    )
            write_table(rows, out)
            print(geom, name, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("files", type=Path, nargs="*")
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--geometries", nargs="+", default=["conditioned", "proportional"]
    )
    for k, v in (
        ("blocks", 3),
        ("families", 64),
        ("nodes", 5),
        ("physical-steps", 4),
    ):
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--guards", type=int, nargs="+", default=[0, 1, 2, 4, 8, 20])
    p.add_argument(
        "--history",
        nargs="+",
        choices=["hold", "warm"],
        default=["hold", "warm"],
    )
    p.add_argument("--bdf", action="store_true")
    p.add_argument("--bootstrap", action="store_true")
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
