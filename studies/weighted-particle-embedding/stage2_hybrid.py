"""Replace selected forecast intervals with coupled physical advances."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage2_population import decoded, fixture, observable_names
from stage2_reference import Clock, advance
from stage2_split import Split


class MappedClock(Clock):
    """Keep original slot indices across physical mergers."""

    def __init__(self, seed, count, remaining, slots, vapor, pair):
        """Restore the stream at the recorded collision index."""
        rng = np.random.default_rng(seed)
        for _ in range(count):
            rng.exponential()
            rng.random()
        rng.exponential()
        super().__init__(rng, remaining, count)
        self.slots = slots.tolist()
        self.vapor, self.pair = vapor, pair

    def event(self, box, i, j, rates):
        """Transport dry-weighted memory to the surviving slot."""
        dry = box.mass[:, -2:].sum(-1)
        a, b = super().event(box, i, j, rates)
        left, right = self.slots[a], self.slots[b]
        frac = dry[a] / (dry[a] + dry[b])
        for h in (self.vapor, self.pair):
            h[left] = h[left] * frac + h[right] * (1 - frac)
            h[right] = 0.0
        del self.slots[b]
        return a, b


def process_kind(process):
    """Choose the physical collision formulation for the requested process."""
    if process in ("both", "water", "organic"):
        return None
    if process in ("charge", "charge-only"):
        return "charged"
    return "constant" if process == "constant" else "brownian"


def correct(sim, s, idx, dt, proc, memory, parts, tol, *, factory=None):
    """Replace an interval on the selected forecast states."""
    nrows = len(idx)
    if not nrows:
        return 0
    selected = s.subset(idx)
    if memory == "reset":
        selected.vapor_memory.zero_()
        selected.pair_memory.zero_()
    elif memory == "reconcile":
        # Observe the evolving forecast once to bring the hidden states to the
        # new physical time. Only memory is adopted from this learned probe.
        if selected.vapor_memory.shape[-1] and proc not in (
            "coag",
            "constant",
            "charge-only",
        ):
            trial = selected.clone()
            sim.vapor(trial, dt, 1)
            selected.vapor_memory = trial.vapor_memory
        if selected.pair_memory.shape[-1]:
            _, selected.pair_memory = sim.pair_rates(selected)
    elif memory != "carry":
        raise ValueError("Unknown memory policy")
    calls = 0
    for local, global_row in enumerate(idx.cpu().numpy()):
        box = sim.boxes[global_row].copy()
        valid = selected.live[local].cpu().numpy()
        box.mass = selected.mass[local].cpu().numpy()[valid].copy()
        box.charge = selected.charge[local].cpu().numpy()[valid].copy()
        box.gas = selected.gas[local].cpu().numpy().copy()
        clock = MappedClock(
            sim.seeds[global_row],
            int(selected.events[local]),
            float(selected.remaining[local]),
            np.flatnonzero(valid),
            selected.vapor_memory[local],
            selected.pair_memory[local],
        )
        for _ in range(parts):
            _, f = advance(
                box,
                clock,
                np.array([dt / parts]),
                tol,
                proc not in ("coag", "constant", "charge-only"),
                process_kind(proc),
                **(
                    {"reservoir": factory.reservoir, "kernel": factory.kernel}
                    if factory is not None
                    else {}
                ),
            )
            calls += f
        for field in ("mass", "charge", "live"):
            getattr(selected, field)[local].zero_()
        slots = torch.tensor(clock.slots, device=sim.device)
        selected.live[local, slots] = True
        selected.mass[local, slots] = torch.as_tensor(
            box.mass, device=sim.device
        )
        selected.charge[local, slots] = torch.as_tensor(
            box.charge, device=sim.device
        )
        selected.gas[local] = torch.as_tensor(box.gas, device=sim.device)
        selected.remaining[local] = clock.remaining
        selected.events[local] = clock.collisions
    s.put(idx, selected)
    return calls


def indicator(sim, s, dt):
    """Measure the predicted phase change relative to available material."""
    net = sim.transfer
    n = s.live / s.volume[:, None]
    state = net.static(s.mass, n)
    q = s.mass[..., : sim.spec.species] / state[0][..., None]
    rate = net.linearization(q, s.gas, state, rate_only=True)
    bulk = (q * state[3][..., None]).sum(1)
    flux = (rate * state[3][..., None]).sum(1).abs() * dt
    return torch.maximum(
        flux / bulk.clamp_min(1e-11), flux / s.gas.clamp_min(1e-11)
    ).amax(-1)


def correction_decision(
    sim, s, dt, policy, counter, age, total, used, threshold, learned
):
    """Choose physical advances from current state and remaining work."""
    check = torch.zeros_like(used, dtype=torch.bool)
    score = torch.zeros_like(s.volume)
    probes = 0
    coarse = None
    interval = int(policy.split("-")[-1]) if "-" in policy else 1
    if policy == "full":
        check[:] = True
    elif policy.startswith("duration-"):
        check.fill_(bool(dt >= interval))
    elif policy.startswith("fixed-"):
        check[:] = (counter + 1) % interval == 0
    elif policy.startswith("jump-"):
        check[:] = age < interval
    elif policy.startswith("single-"):
        _, first, length = policy.split("-")
        check[:] = int(first) <= counter < int(first) + int(length)
    elif policy.startswith(("drive-", "double-", "benefit-")):
        budget = total // interval
        if policy.startswith("drive"):
            score = indicator(sim, s, dt)
        elif policy.startswith("benefit"):
            from stage2_controller import features

            n = s.live / s.volume[:, None]
            context = sim.transfer.static(s.mass, n)
            q = s.mass[..., : sim.spec.species] / context[0][..., None]
            x = features(
                sim.transfer,
                q,
                s.gas,
                context,
                s.volume.new_full(s.volume.shape, dt),
                s.volume.new_full(s.volume.shape, float(age)),
            )
            score = sim.benefit(x)
            threshold = sim.benefit_thresholds[interval]
        else:
            coarse = s.clone()
            fine = s.clone()
            learned(coarse, dt, sim.indices)
            learned(fine, dt, sim.indices, 2)
            # Include count and both volatile phases in a causal defect.
            a, b = [
                sim.observe(x)[:, : 2 * sim.spec.species + 3]
                for x in (coarse, fine)
            ]
            score = ((a - b).abs() / b.abs().clamp_min(1e-11)).amax(-1)
            probes += len(used) * 2
        remaining = budget - used
        check = (remaining > 0) & (
            (score >= threshold) | (remaining >= total - counter)
        )
    elif policy != "none":
        raise ValueError(f"Unknown correction policy {policy}")
    return check, score, coarse, probes


@torch.inference_mode()
def hybrid(
    sim,
    proc,
    policy,
    scheme,
    vapor_steps,
    split_steps,
    threshold,
    memory="reset",
    parts=1,
    tol=1e-9,
    replacement=correct,
    shadow=False,
):
    """Roll forward under a causal, countable physical-correction budget."""
    s = sim.start()
    q0 = s.charge.sum(1)
    expected = s.mass.sum(1) / s.volume[:, None]
    expected[:, : sim.spec.species] += s.gas
    exchange = proc not in ("coag", "constant", "charge-only")
    obs = [sim.observe(s)]
    before = [obs[0]]
    decision = [torch.zeros_like(s.events, dtype=torch.bool)]
    scores = [torch.zeros_like(s.volume)]
    local_errors = [
        torch.zeros(
            (len(s.volume), 2), dtype=s.volume.dtype, device=s.volume.device
        )
    ]
    shadow_calls = 0
    total = int((np.diff(sim.t) > 0).sum())
    used = torch.zeros_like(s.events)
    counter = 0
    leg = 0
    age = 0
    calls = probes = 0
    drift = torch.zeros_like(s.volume[:1])

    def learned(a, dt, idx, halves=1):
        for _ in range(split_steps * halves):
            sim.step(
                a,
                dt / (split_steps * halves),
                idx,
                scheme,
                vapor_steps,
                "joint",
                exchange,
            )

    for dt in np.diff(sim.t):
        check = torch.zeros_like(used, dtype=torch.bool)
        score = torch.zeros_like(s.volume)
        prior = sim.observe(s)
        local_error = local_errors[0]
        if shadow and dt:
            alternative, physical = s.clone(), s.clone()
            learned(alternative, dt, sim.indices)
            shadow_calls += correct(
                sim, physical, sim.indices, dt, proc, "reset", 1, tol
            )
            oracle, tested = sim.observe(physical), sim.observe(alternative)
            count = abs(tested[:, 0] - oracle[:, 0]) / oracle[:, 0].clamp_min(
                1.0
            )
            end = 2 * sim.spec.species + 3
            phase = (
                (tested[:, 1:end] - oracle[:, 1:end]).abs()
                / oracle[:, 1:end].abs().clamp_min(1e-11)
            ).amax(-1)
            local_error = torch.stack((count, phase), -1)
        if dt == 0:
            leg += 1
            age = 0
            gas = torch.where(sim.masks[:, leg], sim.levels[:, leg], s.gas)
            expected[:, : sim.spec.species] += gas - s.gas
            s.gas = gas
        else:
            check, score, coarse, extra_probes = correction_decision(
                sim,
                s,
                dt,
                policy,
                counter,
                age,
                total,
                used,
                threshold,
                learned,
            )
            probes += extra_probes
            if coarse is not None:
                selected = torch.where(~check)[0]
                s.put(selected, coarse.subset(selected))
                probes += int(check.sum())
            elif bool((~check).any()):
                idx = torch.where(~check)[0]
                a = s.subset(idx)
                learned(a, dt, idx)
                s.put(idx, a)
            active = torch.where(check)[0]
            calls += replacement(sim, s, active, dt, proc, memory, parts, tol)
            used += check
            counter += 1
            age += 1
        y = sim.observe(s)
        drift = torch.maximum(drift, sim.audit(s, expected, q0))
        obs.append(y)
        before.append(prior)
        decision.append(check)
        scores.append(score)
        if shadow:
            local_errors.append(local_error)
    if policy.startswith(("drive-", "double-", "benefit-")) and not torch.all(
        used == total // int(policy.split("-")[-1])
    ):
        raise ArithmeticError("Correction budgets differ")
    return dict(
        obs=torch.stack(obs, 1).cpu().numpy(),
        before=torch.stack(before, 1).cpu().numpy(),
        corrected=torch.stack(decision, 1).cpu().numpy(),
        trigger=torch.stack(scores, 1).cpu().numpy(),
        max_drift=float(drift.max()),
        physical_advances=used.cpu().numpy() * parts,
        shadow_errors=torch.stack(local_errors, 1).cpu().numpy()
        if shadow
        else None,
        shadow_calls=shadow_calls,
        physical_rhs_calls=calls,
        diagnostic_step_equivalents=probes,
    )


def execute(a):
    """Save policy decisions and accepted trajectories."""
    keys = [
        (f, m, d, seed)
        for f in a.families
        for d in a.densities
        for m in (0, 1)
        for seed in a.seeds
    ]
    a.out.mkdir(exist_ok=True, parents=True)
    for policy, rep, mem, parts in product(a.policy, a.rep, a.memory, a.parts):
        name = (
            f"{a.process}-{rep}-{Path(a.transfer).stem}"
            f"-{Path(a.kernel).stem}-{policy}-th{a.threshold:g}"
            f"-{mem}-p{parts}-v{a.vapor_steps}-m{a.split_steps}"
        )
        if a.backup_vapor:
            name += f"-gpu-equivalent-p{a.backup_vapor}-s{a.backup_split}"
        if a.shadow:
            name += "-shadow"
        path = a.out / (name + ".npz")
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z["keys"], keys)
            continue
        data = []
        seconds = []
        for start in range(0, len(keys), a.batch):
            k = keys[start : start + a.batch]
            boxes = [
                decoded(
                    fixture(
                        f,
                        m,
                        a.slots,
                        d,
                        a.process,
                        a.charge,
                        a.frames,
                        a.legs,
                    ),
                    rep,
                )
                for f, m, d, _ in k
            ]
            tic = perf_counter()
            sim = Split(
                boxes,
                [x[3] for x in k],
                a.transfer,
                a.kernel,
                a.device,
            )
            if policy.startswith("benefit-"):
                from stage2_controller import load_controller

                sim.benefit, sim.benefit_thresholds = load_controller(
                    a.controller, a.device
                )
            replacement = correct
            if a.backup_vapor:
                from stage2_batched_correction import (
                    host_replacement,
                )

                replacement = host_replacement(
                    sim, a.process, a.backup_vapor, a.backup_split
                )
            data.append(
                hybrid(
                    sim,
                    a.process,
                    policy,
                    a.scheme,
                    a.vapor_steps,
                    a.split_steps,
                    a.threshold,
                    mem,
                    parts,
                    a.tol,
                    replacement,
                    a.shadow,
                )
            )
            seconds.append(perf_counter() - tic)
        results = {
            k: np.concatenate([x[k] for x in data])
            for k in (
                "obs",
                "before",
                "corrected",
                "trigger",
                "physical_advances",
            )
        }
        if a.shadow:
            results["shadow_errors"] = np.concatenate(
                [x["shadow_errors"] for x in data]
            )
            results["shadow_calls"] = sum(x["shadow_calls"] for x in data)
        meta = dict(
            process=a.process,
            charge=a.charge,
            slots=a.slots,
            frames=a.frames,
            legs=a.legs,
            representation=rep,
            transfer=Path(a.transfer).stem,
            kernel=Path(a.kernel).stem,
            scheme=a.scheme,
            vapor_steps=a.vapor_steps,
            split_steps=a.split_steps,
            species_order="joint",
            device=a.device,
            batch=a.batch,
            policy=policy,
            threshold=sim.benefit_thresholds[int(policy.split("-")[-1])]
            if policy.startswith("benefit-")
            else a.threshold,
            memory=mem,
            physical_parts=parts,
            offline_shadow=a.shadow,
            backup_vapor=a.backup_vapor,
            backup_split=a.backup_split,
        )
        np.savez_compressed(
            path,
            keys=keys,
            t=sim.t,
            names=observable_names(sim.spec),
            meta=json.dumps(meta, sort_keys=True),
            max_drift=max(d["max_drift"] for d in data),
            seconds=seconds,
            physical_rhs_calls=sum(d["physical_rhs_calls"] for d in data),
            diagnostic_step_equivalents=sum(
                d["diagnostic_step_equivalents"] for d in data
            ),
            **results,
        )
        print(name, sum(seconds), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    for k, v in (
        ("families", [101, 102, 103, 104]),
        ("densities", [1, 8]),
        ("seeds", [101]),
        ("parts", [1]),
    ):
        p.add_argument("--" + k, nargs="+", type=int, default=v)
    for k, v in (
        (
            "policy",
            [
                "none",
                "full",
                "fixed-4",
                "fixed-16",
                "fixed-64",
                "jump-1",
                "jump-2",
                "drive-4",
                "double-4",
            ],
        ),
        ("rep", ["full"]),
        ("memory", ["reset"]),
    ):
        p.add_argument("--" + k, nargs="+", default=v)
    for k, v in (
        ("frames", 8),
        ("legs", 4),
        ("slots", 16),
        ("batch", 64),
        ("vapor-steps", 4),
        ("split-steps", 1),
    ):
        p.add_argument("--" + k, type=int, default=v)
    for k, v in (
        ("charge", "zero"),
        ("transfer", "parent"),
        ("kernel", "brownian"),
        ("process", "coupled"),
        ("device", "cpu"),
        ("scheme", "strang"),
    ):
        p.add_argument("--" + k, default=v)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--controller", type=Path)
    p.add_argument("--shadow", action="store_true")
    p.add_argument("--backup-vapor", type=int, default=0)
    p.add_argument("--backup-split", type=int, default=2)
    p.add_argument("--tol", type=float, default=1e-9)
    torch.set_num_threads(1)
    execute(p.parse_args())
