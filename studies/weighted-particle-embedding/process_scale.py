"""Measure paired neural substitutions during reversible 32-vapor exchange."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from process_cost import CHECKPOINT, POLICIES, CostBox, forecast, load_models
from stage3_explicit_formation import CLUSTER

SPECIES = 32
ROW_TILE = 32768
DT = 0.125
STEPS = 80
FLOOR = 1e-11
GROUPS = ("condensing",) * 16 + ("evaporating",) * 8 + ("near-steady",) * 8


class ReversibleBox(CostBox):
    """Use mass-fraction activity and conservative implicit vapor exchange."""

    def initialize_reservoirs(self):
        """Set initial particle fractions and saturation concentrations."""
        # Hosts share mass fractions and vary in size.
        fraction = 0.002
        self.mass[:, 1:] = self.mass[:, :1] * (
            fraction / (1 - SPECIES * fraction)
        )
        self.gas.copy_(
            torch.linspace(
                1e-9, 2e-9, SPECIES, device=self.device, dtype=self.dtype
            )
        )
        saturation = self.gas.new_tensor([2.0] * 16 + [0.5] * 8 + [1.0] * 8)
        self.cstar = self.gas / (fraction * saturation)
        # Stable 16-molecule clusters retain the nonvolatile precursor.
        self.cstar[0] = 0
        self.initial = self.mass.sum(0) / self.volume
        self.initial[1:] += self.gas

    def conductance(self, radius, j):
        """Evaluate the original network in bounded batches."""
        if self.uptake is None or len(radius) <= ROW_TILE:
            return super().conductance(radius, j)
        return torch.cat(
            [
                super(ReversibleBox, self).conductance(block, j)
                for block in radius.split(ROW_TILE)
            ]
        )

    def vapor(self, duration, vmax, tile=4):
        """Solve the gas and particle fluxes together for each species."""
        radius, total = self.geometry()
        for start in range(0, SPECIES, tile):
            j = slice(start, start + tile)
            m = self.mass[:, start + 1 : start + 1 + tile]
            kdt = self.conductance(radius, j) * duration
            # Freeze total mass and geometry throughout this half step.
            inv = (1 + kdt * self.cstar[j] / total[:, None]).reciprocal()
            weight = kdt * inv
            released = (weight * self.cstar[j] * (m / total[:, None])).sum(0)
            gas = (self.gas[j] + released / self.volume) / (
                1 + weight.sum(0) / self.volume
            )
            m.copy_(m * inv + weight * gas)
            self.gas[j] = gas
        self.vsteps += 1


def score(trial, ref, dt):
    """Score every observation, including individual and summed phases."""
    detail = {}
    gas_end = SPECIES
    mass_end = 2 * SPECIES + 1

    def phases(obs):
        return np.column_stack(
            (
                obs[:, :mass_end],
                obs[:, :gas_end].sum(1),
                obs[:, gas_end:mass_end].sum(1),
            )
        )

    actual, target = phases(trial), phases(ref)
    errors = np.abs(actual - target)
    percent = 100 * errors / np.maximum(target, FLOOR)
    i, j = np.unravel_index(np.argmax(percent), percent.shape)
    names = (
        [f"gas-{k + 1}" for k in range(SPECIES)]
        + ["dry"]
        + [f"particle-{k + 1}" for k in range(SPECIES)]
        + ["gas-total", "particle-total"]
    )
    detail.update(
        phase_error_pct=percent[i, j],
        worst_phase=names[j],
        worst_time_s=(i + 1) * dt,
        min_reference_kg_m3=target.min(),
        number_error_pct=100 * np.max(np.abs(trial[:, -2] / ref[:, -2] - 1)),
        ledger_error_pct=100 * trial[:, -1].max(),
    )
    for group in dict.fromkeys(GROUPS):
        ids = [k for k, name in enumerate(GROUPS) if name == group]
        detail[group.replace("-", "_") + "_error_pct"] = percent[
            :, ids + [SPECIES + 1 + k for k in ids]
        ].max()
    return detail


def write_csv(path, rows):
    """Install each checkpoint only after all its rows have been written."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def exchange_rows(obs, seed, hosts, births):
    """Save signed net transfer and excursions for each vapor."""
    gas0 = np.linspace(1e-9, 2e-9, SPECIES)
    # Signed net condensation from gas depletion (corrected for
    # precursor incorporated directly into new clusters at each observation).
    gases = obs[:, :SPECIES]
    exchange = (gas0 - gases).copy()
    exchange[:, 0] -= births * CLUSTER / (hosts / 1e12)
    pct = 100 * exchange / gas0
    return [
        dict(
            seed=seed,
            species=k + 1,
            initial_regime=g,
            final_net_condensed_pct=pct[-1, k],
            min_net_condensed_pct=pct[:, k].min(),
            max_net_condensed_pct=pct[:, k].max(),
            first_net_condensed_pct=pct[0, k],
        )
        for k, g in enumerate(GROUPS)
    ]


class ObservedBox(ReversibleBox):
    """Include cumulative births in the trajectory for the transfer budget."""

    def observe(self):
        """Append formation count to the checked inventory."""
        return [*super().observe(), self.births]


class RefinedBox(ObservedBox):
    """Halve the vapor step while retaining the original event clocks."""

    def vapor(self, duration, vmax, tile=4):
        """Refresh geometry and activity halfway through each vapor step."""
        super().vapor(duration / 2, vmax, tile)
        super().vapor(duration / 2, vmax, tile)


def refine_vapor(args, models, opts):
    """Compare standard and halved vapor steps for the physical model."""
    rows = []
    for seed in args.seeds:
        coarse = forecast("physical", seed, models, steps=args.steps, **opts)[1]
        fine_opts = dict(opts, box_type=RefinedBox)
        fine = forecast(
            "physical", seed, models, steps=args.steps, **fine_opts
        )[1]
        rows.append(
            dict(
                hosts=args.hosts,
                seed=seed,
                **score(coarse[:, :-1], fine[:, :-1], args.dt),
            )
        )
        print(
            f"Vapor refinement completed for {args.hosts} hosts, seed {seed}",
            flush=True,
        )
    write_csv(args.output / "vapor-refinement.csv", rows)


@torch.inference_mode()
def main():
    """Time and save one matched block per seed."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--hosts", type=int, required=True)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[70101, 70102, 70103]
    )
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--refine-vapor", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = load_models()
    opts = dict(
        box_type=ObservedBox, hosts=args.hosts, species=SPECIES, dt=args.dt
    )
    if args.refine_vapor:
        refine_vapor(args, models, opts)
        return
    for policy in POLICIES:
        forecast(policy, 70100, models, steps=1, **opts)
    for seed in args.seeds:
        trials = {}
        for policy in np.random.default_rng(seed).permutation(POLICIES):
            start = perf_counter()
            trials[policy] = forecast(
                policy, seed, models, steps=args.steps, **opts
            )
            print(
                f"{seed} {policy} completed in {perf_counter() - start:.2f} s",
                flush=True,
            )
        ref = trials["physical"][1]
        rows = []
        for policy in POLICIES:
            timings, obs = trials[policy]
            rows.append(
                dict(
                    hosts=args.hosts,
                    species=SPECIES,
                    horizon_s=args.steps * args.dt,
                    seed=seed,
                    policy=policy,
                    **timings,
                    **score(obs[:, :-1], ref[:, :-1], args.dt),
                )
            )
        write_csv(args.output / f"trials-{seed}.csv", rows)
        write_csv(
            args.output / f"exchange-{seed}.csv",
            exchange_rows(ref[:, :-1], seed, args.hosts, ref[:, -1]),
        )
    meta = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        hosts=args.hosts,
        density_cm3=1e6,
        volume_cm3=args.hosts / 1e6,
        steps=args.steps,
        dt_s=args.dt,
        seeds=args.seeds,
        species=SPECIES,
        row_tile=ROW_TILE,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        condensation_model=CHECKPOINT,
        collision_model="jump-w4-s901.pt",
        nonvolatile_precursor_species=1,
        initial_saturation=[None] + [2.0] * 15 + [0.5] * 8 + [1.0] * 8,
    )
    (args.output / "settings.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
