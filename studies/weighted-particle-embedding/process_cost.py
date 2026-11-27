"""Time one neural substitution at a time in an explicit three-process box."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.constants import Avogadro
from stage2_conductance import Conductance
from stage3_explicit_formation import CLUSTER, MOLAR, Source
from stage3_one_box import Box

HERE = Path(__file__).resolve().parent
POLICIES = ("physical", "collision", "condensation", "nucleation")
CHECKPOINT = "conductance-features-w128-shared.pt"
ACTIVATION = 1e-6  # s^-1 per precursor molecule
GAS_SCALE = 1e-9  # kg/m3
RATE_SCALE = 1e9  # births/m3/s
STEPS = 8
HOSTS = 32768


def physical_source(gas):
    """Activation-law number production in births per cubic metre per second."""
    return gas * (ACTIVATION * Avogadro / MOLAR)


class CostBox(Box):
    """Use identical gas reservoirs and individual newborns in every policy."""

    def __init__(
        self, policy, seed, uptake, source_net, hosts=HOSTS, species=2
    ):
        """Start a paired forecast with finite precursor and organic vapor."""
        super().__init__(
            hosts,
            species,
            1e12,
            "neutral",
            "cuda",
            901003,
            seed,
            "learned" if policy == "collision" else "physical",
        )
        self.initialize_reservoirs()
        self.births = 0
        self.source = Source(seed + 100000)
        self.source_net = source_net if policy == "nucleation" else None
        self.uptake = uptake if policy == "condensation" else None

    def initialize_reservoirs(self):
        """Set the two finite vapor reservoirs of the irreversible benchmark."""
        self.gas.copy_(self.gas.new_tensor([1e-9, 2e-9]))
        self.initial[1:] = self.gas

    def conductance(self, radius, j):
        """Evaluate the physical or learned transition factor for uptake."""
        if self.uptake is None:
            return super().conductance(radius, j)
        kn = self.path[j] / radius[:, None]
        factor = self.uptake.feature_factor(kn, self.alpha[j])[0]
        return 4 * math.pi * radius[:, None] * self.diff[j] * factor

    def inject(self, dt):
        """Draw unit-weight births and debit their precursor mass."""
        gas = self.gas[:1]
        if self.source_net is None:
            rate = physical_source(gas)
        else:
            rate = self.source_net(gas / GAS_SCALE) * RATE_SCALE
        count = self.source.count(float(rate.item()) * self.volume * dt)
        if count:
            debit = count * CLUSTER / self.volume
            if debit > float(gas.item()):
                raise ArithmeticError("Source exhausts precursor")
            newborns = self.mass.new_zeros((count, self.mass.shape[1]))
            newborns[:, 1] = CLUSTER
            self.mass = torch.cat((self.mass, newborns))
            self.z = torch.cat((self.z, self.z.new_zeros(count)))
            gas -= debit
            self.births += count

    def observe(self):
        """Check conservation and record each phase inventory and count."""
        inventory = self.mass.sum(0) / self.volume
        total = inventory.clone()
        total[1:] += self.gas
        drift = ((total / self.initial - 1).abs().max()).item()
        if len(self.mass) != self.n0 + self.births - self.collisions:
            raise ArithmeticError("Particle number ledger failed")
        if not (drift < 1e-10 and torch.all(self.gas >= 0)):
            raise ArithmeticError("Material ledger failed")
        return [*self.gas.tolist(), *inventory.tolist(), len(self.mass), drift]


def load_models():
    """Restore the two fitted heads and fit the linear activation law."""
    state = torch.load(
        HERE / "examples/stage-2" / CHECKPOINT, weights_only=True
    )
    uptake = Conductance(state["kind"], state["width"])
    uptake.load_state_dict(state["heads"]["shared"])
    # Fit a one-weight, zero-intercept net to the prescribed linear law.
    train = torch.logspace(-2, 1, 64, dtype=torch.float64)[:, None]
    labels = physical_source(train * GAS_SCALE) / RATE_SCALE
    net = torch.nn.Linear(1, 1, bias=False).double()
    net.weight.copy_(torch.linalg.lstsq(train, labels).solution)
    check = torch.logspace(-1.95, 0.95, 39, dtype=torch.float64)[:, None]
    truth = physical_source(check * GAS_SCALE)
    if not (net(check) * RATE_SCALE / truth - 1).abs().max() < 1e-12:
        raise ArithmeticError("Fitted activation rate failed validation")
    return uptake.cuda().eval(), net.cuda().eval()


def forecast(
    policy,
    seed,
    models,
    *,
    steps=STEPS,
    dt=1 / STEPS,
    box_type=CostBox,
    hosts=HOSTS,
    species=2,
):
    """Time the split steps and record observations for one forecast."""
    box = box_type(policy, seed, *models, hosts, species)
    clocks = dict(condensation=0.0, collision=0.0, nucleation=0.0)
    trajectory = []

    def time_step(name, advance, *args):
        start = perf_counter()
        advance(*args)
        torch.cuda.synchronize()
        clocks[name] += (perf_counter() - start) * 1000

    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(steps):
        time_step("condensation", box.vapor, dt / 2, 0)
        time_step("nucleation", box.inject, dt)
        time_step("collision", box.collide, dt)
        time_step("condensation", box.vapor, dt / 2, 0)
        trajectory.append(box.observe())
    torch.cuda.synchronize()
    total_ms = (perf_counter() - start) * 1000
    row = {f"{key}_ms": value for key, value in clocks.items()}
    row.update(
        total_ms=total_ms,
        other_ms=total_ms - sum(clocks.values()),
        births=box.births,
        mergers=box.collisions,
    )
    return row, np.array(trajectory)


@torch.inference_mode()
def main():
    """Run matched trials and save process times and forecast errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    models = load_models()
    for policy in POLICIES:
        forecast(policy, 70100, models)

    rows = []
    order = np.random.default_rng(8725)
    for i in range(args.repeats):
        seed = 70101 + i
        trials = {
            p: forecast(p, seed, models) for p in order.permutation(POLICIES)
        }
        ref = trials["physical"][1]
        for p in POLICIES:
            row, obs = trials[p]
            if np.min(ref[:, :5]) <= 1e-11:
                raise ValueError(
                    "Phase concentration below relative-error floor"
                )
            phase = np.max(np.abs(obs[:, :5] / ref[:, :5] - 1)) * 100
            number = np.max(np.abs(obs[:, 5] / ref[:, 5] - 1)) * 100
            rows.append(
                dict(
                    policy=p,
                    seed=seed,
                    **row,
                    phase_error_pct=phase,
                    number_error_pct=number,
                    ledger_error_pct=obs[:, 6].max() * 100,
                )
            )
    args.folder.mkdir(parents=True, exist_ok=True)
    with (args.folder / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        hosts=HOSTS,
        steps=STEPS,
        horizon_s=1,
        repeats=args.repeats,
        condensation_model=CHECKPOINT,
        collision_model="jump-w4-s901.pt",
        source_weight=models[1].weight.item(),
    )
    (args.folder / "settings.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
