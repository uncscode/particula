"""Test selective physical ODE steps in reversible neural forecasts."""

import argparse
import time
from pathlib import Path

import torch
from material_bank import mixture
from stage2_hybrid import correct, indicator
from stage3_multirate import run
from stage3_species_clocks import SpeciesClocks

FIXED_CORRECTIONS = {
    "none": (),
    "early": (0,),
    "late": (7,),
    "every-fourth": (0, 4),
    "first-two": (0, 1),
    "every-other": (0, 2, 4, 6),
    "all": tuple(range(8)),
}


class LiveODE(SpeciesClocks):
    """Carry live mass, gas and residual hazard through physical steps."""

    def step(self, state, dt, idx, *args):
        """Choose a correction from currently available state and pulse age."""
        opts = self.options
        tick = self.work.get("forecast_observation_steps", 0)
        age = tick % 8
        if age == 0:
            self.used = torch.zeros_like(state.events, dtype=torch.bool)
        policy = opts.refresh
        gate = torch.zeros_like(self.used)
        if policy in FIXED_CORRECTIONS:
            gate[:] = age in FIXED_CORRECTIONS[policy]
        elif policy == "indicator":
            gate = (indicator(self, state, dt) > opts.threshold) & ~self.used
        self.used |= gate
        self.work["forecast_observation_steps"] = tick + 1
        native = idx[gate]
        if len(native):
            tic = time.perf_counter()
            calls = correct(
                self, state, native, dt, opts.process, "carry", 1, opts.ref_tol
            )
            self.work["physical_rhs_calls"] = (
                self.work.get("physical_rhs_calls", 0) + calls
            )
            self.work["physical_box_intervals"] = self.work.get(
                "physical_box_intervals", 0
            ) + len(native)
            self.work["physical_wall_s"] = (
                self.work.get("physical_wall_s", 0.0)
                + time.perf_counter()
                - tic
            )
        if not bool(gate.any()):
            return super().step(state, dt, idx, *args)
        if bool(gate.all()):
            return
        other = idx[~gate]
        sub = state.subset(other)
        full_inventory = self.initial_inventory
        try:
            self.initial_inventory = full_inventory[other]
            super().step(sub, dt, other, *args)
            state.put(other, sub)
        finally:
            self.initial_inventory = full_inventory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    base = Path(__file__).resolve().parent / "examples" / "stage-2"
    p.add_argument(
        "--transfer", default=str(base / "conductance-rational-w0-separate.pt")
    )
    p.add_argument("--kernel", default=str(base / "jump-w4-s901.pt"))
    p.add_argument(
        "--policies",
        nargs="+",
        choices=[*FIXED_CORRECTIONS, "indicator"],
        default=["none", "early", "late", "indicator", "all"],
    )
    p.add_argument("--species", type=int, nargs="+", default=[2, 8])
    p.add_argument("--families", nargs="+", type=int, default=[85001, 85002])
    p.add_argument("--mix", nargs="+", type=int, default=[0, 1])
    p.add_argument("--densities", nargs="+", type=int, default=[1, 8])
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--start", type=int, default=331001)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vtol", type=float, default=0.01)
    p.add_argument("--ctol", type=float, default=0.04)
    p.add_argument("--ref-tol", type=float, default=1e-8)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument(
        "--mode", choices=["joint", "strang", "organic-outer"], default="joint"
    )
    p.add_argument("--organic-groups", type=int, default=1)
    a = p.parse_args()
    a.slots, a.chemistry, a.process, a.charge, a.device = (
        8,
        9177,
        "coupled",
        "zero",
        "cpu",
    )
    a.plateau, a.event_aware, a.block = 0, True, a.mode != "joint"
    a.water_steps, a.organic_steps = 2, 1
    torch.set_num_threads(1)
    for species in a.species:
        a.reference_process = f"reversible-s{species}-c{a.chemistry}"
        for policy in a.policies:
            a.refresh = policy
            tag = f"{a.reference_process}-ode-{policy}-{a.mode}"
            tag += "-" + Path(a.transfer).stem
            if a.refresh == "indicator":
                tag += f"-gate{a.threshold:g}"
            run(a, LiveODE, mixture(species, a.chemistry), tag)
