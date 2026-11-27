"""Compose vapor and collision operators on a shared particle state."""

from dataclasses import dataclass

import numpy as np
import torch
from stage2_pair import charged_kernel, load_pair, physics_kernel
from stage2_transfer import restore, substep


@dataclass
class State:
    """Store all live tensors required for restart."""

    mass: torch.Tensor
    charge: torch.Tensor
    gas: torch.Tensor
    live: torch.Tensor
    volume: torch.Tensor
    remaining: torch.Tensor
    events: torch.Tensor
    vapor_memory: torch.Tensor
    pair_memory: torch.Tensor

    def clone(self):
        """Copy inventories, event clocks and process memory."""
        return State(**{k: v.clone() for k, v in vars(self).items()})

    def subset(self, idx):
        """Copy the selected boxes for a causal branch or correction."""
        return State(**{k: v[idx].clone() for k, v in vars(self).items()})

    def put(self, idx, other):
        """Install the accepted state for the specified boxes."""
        for key, x in vars(self).items():
            x[idx] = getattr(other, key)


class Split:
    """Compose vapor and pair kernels on one physical particle population."""

    def __init__(
        self, boxes, seeds, transfer="physical", kernel="brownian", device="cpu"
    ):
        """Bind processes and allocate reproducible event tapes."""
        self.device = torch.device(device)
        self.spec = boxes[0].spec
        self.transfer = restore(self.spec, transfer, device)
        self.kernel = kernel
        self.pair = (
            None
            if kernel
            in (
                "brownian",
                "constant",
                "off",
                "charged",
                "hard-sphere",
                "charged-analytic",
                "hard-analytic",
            )
            else load_pair(kernel, device)
        )
        self.rho = torch.as_tensor(self.spec.density, device=device)
        self.seeds = np.asarray(seeds)
        self.boxes = boxes
        self.initial_slots = len(boxes[0].mass)
        p = self.initial_slots
        tape = []
        for seed in seeds:
            gen = np.random.default_rng(seed)
            tape.append(
                [(gen.exponential(), gen.random()) for _ in range(p + 1)]
            )
        self.tape = torch.as_tensor(np.array(tape), device=device)
        self.i, self.j = torch.triu_indices(p, p, 1, device=device)
        self.indices = torch.arange(len(boxes), device=device)
        self.levels = torch.as_tensor(
            np.stack([b.levels for b in boxes]), device=device
        )
        self.masks = torch.as_tensor(
            np.stack([b.masks for b in boxes]), device=device
        )
        self.t = boxes[0].t
        if any(not np.array_equal(b.t, self.t) for b in boxes):
            raise ValueError("Batch observation clocks differ")

    def start(self):
        """Restore physical state from the original weighted particles."""

        def stack(field):
            return torch.as_tensor(
                np.stack([getattr(b, field) for b in self.boxes]),
                device=self.device,
            )

        m, z, gas, volume = map(stack, ("mass", "charge", "gas", "volume"))
        number = torch.ones_like(m[..., 0]) / volume[:, None]
        return State(
            m,
            z,
            gas,
            torch.ones_like(z, dtype=torch.bool),
            volume,
            self.tape[:, 0, 0].clone(),
            torch.zeros_like(z[:, 0]),
            self.transfer.start(number),
            self.pair.start(number)
            if self.pair
            else number.new_zeros((*number.shape, 0)),
        )

    def observe(self, s):
        """Measure shared native and learned forecast observables."""
        n = s.live / s.volume[:, None]
        r = (
            (3 * (s.mass / self.rho).sum(-1) / (4 * torch.pi))
            .clamp_min(0)
            .pow(1 / 3)
        )
        total = s.live.sum(1) / s.volume
        bins = torch.bucketize(
            r.contiguous(), r.new_tensor([50e-9, 90e-9, 150e-9]), right=True
        )
        counts = torch.zeros(
            (len(r), 4), device=r.device, dtype=r.dtype
        ).scatter_add_(1, bins, n)
        charges = torch.zeros_like(counts).scatter_add_(1, bins, n * s.charge)
        return torch.cat(
            (
                total[:, None],
                (s.mass * n[..., None]).sum(1),
                s.gas,
                ((r * n).sum(-1) / total.clamp_min(1e-30))[:, None],
                (s.charge * n).sum(-1, keepdim=True),
                counts,
                charges,
                (s.charge.abs() * n).sum(-1, keepdim=True),
            ),
            -1,
        )

    def vapor(self, s, dt, steps, order="joint", net=None):
        """Advance all selected vapor species against finite gas reservoirs."""
        net = self.transfer if net is None else net
        n = s.live / s.volume[:, None]
        fixed = net.static(s.mass, n)
        dry = fixed[0]
        q = s.mass[..., : self.spec.species] / dry[..., None]
        hh = s.vapor_memory if net is self.transfer else net.start(n)
        h = torch.as_tensor(dt, dtype=s.gas.dtype, device=s.gas.device)
        h = h.expand(len(s.gas)) / steps
        for _ in range(steps):
            q, s.gas, hh = substep(net, q, s.gas, fixed, h, hh, order)
        s.mass[..., : self.spec.species] = q * dry[..., None]
        if net is self.transfer:
            s.vapor_memory = hh

    def pair_rates(self, s):
        """Return the rates of all feasible binary mergers."""
        r = (
            (3 * (s.mass / self.rho).sum(-1) / (4 * torch.pi))
            .clamp_min(1e-90)
            .pow(1 / 3)
        )
        total = s.mass.sum(-1)
        i, j = self.i, self.j
        live = s.live[:, i] & s.live[:, j]
        if self.pair:
            return self.pair(r, total, s.charge, s.live, i, j, s.pair_memory)
        if self.kernel == "constant":
            return live * torch.full_like(r[:, i], 2e-15), s.pair_memory
        if self.kernel in ("charged-analytic", "hard-analytic"):
            return charged_kernel(
                torch.where(s.live, r, 80e-9),
                torch.where(s.live, total, 2e-18),
                s.charge,
                i,
                j,
                hard=self.kernel == "hard-analytic",
            ) * live, s.pair_memory
        if self.kernel in ("charged", "hard-sphere"):
            from stage2_population import collision_kernel

            values = []
            for m, z, v in zip(
                s.mass.cpu().numpy(),
                s.charge.cpu().numpy(),
                s.live.cpu().numpy(),
                strict=True,
            ):
                # Inactive slots have zero hazard and need finite native input.
                m = m.copy()
                m[~v, -1] = 1e-18
                values.append(
                    collision_kernel(
                        m,
                        z,
                        self.spec,
                        self.kernel,
                        i.cpu().numpy(),
                        j.cpu().numpy(),
                    )
                )
            return torch.as_tensor(
                np.stack(values), device=self.device
            ) * live, s.pair_memory
        return physics_kernel(
            torch.where(s.live, r, 80e-9),
            torch.where(s.live, total, 2e-18),
            i,
            j,
        ) * live, s.pair_memory

    @staticmethod
    def merge_memory(h, row, left, right, m1, m2):
        """Transport hidden states in proportion to dry mass."""
        if h.shape[-1]:
            shape = (-1,) + (1,) * (h.ndim - 2)
            w = (m1 / (m1 + m2)).reshape(shape)
            h[row, left] = w * h[row, left] + (1 - w) * h[row, right]
            h[row, right] = 0.0

    def merge_selected(self, s, row, idx, cdf):
        """Consume one taped pair draw and conserve each material and charge."""
        hazard = cdf[:, -1]
        draw = self.tape[idx[row], s.events[row], 1] * hazard[row]
        choice = torch.searchsorted(
            cdf[row].contiguous(), draw[:, None].contiguous(), right=True
        ).squeeze(-1)
        a, b = self.i[choice], self.j[choice]
        m1, m2 = [s.mass[row, k, -2:].sum(-1) for k in (a, b)]
        self.merge_memory(s.vapor_memory, row, a, b, m1, m2)
        self.merge_memory(s.pair_memory, row, a, b, m1, m2)
        s.mass[row, a] += s.mass[row, b]
        s.charge[row, a] += s.charge[row, b]
        s.mass[row, b] = 0.0
        s.charge[row, b] = 0
        s.live[row, b] = False
        s.events[row] += 1
        s.remaining[row] = self.tape[idx[row], s.events[row], 0]

    def coag(self, s, dt, idx):
        """Consume sampled events through a closed collision interval."""
        if self.kernel == "off":
            return
        left_time = torch.full_like(s.volume, dt)
        while True:
            rates, h = self.pair_rates(s)
            cdf = (rates / s.volume[:, None]).cumsum(-1)
            if s.pair_memory.shape[-1]:
                s.pair_memory = torch.where(
                    (left_time > 0)[:, None, None], h, s.pair_memory
                )
            hazard = cdf[:, -1]
            event = (
                (left_time > 0)
                & (hazard * left_time >= s.remaining)
                & (hazard > 0)
            )
            s.remaining = torch.where(
                event, s.remaining, s.remaining - hazard * left_time
            )
            if not bool(event.any()):
                break
            row = torch.where(event)[0]
            left_time = torch.where(
                event, left_time - s.remaining / hazard.clamp_min(1e-300), 0.0
            )
            self.merge_selected(s, row, idx, cdf)

    def step(
        self,
        s,
        dt,
        idx,
        scheme="strang",
        vapor_steps=1,
        species_order="joint",
        exchange=True,
    ):
        """Apply one ordered composition of the two physical processes."""
        if scheme == "hazard-linear":
            from stage2_hazard import advance

            return advance(
                self, s, dt, idx, vapor_steps, species_order, exchange
            )
        if scheme not in ("lie", "reverse", "strang"):
            raise ValueError("Unknown composition")
        if exchange and scheme in ("lie", "strang"):
            self.vapor(
                s,
                dt if scheme == "lie" else 0.5 * dt,
                vapor_steps,
                species_order,
            )
        self.coag(s, dt, idx)
        if exchange and scheme in ("reverse", "strang"):
            self.vapor(
                s,
                dt if scheme == "reverse" else 0.5 * dt,
                vapor_steps,
                species_order,
            )

    def audit(self, s, expected, q0):
        """Check number, charge and material budgets."""
        n = s.live.sum(1)
        if not torch.all(n == self.initial_slots - s.events) or not torch.all(
            s.charge.sum(1) == q0
        ):
            raise ArithmeticError("Number or charge ledger diverged")
        material = (s.mass * s.live[..., None]).sum(1) / s.volume[:, None]
        material[:, : self.spec.species] += s.gas
        return (
            (material - expected).abs() / expected.abs().clamp_min(1e-11)
        ).amax()

    @torch.inference_mode()
    def rollout(
        self,
        scheme="strang",
        vapor_steps=1,
        split_steps=1,
        species_order="joint",
        exchange=True,
    ):
        """Advance all forcing legs and retain the trajectories."""
        s = self.start()
        expected = s.mass.sum(1) / s.volume[:, None]
        expected[:, : self.spec.species] += s.gas
        q0 = s.charge.sum(1)
        report = [self.observe(s)]
        drift = s.volume.new_zeros(())
        leg = 0
        for dt in np.diff(self.t):
            if dt == 0:
                leg += 1
                nextgas = torch.where(
                    self.masks[:, leg], self.levels[:, leg], s.gas
                )
                expected[:, : self.spec.species] += nextgas - s.gas
                s.gas = nextgas
            else:
                for _ in range(split_steps):
                    self.step(
                        s,
                        dt / split_steps,
                        self.indices,
                        scheme,
                        vapor_steps,
                        species_order,
                        exchange,
                    )
                    drift = torch.maximum(drift, self.audit(s, expected, q0))
            report.append(self.observe(s))
        return torch.stack(report, 1).cpu().numpy(), float(drift), s
