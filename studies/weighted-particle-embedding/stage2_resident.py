"""Replay resident CUDA vapor and collision graphs."""

import numpy as np
import torch
from stage2_split import Split
from stage2_transfer import substep


class ResidentSplit(Split):
    """Keep live particles and event clocks on the device."""

    def __init__(
        self,
        boxes,
        seeds,
        transfer,
        kernel,
        vapor_steps,
        split_steps,
        event_chunk=2,
    ):
        """Bind CUDA graphs to fixed state addresses."""
        super().__init__(boxes, seeds, transfer, kernel, "cuda")
        self.vapor_steps = vapor_steps
        self.split_steps = split_steps
        self.event_chunk = event_chunk
        self.work = self.start()
        intervals = np.diff(self.t)
        if not np.allclose(
            intervals[intervals > 0], self.t[1] - self.t[0], rtol=2e-13, atol=0
        ):
            raise ValueError(
                "Resident capture requires uniform positive time steps"
            )
        self.left_time = torch.zeros_like(self.work.volume)
        self.dt = self.work.volume.new_full(
            self.work.volume.shape, (self.t[1] - self.t[0]) / split_steps
        )
        self.half = self.dt / (2 * vapor_steps)
        self.vapor_graph = self.capture(self.vapor_body)
        if kernel != "off":
            self.event_graph = self.capture(self.event_body)
        self.reset()

    def reset(self):
        """Restore particles, residual hazards and hidden states."""
        fresh = self.start()
        for k, x in vars(self.work).items():
            x.copy_(getattr(fresh, k))
        self.left_time.zero_()

    def capture(self, body):
        """Warm and capture one resident process operation."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
            body()
        torch.cuda.current_stream().wait_stream(stream)
        self.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        self.reset()
        return graph

    def vapor_body(self):
        """Advance the conservative vapor graph."""
        s = self.work
        n = s.live / s.volume[:, None]
        static = self.transfer.static(s.mass, n)
        q = s.mass[..., : self.spec.species] / static[0][..., None]
        h, g = s.vapor_memory, s.gas
        for _ in range(self.vapor_steps):
            q, g, h = substep(self.transfer, q, g, static, self.half, h)
        s.mass[..., : self.spec.species].copy_(q * static[0][..., None])
        s.gas.copy_(g)
        s.vapor_memory.copy_(h)

    def event_body(self):
        """Consume candidate events while preserving residual time."""
        s = self.work
        rows = self.indices
        for _ in range(self.event_chunk):
            rates, h = self.pair_rates(s)
            pending = self.left_time > 0
            cdf = (rates / s.volume[:, None]).cumsum(-1)
            hazard = cdf[:, -1]
            event = (
                pending
                & (hazard > 0)
                & (self.left_time * hazard >= s.remaining)
            )
            if s.pair_memory.shape[-1]:
                s.pair_memory.copy_(
                    torch.where(pending[:, None, None], h, s.pair_memory)
                )
            u = self.tape[rows, s.events, 1] * hazard
            which = (cdf <= u[:, None]).sum(-1).clamp_max(len(self.i) - 1)
            left, right = self.i[which], self.j[which]
            m1 = s.mass[rows, left]
            m2 = s.mass[rows, right]
            dry1 = m1[:, -2:].sum(-1)
            dry2 = m2[:, -2:].sum(-1)
            frac = dry1 / (dry1 + dry2).clamp_min(1e-100)
            for memory in (s.vapor_memory, s.pair_memory):
                if memory.shape[-1]:
                    shape = (-1,) + (1,) * (memory.ndim - 2)
                    weight = frac.reshape(shape)
                    valid = event.reshape(shape)
                    first, second = memory[rows, left], memory[rows, right]
                    memory[rows, left] = torch.where(
                        valid, weight * first + (1 - weight) * second, first
                    )
                    memory[rows, right] = torch.where(valid, 0.0, second)
            s.mass[rows, left] = torch.where(event[:, None], m1 + m2, m1)
            s.mass[rows, right] = torch.where(event[:, None], 0.0, m2)
            z1, z2 = s.charge[rows, left], s.charge[rows, right]
            s.charge[rows, left] = torch.where(event, z1 + z2, z1)
            s.charge[rows, right] = torch.where(event, 0, z2)
            s.live[rows, right] = s.live[rows, right] & ~event
            elapsed = torch.where(
                event, s.remaining / hazard.clamp_min(1e-300), self.left_time
            )
            residual = s.remaining - hazard * self.left_time
            s.events.add_(event)
            s.remaining.copy_(
                torch.where(event, self.tape[rows, s.events, 0], residual)
            )
            self.left_time.sub_(elapsed).clamp_min_(0.0)

    def physical_replacement(self, state, interval, process, memory, parts):
        """Advance the accepted state with the independent coupled event ODE."""
        from stage2_hybrid import correct

        return correct(
            self, state, self.indices, interval, process, memory, parts, 1e-9
        )

    def advance_neural(self, exchange):
        """Replay the resident split graphs, exhausting every event clock."""
        for _ in range(self.split_steps):
            if exchange:
                self.vapor_graph.replay()
            if self.kernel != "off":
                self.left_time.copy_(self.dt)
                self.event_graph.replay()
                while bool((self.left_time > 0).any()):
                    self.event_graph.replay()
            if exchange:
                self.vapor_graph.replay()

    @torch.inference_mode()
    def rollout_resident(
        self,
        exchange=True,
        policy="none",
        process="coupled",
        parts=1,
        memory="reset",
    ):
        """Replay until every box reaches each requested observation."""
        self.reset()
        s = self.work
        expected = s.mass.sum(1) / s.volume[:, None]
        expected[:, : self.spec.species] += s.gas
        initial_charge = s.charge.sum(1)
        out = [self.observe(s)]
        drifts = []
        counts = []
        leg = age = counter = 0
        self.physical_rhs = self.physical_steps = 0
        prefix, _, suffix = policy.partition("-")
        if prefix not in ("none", "full", "fixed", "jump"):
            raise ValueError("Unsupported resident physical correction policy")
        period = int(suffix) if suffix else 1
        for interval in np.diff(self.t):
            if interval == 0:
                leg += 1
                age = 0
                new = torch.where(
                    self.masks[:, leg], self.levels[:, leg], s.gas
                )
                expected[:, : self.spec.species] += new - s.gas
                s.gas.copy_(new)
            else:
                physical = (
                    prefix == "full"
                    or (prefix == "jump" and age < period)
                    or (prefix == "fixed" and (counter + 1) % period == 0)
                )
                if physical:
                    self.physical_rhs += self.physical_replacement(
                        s, interval, process, memory, parts
                    )
                    self.physical_steps += parts * len(self.indices)
                age += 1
                counter += 1
                if not physical:
                    self.advance_neural(exchange)
            value = self.observe(s)
            material = value[:, 1 : self.spec.species + 3].clone()
            material[:, : self.spec.species] += s.gas
            drifts.append(
                (
                    (material - expected).abs()
                    / expected.abs().clamp_min(1e-11)
                ).max()
            )
            counts.append(
                (
                    (s.live.sum(-1) != self.initial_slots - s.events)
                    | (s.charge.sum(-1) != initial_charge)
                ).any()
            )
            out.append(value)
        if bool(torch.stack(counts).any()):
            raise ArithmeticError("Resident event ledger diverged")
        return (
            torch.stack(out, 1).cpu().numpy(),
            float(torch.stack(drifts).max()),
            s,
        )
