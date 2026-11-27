"""Replace selected neural steps with refined device physics."""

import torch
from stage2_resident import ResidentSplit
from stage2_transfer import restore


class PhysicalBackup(ResidentSplit):
    """Capture a physical vapor ODE and collision sampler on the live state."""

    def bind_physics(self, process, substeps, split_steps):
        """Capture the physical graphs and reset the forecast."""
        self.backup_model = restore(self.spec, "radau-3-4", "cuda")
        self.backup_vapor = substeps
        self.backup_split = split_steps
        duration = self.t[1] - self.t[0]
        self.backup_dt = torch.full_like(self.dt, duration / split_steps)
        self.backup_half = self.backup_dt / (2 * substeps)
        self.phys_vapor_graph = self.capture(self.physical_vapor_body)
        self.has_physical_events = process not in ("both", "water", "organic")
        if self.has_physical_events:
            pair, kernel = self.pair, self.kernel
            self.pair = None
            self.kernel = {
                "charge": "charged-analytic",
                "charge-only": "charged-analytic",
                "constant": "constant",
            }.get(process, "brownian")
            try:
                self.phys_event_graph = self.capture(self.event_body)
            finally:
                self.pair, self.kernel = pair, kernel

    def physical_vapor_body(self):
        """Advance one physical Radau microstep."""
        s = self.work
        number = s.live / s.volume[:, None]
        ctx = self.backup_model.static(s.mass, number)
        q = s.mass[..., : self.spec.species] / ctx[0][..., None]
        q, gas, _ = self.backup_model.step(
            q, s.gas, ctx, self.backup_half, self.backup_model.start(number)
        )
        s.mass[..., : self.spec.species].copy_(q * ctx[0][..., None])
        s.gas.copy_(gas)

    def physical_replacement(self, state, interval, process, memory, parts):
        """Replace an interval with refined physical advances."""
        if memory != "reset":
            raise ValueError(
                "The registered device correction resets process memory"
            )
        dt = interval / (parts * self.backup_split)
        self.backup_dt.fill_(dt)
        self.backup_half.fill_(dt / (2 * self.backup_vapor))
        state.vapor_memory.zero_()
        state.pair_memory.zero_()
        exchange = process not in ("coag", "charge-only", "constant")
        for _ in range(parts * self.backup_split):
            if exchange:
                for _ in range(self.backup_vapor):
                    self.phys_vapor_graph.replay()
            if self.has_physical_events:
                self.left_time.copy_(self.backup_dt)
                self.phys_event_graph.replay()
                while bool((self.left_time > 0).any()):
                    self.phys_event_graph.replay()
            if exchange:
                for _ in range(self.backup_vapor):
                    self.phys_vapor_graph.replay()
        # Counts box-wide stage right-hand sides, excluding collision rates.
        return (
            len(self.indices)
            * parts
            * self.backup_split
            * 2
            * self.backup_vapor
            * 3
            * 4
            * int(exchange)
        )


def host_replacement(sim, process, vapor_steps, split_steps):
    """Build the independent eager implementation of the device correction."""
    from stage2_split import Split

    exchange = process not in ("coag", "constant", "charge-only")
    kernel = {
        "both": "off",
        "water": "off",
        "organic": "off",
        "charge": "charged-analytic",
        "charge-only": "charged-analytic",
        "constant": "constant",
    }.get(process, "brownian")
    backup = Split(sim.boxes, sim.seeds, "radau-3-4", kernel, sim.device)

    def replace(main, state, idx, dt, proc, memory, parts, tol):
        if not len(idx):
            return 0
        if memory != "reset":
            raise ValueError(
                "The physical backup resets learned process memory"
            )
        s = state.subset(idx)
        s.vapor_memory.zero_()
        s.pair_memory.zero_()
        micro = dt / (parts * split_steps)
        for _ in range(parts * split_steps):
            if exchange:
                main.vapor(s, micro / 2, vapor_steps, net=backup.transfer)
            if kernel != "off":
                backup.coag(s, micro, idx)
            if exchange:
                main.vapor(s, micro / 2, vapor_steps, net=backup.transfer)
        state.put(idx, s)
        return (
            len(idx)
            * parts
            * split_steps
            * 2
            * vapor_steps
            * 3
            * 4
            * int(exchange)
        )

    return replace
