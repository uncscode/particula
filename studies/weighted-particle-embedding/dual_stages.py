"""Apply shared recursive Newton updates to simultaneous vapor transfer."""

import torch
from dual_transfer import DualExchange
from recursive_stages import tableau


class DualStages(DualExchange):
    """Implicit stages share one physical or learned two-reservoir update."""

    def __init__(self, order=2, iterations=2, kind="physical", width=12):
        """Configure the implicit tableau, Newton depth and shared head."""
        super().__init__(kind=kind, width=width)
        self.gamma, self.rows = tableau(order)
        if iterations < 1:
            raise ValueError("At least one stage update is required")
        self.iterations = iterations

    def step(self, q, gas, state, dt, memory):
        """Solve each stage using both conserved reservoirs."""
        h = dt[:, None, None]
        weight = state[3]
        sub = h * self.gamma
        previous = []
        for row in self.rows:
            base = q
            for a, slope in zip(row, previous, strict=True):
                base = base + h * a * slope
            y, vapor = q, gas
            for _ in range(self.iterations):
                rate, jac, gain, drive, radius = self.linearization(
                    y, vapor, state
                )
                residual = base - y + sub * rate
                delta = self.solve_increment(weight, jac, gain, sub, residual)
                stiff = sub * (
                    jac.abs().sum(-1)
                    + (weight[..., None] * gain).sum(1, keepdim=True)
                )
                delta, memory = self.adjustment(
                    y, vapor, state, dt, memory, delta, stiff, drive, radius
                )
                y, vapor = self.balance(y, vapor, weight, delta)
            # Stage slopes use the achieved state. This also carries any
            # positivity-limiter effect into the following stage.
            previous.append(
                torch.where(sub > 0, (y - base) / sub.clamp_min(1e-30), 0)
            )
        return y, vapor, memory


def build(config):
    """Restore both early one-step heads and recursive implicit-stage heads."""
    return DualStages(**config) if "order" in config else DualExchange(**config)
