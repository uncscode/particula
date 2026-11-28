"""Chamber-scale number balance with differentiable, volume-conserving coagulation."""

import numpy as np
from geometry import inputs
from simulate import DYN, RUN_EPS, RHO
from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state,
)
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class Chamber:
    def __init__(self, run, width=3, dt=0.05, device="cpu"):
        self.run = run
        self.device = device
        self.width = width
        self.tensor = lambda a: torch.as_tensor(a, device=device)
        put = self.tensor
        self.d = np.arange(width / 2, 150, width)
        d = self.d
        _, p, base, *_ = inputs(run, "local", d, dt)
        self.diam = put(d)
        self.vol = put(d**3)
        # Native pair kernels in m3/s, converted to cm3/s. Fixed at run-mean T.
        tr = base.mean()
        rgas = p * 0.0280134 / (8.314462618 * tr)
        rad = d * 0.5e-6
        brown = DYN.get_brownian_kernel_via_system_state(rad, 4 * np.pi * RHO * rad**3 / 3, tr, p)
        grav = get_sedimentation_kernel_sp2016_via_system_state(
            rad, np.full_like(rad, RHO), tr, p, False
        )
        shear = DYN.get_turbulent_shear_kernel_st1956_via_system_state(rad, RUN_EPS[run], tr, rgas)
        left, right = np.triu_indices(len(d))
        mass = d[left] ** 3 + d[right] ** 3
        # Suppress only out-of-grid mergers, and report the upper-edge population.
        keep = mass <= d[-1] ** 3
        left = left[keep]
        right = right[keep]
        mass = mass[keep]
        hi = np.searchsorted(d**3, mass)
        lo = hi - 1
        self.ij = put(left)
        self.jj = put(right)
        self.lo = put(lo)
        self.hi = put(hi)
        self.frac = put((mass - d[lo] ** 3) / (d[hi] ** 3 - d[lo] ** 3))
        half = np.where(left == right, 0.5, 1.0) * 1e6
        self.kernel = {
            name: put(mat[left, right] * half)
            for name, mat in [("shear", shear), ("settling", grav), ("brownian", brown)]
        }
        self.base = sum(self.kernel.values())
        # Select pairs by parent size independently of the resulting product band.
        self.kernel["midpairs"] = self.base * put((d[left] >= 20) & (d[right] <= 48))
        self.kernel["all"] = self.base

    def collision(self, conc, k, dt):
        a, b = self.ij, self.jj
        demand = conc[:, a] * conc[:, b] * k * dt
        zeros = torch.zeros_like(conc)
        requested = zeros.index_add(1, a, demand).index_add(1, b, demand)
        hazard = requested / conc.clamp(min=1e-30)
        # Substep depletion caps the usable fraction when large multipliers are tested.
        cap = -torch.expm1(-hazard) / hazard.clamp(min=1e-20)
        cap = torch.where(hazard < 1e-10, 1 - hazard / 2, cap)
        events = demand * torch.minimum(cap[:, a], cap[:, b])
        result = conc.index_add(1, a, -events).index_add(1, b, -events)
        result = result.index_add(1, self.lo, events * (1 - self.frac)).index_add(
            1, self.hi, events * self.frac
        )
        return result, events.sum(1)
