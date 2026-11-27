"""Conservative shared neural transfer across variable numbers of vapors."""

import numpy as np
import torch
from dual_materials import SURFACE_DENSITY, SURFACE_TENSION, TEMPERATURE, R
from dual_stages import DualStages
from dual_transfer import DualExchange, solve2
from recursive_stages import tableau
from torch import nn
from water_constants import MEAN_FREE_PATH


class SpeciesHead(nn.Module):
    """Share each expert over volatile species and computational particles."""

    def __init__(self, kind, width):
        """Allocate weights for the selected experts."""
        super().__init__()
        self.kind = kind
        self.width = width
        self.cell = (
            nn.GRUCell(14, width)
            if kind in ("gru", "mixture", "uniform")
            else None
        )
        self.ff = (
            nn.Sequential(nn.Linear(14, width), nn.SiLU(), nn.Linear(width, 1))
            if kind in ("ff", "mixture", "uniform")
            else None
        )
        if self.cell:
            self.recurrent = nn.Linear(width, 1)
        if kind in ("attention", "mixture", "uniform"):
            self.key = nn.Linear(14, width, bias=False)
            self.query = nn.Linear(14, width, bias=False)
            self.value = nn.Linear(14, width)
            self.spatial = nn.Sequential(
                nn.Linear(14 + width, width), nn.SiLU(), nn.Linear(width, 1)
            )
        if kind == "mixture":
            self.gate = nn.Linear(14, 4)
        for head in (
            self.ff[-1] if self.ff else None,
            getattr(self, "recurrent", None),
            self.spatial[-1] if hasattr(self, "spatial") else None,
        ):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.double()

    def forward(self, x, h, update):
        """Evaluate one causal, species-permutation-equivariant correction."""
        if self.ff:
            ff = self.ff(x).squeeze(-1).tanh()
            if self.kind == "ff":
                return ff, h
        if self.cell:
            next_state = self.cell(
                x.reshape(-1, 14), h.reshape(-1, self.width)
            ).reshape_as(h)
            h = torch.where(update[..., None, None], next_state, h)
            recurrent = self.recurrent(h).squeeze(-1).tanh()
            if self.kind == "gru":
                return recurrent, h
        score = (
            self.query(x) @ self.key(x).transpose(-1, -2) / (self.width**0.5)
        )
        pool = score.softmax(-1) @ self.value(x).tanh()
        context = self.spatial(torch.cat((x, pool), -1)).squeeze(-1).tanh()
        if self.kind == "attention":
            return context, h
        experts = torch.stack(
            (torch.zeros_like(ff), ff, recurrent, context), -1
        )
        if self.kind == "uniform":
            return experts.mean(-1), h
        return (self.gate(x).softmax(-1) * experts).sum(-1), h


class MultiExchange(DualExchange):
    """Use local species blocks and a gas solve of rank equal to vapor count."""

    def __init__(self, spec, kind="physical", width=16, order=0, iterations=2):
        """Keep material properties separate from shared learned weights."""
        nn.Module.__init__(self)
        self.kind, self.width, self.order, self.iterations = (
            kind,
            width,
            order,
            iterations,
        )
        if order:
            self.gamma, self.rows = tableau(order)
        s = spec.species
        water_molar = 0.01801528
        for name, value in (
            ("molar", spec.molar[:s]),
            ("dry_molar", spec.molar[s:]),
            ("density", spec.density[:s]),
            ("dry_density", spec.density[s:]),
            ("alpha", spec.accommodation),
            ("sat", spec.saturation),
            ("diffusion", spec.diffusivity),
            ("path", MEAN_FREE_PATH * np.sqrt(water_molar / spec.molar[:s])),
            (
                "kelvin",
                2
                * SURFACE_TENSION
                * spec.molar[:s]
                / (SURFACE_DENSITY * R * TEMPERATURE),
            ),
            ("identity", np.eye(s)),
        ):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        self.head = SpeciesHead(kind, width) if kind != "physical" else None

    def static(self, mass, number):
        """Represent two dry materials and all active volatile lanes."""
        fixed = torch.where((number > 0)[..., None], mass[..., -2:], 1e-18)
        dry = fixed.sum(-1)
        return (
            dry,
            (fixed / self.dry_molar).sum(-1) / dry,
            (fixed / self.dry_density).sum(-1),
            dry * number,
            fixed[..., 1] / dry,
            number > 0,
        )

    def start(self, number):
        """Return one memory vector per particle and volatile species."""
        return number.new_zeros(
            (
                *number.shape,
                len(self.molar),
                self.width if self.head is not None and self.head.cell else 0,
            )
        )

    def solve_increment(self, weight, jac, gain, h, rhs):
        """Apply the shared-reservoir inverse with one small solve per box."""

        def solve(a, b):
            if len(self.molar) == 2:
                return solve2(a, b)
            value, info = torch.linalg.solve_ex(a, b, check_errors=False)
            torch._assert_async((info == 0).all(), "Singular multiphase system")
            return value

        response = solve(
            self.identity - h[..., None] * jac,
            torch.cat((rhs[..., None], torch.diag_embed(h * gain)), -1),
        )
        free, shared = response[..., 0], response[..., 1:]
        change = solve(
            self.identity + (weight[..., None, None] * shared).sum(1),
            (weight[..., None] * free).sum(1)[..., None],
        )
        return free - (shared @ change[:, None]).squeeze(-1)

    def adjustment(self, q, g, state, dt, memory, delta, stiff, drive, radius):
        """Bound each learned species correction to the physical update."""
        if self.head is None:
            return delta, memory
        _, _, _, weight, f, active = state
        scalars = [
            (radius * 1e6).log(),
            f,
            torch.log1p(weight.sum(1, keepdim=True) * 1e4).expand_as(f),
            torch.log1p(q.sum(-1)),
        ]
        props = [
            (self.molar / 0.1).log(),
            (self.sat / 0.001).log(),
            self.diffusion / 1e-5,
            self.alpha,
        ]
        tokens = torch.stack(
            [
                *(v[..., None].expand_as(q) for v in scalars),
                torch.log1p(q),
                (g / self.sat)[:, None].expand_as(q),
                drive / self.sat,
                torch.log1p(stiff),
                torch.asinh(delta),
                *(p.expand_as(q) for p in props),
                torch.log1p(dt / 0.0025)[:, None, None].expand_as(q),
            ],
            -1,
        )
        adj, memory = self.head(tokens, memory, active & (dt[:, None] > 0))
        return delta * (1 + 0.75 * adj * stiff / (1 + stiff)), memory

    def step(self, q, gas, state, dt, memory):
        """Use one implicit increment or recursive nonlinear stages."""
        if self.order:
            return DualStages.step(self, q, gas, state, dt, memory)
        return super().step(q, gas, state, dt, memory)


class ExtrapolatedExchange(MultiExchange):
    """Correct the two-half-step estimate with a learned local error term."""

    def step(self, q, gas, state, dt, memory):
        """Bound the Richardson correction and conserve each material."""
        weight = state[3]
        full, stiff, drive, radius = self.increment(q, gas, state, dt)
        coarse, _ = self.balance(q, gas, weight, full)
        half_dt = dt * 0.5
        first = self.increment(q, gas, state, half_dt)[0]
        midpoint, midgas = self.balance(q, gas, weight, first)
        second = self.increment(midpoint, midgas, state, half_dt)[0]
        fine, _ = self.balance(midpoint, midgas, weight, second)
        error, memory = self.adjustment(
            q, gas, state, dt, memory, fine - coarse, stiff, drive, radius
        )
        result, vapor = self.balance(q, gas, weight, fine - q + error)
        return result, vapor, memory


class RosenbrockExchange(MultiExchange):
    """Two ROS-2 stages share their local and reservoir factorizations."""

    gamma = 1 + 1 / np.sqrt(2)

    def factorized_stages(self, jac, gain, weight, h, rate):
        """Factor each local block and its shared reservoir correction once."""
        gamma = self.gamma
        local, pivots, status = torch.linalg.lu_factor_ex(
            self.identity - gamma * h[..., None] * jac
        )
        torch._assert_async((status == 0).all(), "Singular local stage")
        rhs = torch.cat(
            ((h * rate)[..., None], torch.diag_embed(gamma * h * gain)), -1
        )
        response = torch.linalg.lu_solve(local, pivots, rhs)
        shared = response[..., 1:]
        reservoir, rpivots, status = torch.linalg.lu_factor_ex(
            self.identity + (weight[..., None, None] * shared).sum(1)
        )
        torch._assert_async((status == 0).all(), "Singular shared stage")

        def close_reservoir(free):
            rhs_gas = (weight[..., None] * free).sum(1)[..., None]
            change = torch.linalg.lu_solve(reservoir, rpivots, rhs_gas)
            return free - (shared @ change[:, None]).squeeze(-1)

        def solve(rhs):
            free = torch.linalg.lu_solve(local, pivots, rhs[..., None])
            return close_reservoir(free.squeeze(-1))

        return close_reservoir(response[..., 0]), solve

    def increments(self, q, gas, state, h, first, solve):
        """Evaluate the physical update and the embedded error estimate."""
        weight = state[3]
        stage_q, stage_gas = self.balance(q, gas, weight, first)
        stage_rate = self.linearization(
            stage_q, stage_gas, state, rate_only=True
        )
        second = solve(h * stage_rate - 2 * first)
        error = 0.5 * (first + second)
        return first, error

    def step(self, q, gas, state, dt, memory):
        """Bound the learned adjustment to the physical error estimate."""
        h = dt[:, None, None]
        weight = state[3]
        rate, jac, gain, drive, radius = self.linearization(q, gas, state)
        first, solve = self.factorized_stages(jac, gain, weight, h, rate)
        lower, error = self.increments(q, gas, state, h, first, solve)
        stiff = h * (
            jac.abs().sum(-1) + (weight[..., None] * gain).sum(1, keepdim=True)
        )
        embedded, memory = self.adjustment(
            q,
            gas,
            state,
            dt,
            memory,
            error,
            stiff,
            drive,
            radius,
        )
        y, vapor = self.balance(q, gas, weight, lower + embedded)
        return y, vapor, memory


class Rosenbrock3Exchange(RosenbrockExchange):
    """Apply the ROS-3 coefficients on each fixed-forcing interval."""

    # ROS-3 and ROS-4 use KPP's double-precision Rosenbrock coefficients.
    # https://github.com/KineticPreProcessor/KPP/blob/main/int/rosenbrock.f90
    gamma = 0.4358665215084590
    positions = ((1.0,), None)
    coupling = ((-1.0156171083877702,), (4.07599564525377, 9.207679429833079))
    principal = (1.0, 6.169794704382825, -0.42772256543218573)
    errors = (0.5, -2.907955871680547, 0.2235406989781157)

    def increments(self, q, gas, state, h, first, solve):
        """Reuse the previous rate whenever two stage positions coincide."""
        ks = [self.gamma * first]
        stage_rate = None
        for a, c in zip(self.positions, self.coupling, strict=True):
            if a is not None:
                shift = sum(w * k for w, k in zip(a, ks, strict=True))
                y, g = self.balance(q, gas, state[3], shift)
                stage_rate = self.linearization(y, g, state, rate_only=True)
            source = h * stage_rate + sum(
                w * k for w, k in zip(c, ks, strict=True)
            )
            ks.append(solve(self.gamma * source))
        main = sum(w * k for w, k in zip(self.principal, ks, strict=True))
        error = sum(w * k for w, k in zip(self.errors, ks, strict=True))
        return main - error, error


class Rosenbrock4Exchange(Rosenbrock3Exchange):
    """Apply the ROS-4 coefficients using three rate evaluations."""

    gamma = 0.57282
    positions = ((2.0,), (1.867943637803922, 0.2344449711399156), None)
    coupling = (
        (-7.13761503641231,),
        (2.580708087951457, 0.6515950076447975),
        (-2.137148994382534, -0.3214669691237626, -0.6949742501781779),
    )
    principal = (
        2.255570073418735,
        0.2870493262186792,
        0.435317943184018,
        1.093502252409163,
    )
    errors = (
        -0.2815431932141155,
        -0.0727619912493892,
        -0.1082196201495311,
        -1.093502252409163,
    )


def transfer_class(*, extrapolate=False, rosenbrock=False, ros_order=0):
    """Restore the numerical method used when fitting the shared head."""
    if sum(bool(x) for x in (extrapolate, rosenbrock, ros_order)) > 1:
        raise ValueError("Choose one base integration rule")
    if ros_order:
        return {3: Rosenbrock3Exchange, 4: Rosenbrock4Exchange}[ros_order]
    if rosenbrock:
        return RosenbrockExchange
    if extrapolate:
        return ExtrapolatedExchange
    return MultiExchange


def from_checkpoint(spec, cp):
    """Install the species-shared weights on the specified material system."""
    cls = transfer_class(
        extrapolate=cp.get("extrapolate", False),
        rosenbrock=cp.get("rosenbrock", False),
        ros_order=cp.get("ros_order", 0),
    )
    net = cls(spec, **cp["config"])
    net.head.load_state_dict(cp["head"])
    return net
