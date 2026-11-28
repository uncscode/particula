"""Advance native CUDA condensation and coagulation on shared particles."""

import numpy as np
import warp as wp
from dual_native_gpu import DirectDualCondensation
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.coagulation import (
    _enqueue_prepared_coagulation_call,
    _prepare_coagulation_step_gpu,
)
from particula.gpu.kernels.condensation import (
    _enqueue_prepared_condensation_call,
)
from particula.particles import ParticleData


def diagnostics(m, n, z, g, v, rho):
    """Compute native counts, phases, sizes and signed charge concentrations."""
    w = n / v[:, None]
    radius = np.cbrt(3 * (m / rho).sum(-1) / (4 * np.pi))
    bins = np.searchsorted([50e-9, 90e-9, 150e-9], radius, side="right")
    count = w.sum(-1)
    return np.column_stack(
        (
            count,
            (m * w[..., None]).sum(1),
            g,
            (radius * w).sum(1) / np.maximum(count, 1e-30),
            (z * w).sum(1),
            *[(w * (bins == k)).sum(1) for k in range(4)],
            *[(w * z * (bins == k)).sum(1) for k in range(4)],
            (w * abs(z)).sum(-1),
        )
    )


class NativeSplit:
    """Advance native CUDA vapor transfer and Brownian coagulation in place."""

    def __init__(
        self,
        boxes,
        steps,
        vapor_steps,
        *,
        mechanism_config=None,
        turbulent_dissipation=None,
        fluid_density=None,
        collisions=True,
        sampled_collision=None,
    ):
        """Bind both processes to the same device-resident particle arrays."""
        if sampled_collision is not None and collisions:
            raise ValueError("Choose one collision operator")
        self.sampled_collision = sampled_collision
        self.steps = steps
        self.spec = boxes[0].spec
        m = np.stack([b.mass for b in boxes])
        v = np.array([b.volume for b in boxes])
        z = np.stack([b.charge for b in boxes]).astype(float)
        g = np.stack([b.gas for b in boxes])
        b = len(g)
        particle = ParticleData(
            masses=m.copy(),
            concentration=np.ones_like(z),
            density=self.spec.density.copy(),
            charge=z.copy(),
            volume=v,
            distribution_type="particle_resolved",
        )
        gas = GasData(
            list(self.spec.names),
            self.spec.molar[: self.spec.species].copy(),
            g.copy(),
            np.ones(self.spec.species, bool),
        )
        env = EnvironmentData(
            np.full(b, 293.0), np.full(b, 101325.0), g / self.spec.saturation
        )
        dt = (boxes[0].t[1] - boxes[0].t[0]) / steps
        self.owner = DirectDualCondensation(
            Aerosol(particles=particle, gas=gas, environment=env),
            dt / (2 * vapor_steps),
            self.spec,
        )
        self.collision_dt = dt
        self.rng = wp.empty(b, dtype=wp.uint32, device="cuda:0")
        self.coag = None
        if collisions:
            self.coag = _prepare_coagulation_step_gpu(
                self.owner.particles,
                293.0,
                101325.0,
                dt,
                rng_states=self.rng,
                initialize_rng=False,
                mechanism_config=mechanism_config,
                turbulent_dissipation=turbulent_dissipation,
                fluid_density=fluid_density,
                max_collisions=m.shape[1],
            )
        with wp.ScopedCapture(
            device="cuda:0", force_module_load=True
        ) as capture:
            for _ in range(vapor_steps):
                _enqueue_prepared_condensation_call(self.owner.call)
            if sampled_collision is None:
                if self.coag is not None:
                    _enqueue_prepared_coagulation_call(self.coag)
                for _ in range(vapor_steps):
                    _enqueue_prepared_condensation_call(self.owner.call)
        self.graph = capture.graph

    def run(self, boxes, seeds):
        """Restore inputs and collect observations from every native step."""
        owner = self.owner
        if self.sampled_collision is not None:
            seeds = tuple(seeds)
            self.sampled_collision.reset(seeds)
        m = np.stack([b.mass for b in boxes])
        z = np.stack([b.charge for b in boxes]).astype(float)
        v = np.array([b.volume for b in boxes])
        levels = np.stack([b.levels for b in boxes])
        masks = np.stack([b.masks for b in boxes])
        owner.particles.masses.assign(m)
        owner.particles.population.assign(np.ones_like(z))
        owner.particles.charge.assign(z)
        g = levels[:, 0].copy()
        owner.update_gas(g)
        self.rng.assign(
            np.array(
                [
                    np.random.default_rng(int(x)).integers(
                        1, 2**32, dtype=np.uint32
                    )
                    for x in seeds
                ]
            )
        )
        leg = 0
        reports = []
        for i, dt in enumerate(np.diff(boxes[0].t, prepend=boxes[0].t[0])):
            if i and dt == 0:
                leg += 1
                owner.update_gas(
                    np.where(masks[:, leg], levels[:, leg], g).copy()
                )
            elif dt:
                for _ in range(self.steps):
                    wp.capture_launch(self.graph)
                    if self.sampled_collision is not None:
                        self.sampled_collision(
                            owner.particles, self.collision_dt
                        )
                        wp.capture_launch(self.graph)
            m, n, z, g = [
                a.numpy()
                for a in (
                    owner.particles.masses,
                    owner.particles.population,
                    owner.particles.charge,
                    owner.gas.concentration,
                )
            ]
            reports.append(diagnostics(m, n, z, g, v, self.spec.density))
        if self.coag is not None:
            self.coag.check_admission()
        if owner.call.failure.numpy()[0]:
            raise ArithmeticError("Native vapor solver rejected a step")
        return np.stack(reports, 1)
