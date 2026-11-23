"""ROS3 stages with coupled material blocks and conservative reservoirs."""

import numpy as np
import warp as wp

from particula.execution import _condensation_block_kernels as block
from particula.execution import _condensation_reservoirs as reservoirs
from particula.execution import _condensation_ros3_kernels as algebra
from particula.execution._gpu_condensation_ros3 import ROS3Stages
from particula.execution.forcing import _EnvironmentSegment


class BlockROS3Stages(ROS3Stages):
    """Store O(particles * exchanging_materials**2) device scratch."""

    def __init__(self, owner, pairs):
        self.pairs = pairs
        self.width = len(pairs)
        self.plus, self.minus, self.span, self.rhs, self.partial = (
            owner._array(np.zeros(owner._size)) for _ in range(5)
        )
        self.increments = tuple(
            owner._array(np.zeros(owner._size)) for _ in range(3)
        )
        particles = owner._shape[0] * owner._shape[1]
        shape = (particles, self.width, self.width)
        self.local, self.gas_column, self.inverse, self.response = (
            owner._array(np.zeros(shape)) for _ in range(4)
        )
        self.augmented = owner._array(
            np.zeros((particles, self.width, 2 * self.width))
        )
        self.particular = owner._array(np.zeros((particles, self.width)))
        self.schur = owner._array(
            np.zeros((owner._shape[0], self.width, self.width + 1))
        )
        self.parts = owner._array(
            np.zeros(
                (
                    owner._shape[0] * self.width,
                    self.width + 1,
                    max(1, min(64, owner._shape[1])),
                )
            )
        )
        self.sources = owner._array([s for s, _ in pairs], wp.int32)
        self.targets = owner._array([t for _, t in pairs], wp.int32)
        reverse_target = np.full(owner._shape[2], -1, dtype=np.int32)
        reverse_source = np.full(owner._gas_shape[1], -1, dtype=np.int32)
        for i, (s, t) in enumerate(pairs):
            reverse_target[t] = reverse_source[s] = i
        self.reverse_target = owner._array(reverse_target, wp.int32)
        self.reverse_source = owner._array(reverse_source, wp.int32)
        self.h = owner._array(np.ones(1))
        self._factorized = False

    def linearize(self, owner):
        """Color independent particles, retaining every cross-material row."""
        self._factorized = False
        owner.statistics.jacobian_calls += 1
        owner._rate(owner.time, owner._state, owner._rates[0])
        for column, (source, target) in enumerate(self.pairs):
            for gas_column, derivative in enumerate(
                (self.local, self.gas_column)
            ):
                owner._launch(
                    algebra.perturb,
                    owner._size,
                    [
                        owner._state,
                        owner._scale,
                        owner._counts,
                        owner._shape[2],
                        owner._gas_shape[1],
                        target,
                        source,
                        gas_column,
                        self.plus,
                        self.minus,
                        self.span,
                    ],
                )
                owner._rate(
                    owner.time, self.plus, owner._rates[2], linearize=True
                )
                owner._rate(
                    owner.time, self.minus, owner._rates[3], linearize=True
                )
                owner._launch(
                    block.extract,
                    self.particular.shape,
                    [
                        owner._rates[2],
                        owner._rates[3],
                        self.span,
                        owner._counts,
                        self.targets,
                        self.sources,
                        owner._shape[2],
                        owner._gas_shape[1],
                        column,
                        gas_column,
                        derivative,
                        owner._invalid,
                    ],
                )
                owner._status()
        self.partial.zero_()
        if owner._forcing is not None:
            delta = np.sqrt(np.finfo(float).eps) * max(abs(owner.time), 1.0)
            if type(owner._forcing) is _EnvironmentSegment:
                delta = min(delta, owner._forcing.end - owner.time)
            if delta > 0:
                owner.statistics.time_derivative_calls += 1
                base = owner._rates[0]
                if owner._flux_guard is not None:
                    base = owner._rates[2]
                    owner._rate(owner.time, owner._state, base, linearize=True)
                owner._rate(
                    owner.time + delta,
                    owner._state,
                    owner._rates[3],
                    linearize=True,
                )
                owner._launch(
                    algebra.partial_time,
                    owner._size,
                    [
                        base,
                        owner._rates[3],
                        wp.float64(delta),
                        self.partial,
                    ],
                )

    def solve(self, owner, h, result):
        """Factor once per trial and solve each stage on device."""
        if not self._factorized:
            if h is not None:
                self.h.assign(np.array([h]))
            owner._launch(
                block.invert_local,
                self.local.shape[0],
                [
                    self.local,
                    self.h,
                    self.augmented,
                    self.inverse,
                    owner._invalid,
                ],
            )
            self._factorized = True
        owner._launch(
            block.eliminate_local,
            self.particular.shape,
            [
                self.inverse,
                self.gas_column,
                self.rhs,
                self.targets,
                owner._shape[2],
                self.response,
                self.particular,
            ],
        )
        owner._launch(
            reservoirs.assemble_parts,
            (
                owner._shape[0],
                self.width,
                (self.width + 1) * self.parts.shape[2],
            ),
            [
                self.local,
                self.response,
                self.particular,
                owner._counts,
                owner._volume,
                owner._scale,
                self.rhs,
                self.targets,
                self.sources,
                owner._shape[2],
                owner._gas_shape[1],
                int(
                    owner._base.particles.distribution_type
                    == "particle_resolved"
                ),
                self.h,
                self.parts,
            ],
        )
        owner._launch(
            reservoirs.finish_schur,
            self.schur.shape,
            [
                self.parts,
                self.rhs,
                self.sources,
                owner._mass_size,
                owner._gas_shape[1],
                self.h,
                self.schur,
            ],
        )
        owner._launch(
            block.solve_reservoirs,
            owner._shape[0],
            [
                self.schur,
                self.sources,
                owner._mass_size,
                owner._gas_shape[1],
                result,
                owner._invalid,
            ],
        )
        owner._launch(
            block.back_substitute,
            owner._size,
            [
                self.response,
                self.particular,
                self.rhs,
                self.targets,
                self.sources,
                self.reverse_target,
                self.reverse_source,
                owner._shape[1],
                owner._shape[2],
                owner._gas_shape[1],
                self.h,
                result,
                owner._invalid,
            ],
        )
        owner.statistics.linear_solves += 1
        owner._status()
