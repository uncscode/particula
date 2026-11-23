"""Resident ROS3 stages for the single-exchange condensation profile.

The accepted state and lifecycle belong to GPUCondensationODE. This helper
owns only trial scratch and never downloads a physical state. Derivatives use
native device rates; a conservative arrowhead solve replaces dense LU storage.
"""

import numpy as np
import warp as wp

from particula.execution import _condensation_ros3_kernels as algebra
from particula.execution.forcing import _EnvironmentSegment


class ROS3Stages:
    """Allocate linear-storage scratch on the owner's requested device."""

    def __init__(self, owner, pair):
        self.source, self.target = pair
        self.plus, self.minus, self.span, self.rhs, self.partial = (
            owner._array(np.zeros(owner._size)) for _ in range(5)
        )
        self.increments = tuple(
            owner._array(np.zeros(owner._size)) for _ in range(3)
        )
        self.local, self.gas_column = (
            owner._array(np.zeros(owner._shape[:2])) for _ in range(2)
        )

    def linearize(self, owner):
        """Differentiate local particle/gas columns at the accepted state."""
        owner.statistics.jacobian_calls += 1
        owner._rate(owner.time, owner._state, owner._rates[0])
        for gas_columns, derivative in enumerate((self.local, self.gas_column)):
            owner._launch(
                algebra.perturb,
                owner._size,
                [
                    owner._state,
                    owner._scale,
                    owner._counts,
                    owner._shape[2],
                    owner._gas_shape[1],
                    self.target,
                    self.source,
                    gas_columns,
                    self.plus,
                    self.minus,
                    self.span,
                ],
            )
            owner._rate(owner.time, self.plus, owner._rates[2], linearize=True)
            owner._rate(owner.time, self.minus, owner._rates[3], linearize=True)
            owner._launch(
                algebra.extract_derivative,
                owner._shape[:2],
                [
                    owner._rates[2],
                    owner._rates[3],
                    self.span,
                    owner._counts,
                    owner._shape[2],
                    owner._gas_shape[1],
                    self.target,
                    self.source,
                    gas_columns,
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
        """Eliminate gas coupling and back-substitute particle increments."""
        inverse = 1.0 / (h * 0.43586652150845899941601945119356)
        owner._invalid.zero_()
        owner._launch(
            algebra.solve_gas,
            owner._shape[0],
            [
                self.local,
                self.gas_column,
                owner._counts,
                owner._volume,
                owner._scale,
                self.rhs,
                owner._shape[2],
                owner._gas_shape[1],
                self.target,
                self.source,
                int(
                    owner._base.particles.distribution_type
                    == "particle_resolved"
                ),
                wp.float64(inverse),
                result,
                owner._invalid,
            ],
        )
        owner._launch(
            algebra.solve_remaining,
            owner._size,
            [
                self.local,
                self.gas_column,
                self.rhs,
                owner._shape[2],
                owner._gas_shape[1],
                self.target,
                self.source,
                wp.float64(inverse),
                result,
                owner._invalid,
            ],
        )
        owner.statistics.linear_solves += 1
        owner._status()

    def trial(self, owner, h):
        """Check the full candidate and return the embedded maximum error."""
        self.linearize(owner)
        first, second, third = self.increments
        for stage, rate, result in (
            (1, owner._rates[0], first),
            (2, owner._rates[1], second),
            (3, owner._rates[1], third),
        ):
            owner._launch(
                algebra.stage_rhs,
                owner._size,
                [
                    rate,
                    first,
                    second,
                    self.partial,
                    wp.float64(h),
                    stage,
                    self.rhs,
                ],
            )
            self.solve(owner, h, result)
            if stage == 1:
                owner._launch(
                    algebra.stage_state,
                    owner._size,
                    [owner._state, first, owner._stage],
                )
                owner._rate(
                    owner.time + 0.43586652150845899941601945119356 * h,
                    owner._stage,
                    owner._rates[1],
                )
        owner._norm.zero_()
        owner._launch(
            algebra.finish,
            owner._size,
            [
                owner._state,
                first,
                second,
                third,
                owner._atol,
                owner._rtol,
                owner._candidate,
                owner._norm,
            ],
        )
        if owner._provider is None:
            owner._unpack_trial(owner._candidate)
        else:
            owner._rate(owner.time + h, owner._candidate, owner._rates[1])
        owner.control_readbacks += 1
        return float(owner._norm.numpy()[0])
