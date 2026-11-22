"""Scalar and vector types for the frozen twenty legacy modifiers."""

import warp as wp

from . import kernels as k

F = k.F
Vec = wp.types.vector(length=20, dtype=F)
