"""Shared FP64 device coefficients for the ROS3 embedded pair."""

import warp as wp

# Typed constants avoid rounding Python literals through float32 in Warp.
# KPP Ros3, int/rosenbrock.f90 blob baf1ef64538a5128db5e1154ac7fd1b39c41f665.
GAMMA = wp.constant(wp.float64(0.43586652150845899941601945119356))
GAMMA2 = wp.constant(wp.float64(0.24291996454816804366592249683314))
GAMMA3 = wp.constant(wp.float64(2.1851380027664058511513169485832))
C21 = wp.constant(wp.float64(1.0156171083877702092))
C31 = wp.constant(wp.float64(4.0759956452537699825))
C32 = wp.constant(wp.float64(9.2076794298330791242))
M2 = wp.constant(wp.float64(6.1697947043828245593))
M3 = wp.constant(wp.float64(0.4277225654321857333))
E2 = wp.constant(wp.float64(2.9079558716805469822))
E3 = wp.constant(wp.float64(0.2235406989781156963))
