# Measured cost and accuracy of prepared simulations

The timing and reference checks below come from the same RTX 5060 campaign. Each result retains its workload, synchronization boundary and machine context.

## Machine and executable provenance

- NVIDIA GeForce RTX 5060, 8,151 MiB reported device memory, architecture `sm_120`.
- NVIDIA driver 615.78.08; Warp 1.11.1 built with CUDA toolkit 12.9, reporting
  CUDA driver compatibility 13.4.
- Python 3.12.12; AMD Ryzen AI 9 HX 370 CPU.
- CUDA target `cuda:0`, float64 scientific state. Warp CPU is separately
  parameterized regression evidence and is never reported as GPU performance.
- Campaign executable revision `464738b14`.
  The source digest covers library and study Python files; test and documentation
  edits are outside that digest. Each result retains its own revision, digest,
  input-state hash, hardware/runtime information and dirty-tree indicator.

The [downloadable scaling report](gpu_scaling_results.py)
contains only small point/line/bar series and displayed provenance text (about
34 kB). Full timing repetitions, validation diagnostics, raw histories, profiler
traces and subprocess logs remain local artifacts. Particle states and image
arrays are not included in downloadable data; rerun the scripts to regenerate
them.

## Measured cost of residency

These medians cover three timesteps on this machine. The NumPy and prepared
CUDA cases share physical inputs and pass the separate full-array parity check.
Fresh time includes preparation, work and download; resident time starts after
preparation. Compilation and context startup are warmed in both measurements.

| Workload | NumPy resident | Prepared CUDA resident | NumPy fresh | Prepared CUDA fresh |
| --- | ---: | ---: | ---: | ---: |
| One box, 1,048,576 particles, condensation + dilution | 2.197 s | 0.0672 s | 2.200 s | 0.777 s |
| 1,024 boxes × 1,024 particles, condensation + dilution | 6.982 s | 0.0698 s | 6.986 s | 0.160 s |
| One box, 1,048,576 particles, dilution | 0.0643 s | 0.000529 s | 0.0674 s | 0.686 s |

The one-box condensation case is about 33 times faster for resident work and
2.8 times faster for a fresh run. Pure dilution illustrates the opposite
end-to-end tradeoff: its very short device computation does not repay preparation
and transfer costs in this three-step run. Keeping state resident matters.
These ratios describe the current implementations and workloads, not universal
GPU speedups. The downloadable report keeps the plotted median and minimum–maximum spread.

For one Brownian step at 16,384 particles per box, the retained previous CUDA
implementation measured 161.556 s and prepared CUDA 0.1378 s (about 1,172 times
faster). All four CUDA modes produced the same final-state hash. This compares
two GPU implementations, not GPU against CPU, and retains quadratic pair work.
The complete old-code case took 1,620.8 s, including three warmups and seven
samples, within its forty-minute allowance.

Timing workloads use fixed coarse steps to stress execution and inventory
accounting. They establish cost and parity, not timestep-converged physical
predictions. The refined histories below provide separate scientific evidence.

## Scientific acceptance

- Particle scaling reaches 1,048,576 active particles in one box for
  condensation/dilution. Box scaling reaches 1,024 boxes of 1,024 slots.
- Brownian comparisons cover 128 through 16,384 particles per box. Pair work
  remains quadratic; bounded scratch is not a linear-complexity claim.
- Four-material/five-gas cases cover 25%, 75% and 100% active occupancy.
- Heterogeneous boxes vary temperature, pressure, volume, composition and gas.
  The forcing study records actual external vapor exchange before checking
  per-box/per-material inventories. Both histories require timestep refinement:
  6,000 steps of 1 microsecond versus 12,000 steps of 0.5 microsecond,
  preserving duration and forcing times; each particle's final total mass
  changes by less than 1% and each box's final water vapor by less than 0.1%.
  The separate illustrated runs measured maximum particle-mass changes of
  0.126% (heterogeneous boxes) and 0.084% (forcing), with per-box/material
  inventory relative errors below 4.1e-14.
- Sixteen stream seeds share one initial aerosol. Ensemble spread is not a
  confidence interval; each realization must conserve mass and charge and
  exhibit intended nonzero coagulation.
- The 1,572,864-active-particle lifecycle checks all five process families,
  independent process source/sink ledgers, detached snapshots, and exact
  captured/ordinary/checkpoint-resumed continuation.
- The longer cyclic chamber tutorial applies repeated unit-count source pulses
  and temperature/vapor schedules, then repeats its second half from a checkpoint.
  It omits weight-changing sinks to retain equal-count Brownian compatibility.
  It is a deliberately refreshed illustration, not a timestep-converged chamber
  prediction.

## Profiling findings and limits

Nsight Systems identified the previous million-slot condensation bottleneck:
serial finalized-transfer reduction accounted for about 79.6% of total traced
kernel time and candidate reduction for another 14.5%. Their combined cost was
about 10.91 seconds for one four-substep call. The implementation now uses
fixed-size chunk reductions for large arrays, retaining bounded rounded-transfer
correction and inventory checks. Reduction grouping changes floating-point
roundoff, so numerical and conservation tests accompany the optimization.

In the optimized trace, preparation's population validation consumed about
0.679 seconds and 96.8% of traced kernel time; stepping kernels together took
roughly 22 milliseconds. This identifies remaining setup cost. Trace timing is
not substituted for the independent repeated benchmark measurements.

Nsight Compute was attempted, but the driver rejected access to performance
counters with `ERR_NVGPUCTRPERM`. System security settings were not changed.
Hardware-counter occupancy/bandwidth evidence is therefore unavailable; Nsight
Systems timing and kernel attribution remain available. Runtime and timing
methodology follow the version-matched
[Warp runtime](https://nvidia.github.io/warp/v1.11/user_guide/runtime.html) and
[profiling documentation](https://nvidia.github.io/warp/v1.11/deep_dive/profiling.html).
