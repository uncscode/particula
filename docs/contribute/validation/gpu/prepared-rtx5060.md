# Prepared simulations: RTX 5060 qualification

This record belongs to the prepared GPU simulation feature branch. It separates
API correctness, scientific validation, performance measurements and publication
checks. Implementation is development work until its PR is merged; this record
does not authorize a release.

## Machine and executable provenance

- NVIDIA GeForce RTX 5060, 8,151 MiB reported device memory, architecture `sm_120`.
- NVIDIA driver 615.78.08; Warp 1.11.1 built with CUDA toolkit 12.9, reporting
  CUDA driver compatibility 13.4.
- Python 3.12.12; AMD Ryzen AI 9 HX 370 CPU.
- CUDA target `cuda:0`, float64 scientific state. Warp CPU is separately
  parameterized regression evidence and is never reported as GPU performance.
- Campaign executable revision `0a8307046`.
  The source digest covers library and study Python files; test and documentation
  edits are outside that digest. Each result retains its own revision, digest,
  input-state hash, hardware/runtime information and dirty-tree indicator.

The [downloadable scaling report](../../../examples/gpu/prepared/gpu_scaling_results.py)
contains only small point/line/bar series and displayed provenance text (about
34 kB). Full timing repetitions, validation diagnostics, raw histories, profiler
traces and subprocess logs remain local artifacts. Particle states and image
arrays are not included in downloadable data; rerun the scripts to regenerate
them.

## Reproduce the campaign

Run on the actual CUDA host, with no competing GPU experiments:

```bash
python -u -m scripts.gpu_studies.campaign \
  --output .artifacts/gpu-campaign --hours 8 --legacy-seconds 2400
python -u -m scripts.validate_gpu_campaign_parity \
  .artifacts/gpu-campaign/cpu-parity.json
python -m scripts.publish_gpu_campaign \
  .artifacts/gpu-campaign docs/data/gpu/rtx5060.json
```

The fixed matrix has 104 jobs: 100 case/mode timing runs and four scientific
histories. Each case runs in a separate process. Ordinary cases have a ten-minute
limit; the specifically approved previous Brownian baseline has forty minutes,
within the same eight-hour campaign budget. Resume preserves the original
deadline and rejects stale executable or mismatched case results. Incomplete
required cases remain incomplete instead of being replaced with smaller jobs.

Timing runs use three warmups and seven repetitions restored to identical
scientific and random state. Restoration and analysis are outside resident
timing; work and completion/status synchronization are inside it. GPU modes
also have seven fresh preparation/work/download runs, except the previous
Brownian baseline, which measures resident work only. JIT/context startup is
warmed. Graph recording and its first replay have distinct costs.

"Direct" refers to retained low-level kernel calls without repeated public
wrapper preflight. The old Brownian implementation is a CUDA comparison, not a
CPU reference. Native NumPy comparisons are limited to matching condensation
and dilution semantics. Condensation workloads include exponential dilution and
four equal coupled condensation substeps. Full-array CPU/CUDA parity is checked
separately after timing. Direct/prepared/captured/legacy GPU results must match
exactly within each common case before publication.

Campaign result hashes cover native aerosol arrays, ordered gas names and the
population tag. They do not hash raw RNG words. Random-state equivalence and
canonical checkpoint continuation are checked separately by the Brownian and
prepared restart/capture regression suites.

Memory admission is the smaller of 6 GiB and 75% of available device memory,
with a conservative primary/scratch/restore estimate. Observed memory is the
Warp memory-pool high-water mark and excludes CUDA context and driver storage.
Both admission and observed-allocation gates must pass.

## Measured cost of residency

These medians cover three timesteps on this machine. The NumPy and prepared
CUDA cases share physical inputs and pass the separate full-array parity check.
Fresh time includes preparation, work and download; resident time starts after
preparation. Compilation and context startup are warmed in both measurements.

| Workload | NumPy resident | Prepared CUDA resident | NumPy fresh | Prepared CUDA fresh |
| --- | ---: | ---: | ---: | ---: |
| One box, 1,048,576 particles, condensation + dilution | 2.364 s | 0.0672 s | 2.367 s | 0.778 s |
| 1,024 boxes × 1,024 particles, condensation + dilution | 7.287 s | 0.0701 s | 7.291 s | 0.160 s |
| One box, 1,048,576 particles, dilution | 0.0682 s | 0.000553 s | 0.0722 s | 0.692 s |

The one-box condensation case is about 35 times faster for resident work and
three times faster for a fresh run. Pure dilution illustrates the opposite
end-to-end tradeoff: its very short device computation does not repay preparation
and transfer costs in this three-step run. Keeping state resident matters.
These ratios describe the current implementations and workloads, not universal
GPU speedups. The downloadable report keeps the plotted median and minimum–maximum spread.

For one Brownian step at 16,384 particles per box, the retained previous CUDA
implementation measured 161.648 s and prepared CUDA 0.1378 s (about 1,173 times
faster). All four CUDA modes produced the same final-state hash. This compares
two GPU implementations, not GPU against CPU, and retains quadratic pair work.
The complete old-code case took 1,619.7 s, including three warmups and seven
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

## Final validation ledger

- Library: `pytest -q -Werror -m "not slow and not performance"
  --cov=particula --cov-report=term-missing --cov-fail-under=80`: 9,936 passed,
  13 skipped, 57 deselected, one expected failure; **91.91% coverage**.
- Examples and infrastructure: `pytest examples_tests scripts/tests -q -Werror
  -m "not slow and not performance"`: **450 passed, one skipped**.
- Strict RTX gate: `python scripts/qualify_gpu.py --output
  .artifacts/qualification/final/gpu`: **453 passed, zero skips**.
- Full-array CPU/CUDA campaign parity: **17 matching cases passed**,
  `rtol=2e-12`, `atol=1e-30`, against executable digest
  `41e30ec6c05cbc75fe05639f84149373fd173a4a15c7ad55412a456b5ecf20e5`.
- Final campaign: **104 of 104 jobs passed**, including 100 timing runs and
  four scientific histories, within the original eight-hour window.
- The quickstart refinement changed final particle mass by at most 0.0545%,
  below its 1% acceptance threshold.
- Ruff and formatting passed; mypy passed on 558 source files; documentation
  structure reported zero errors; all 11 Particula Helper packages validated.

- All 54 CPU-policy documentation examples executed successfully; all saved
  examples passed fingerprint and file-integrity checks. Fourteen CUDA tutorials
  were explicitly refreshed on the RTX GPU. Their execution remains manual.
- Website unit checks and build passed: 432 authored/API pages plus 130
  redirects, with zero link/anchor errors. Existing relocated support anchors
  remain valid. Quickstart and report pages were visually reviewed at desktop
  (1440 pixels) and mobile (390 pixels) widths with loaded figures and no
  page-level horizontal overflow. Scientific plots were also inspected directly.
- Wheel/sdist checks passed for version 0.3.0: 194 documentation files, isolated
  installation checks and an offline sdist rebuild. Build artifacts are local
  validation evidence, not a package release.
- The downloaded scaling ZIP ran outside the checkout using its bundled JSON.
  The multibox ZIP also passed its complete 196,608-active-particle coarse/fine
  CUDA study outside the checkout, with imports verified to resolve to the
  extracted helper files. An initial externally terminated run was not accepted
  as evidence; the independently completed retry exited successfully.

Publication follow-up: the data exporter now selects only the 78 plotted timing
series, their displayed cost/memory summaries and the one small distribution
line. All 104 jobs still pass the same pre-publication gates; raw repetitions
and detailed scientific records are validated locally and omitted from downloads.
