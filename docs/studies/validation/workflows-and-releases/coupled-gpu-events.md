# Coupled GPU event boundaries

<a id="coupled-gpu-event-boundaries"></a>

This record extends the [recording and restart qualification](recording-and-restart.md) to coupled events on the GPU.

The follow-through implementation adds `GPUCoupledSimulation`: fixed native
source/collision/removal/dilution events and a separately controlled GPU
condensation flow. It preserves persistent event buffers/RNG, explicit
sequential order and full host transfer boundaries. The controller resets
at every event boundary. Its distinct schema-1 checkpoint contains canonical
event continuation, condensation identity, scalar controls, order and work
counters; the outer archive codec remains explicit data, without pickle.

The focused coupled/numerics/archive/restart selection passed **65 tests with
one deliberate Warp-CPU capture skip** on Warp CPU and the RTX 5060. It includes
exact disk continuation with source and Brownian collisions, captured events,
rejected condensation trials without repeated event draws, unchanged caller
state and event-buffer identity, independent dilution/inventory checks,
empty-population continuation, shorter-bound rejection and faulted-cycle
observation rejection. The latter retains the last complete public clock while
rejecting further snapshots and checkpoints after partial physical work.
Changed condensation physics rejects before device access. Unsupported event
families reject before allocation. All-package Mypy and Ruff checks passed.

These tests do not substitute for the full published workloads. The cyclic
chamber still requires all 32 boxes, 8,192 slots, 600 one-second cycles, 128-event
pulses and complete second-half restart. The forcing example retains 16 boxes,
4,096 slots, 6,000 one-microsecond intervals and the full doubled-step refinement
with unchanged control times. Their updated sources and compact recording are
implemented. Both full publications were launched in a detached checkout pinned
to `480ecf2c1a`, separately from subsequent edits. The forcing publication and
figure inspection subsequently completed, as recorded below. The cyclic chamber
still requires full-workload completion and figure review. No GPU speedup is
inferred from this explicit transfer path.

The broader integration gate completed with **10,809 package tests passed**, 21
skips and one expected failure in 436.94 s; unfiltered coverage was **91.33%**.
This includes actual CUDA. The additional intentional skip is the coupled
native-capture case under the Warp-CPU fixture. Example/tooling checks passed
**554 tests with one skip** in 141.64 s. The 563-page website build and rendered
structure check passed with zero errors. A subsequent explicit checkpoint-plan
type guard passed checked-body Mypy and all **17 coupled tests**, with the same
single Warp-CPU capture skip. The full suite began before that small guard was
added; its coverage is complemented by this focused rerun, not represented as
an untouched-source final-head run. Logs are retained in
`.artifacts/gpu-coupled-package-coverage.log`,
`.artifacts/gpu-coupled-tooling-examples.log`,
`.artifacts/gpu-coupled-final-focused.log` and
`.artifacts/gpu-coupled-website.log`.

Direct `checkpoint_coupled(owner)` calls now hold the same reentrant cycle lock
as `owner.checkpoint()`. Review found that the standalone helper could otherwise
copy the event checkpoint and aggregate counters at different cycle boundaries.
A deterministic threaded regression fails against the previous unlocked helper
and passes with the lock. The full coupled test module then passed **18 tests
with one intentional Warp-CPU capture skip** in 62.25 s. Ruff, formatting and
checked-body Mypy passed for the affected modules. The reproducer and focused
logs are `.artifacts/gpu-coupled-lock-regression-before.log` and
`.artifacts/gpu-coupled-checkpoint-lock.log`. The ongoing full publications use
the already-locked owner method; this correction does not change their physics.

The subsequent unfiltered package gate ran against executable revision
`2b4bddefd8`, including the checkpoint lock and plan-type guards:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  --cov=particula --cov-report=term-missing --cov-fail-under=80
```

It passed **10,810 tests**, with 21 skips and one expected failure, in 514.17 s;
coverage was **91.34%**. Actual CUDA cases ran. Full Ruff/format checks passed
for 764 files, and ordinary Mypy passed for 681 sources. Documentation structure
reported zero errors, and the topic graph validated 33 topics, 170 resources and
35 supporting pages. Logs are `.artifacts/stage-02-current-package-coverage.log`,
`.artifacts/stage-02-current-ruff.log`, `.artifacts/stage-02-current-mypy.log`,
`.artifacts/stage-02-current-structure.log` and
`.artifacts/stage-02-current-topics.log`.

The separate respiratory-refinement diagnostics passed all three tests with
warnings as errors. They verify complete source setup reuse without time
advancement, detection of a minor-species budget defect and normalization that
does not hide a small solute behind water. These are checks of the assessment,
not completion of its full 20 s numerical comparison. The pinned full comparison
and the outstanding full example publications remain required.
