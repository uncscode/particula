# Stage 2A execution target and evidence

Status: initial implementation delivered in
[PR #1659](https://github.com/uncscode/particula/pull/1659). The discrete-bin review
correction below is locally qualified and awaits publication authorization.
Worktree: `/tmp/particula-stage-02a`.
Branch: `codex/stage-02a-numerical-integration`.
Verified upstream base: `356fc054cd9a4f4ececc3c64c7144e889a789494`.
PR target: `uncscode/particula:main`; separate from the Stage 2B PR.
Merge is outside execution scope. The PR is open; no merge is claimed.

## Completion boundary

All six work packages and acceptance items in
[Stage 2A](stage-02a-numerical-integration.md) remain required. Evidence below
qualifies the stated methods and cases; it does not waive unsupported-regime
rejection, workflow integration, review or delivery requirements.

The shared case is Stage 1 monodisperse closed-box growth with common dilution.
Additional assessment covers exact decay, smooth/stiff reversible exchange,
near-depletion exchange, noncommuting deterministic operators and Robertson
kinetics. Predeclared global radius/inventory targets are 1e-3, 1e-5 and 1e-7,
with each observable scaled by its own characteristic inventory and reference
uncertainty below 1% of target. The native adaptive report explicitly identifies
local tolerances that meet those targets and settings that do not.

## Requirement ledger

- [x] Independent reference checks and fixed/adaptive method comparison:
      analytical decay/exchange, radius quadrature, DOP853/Radau cross-checks,
      Radau/BDF Robertson reference, and retained failures.
- [x] Existing native condensation compared with shortlisted methods on growth,
      equilibrium and depletion; full-step limiting proposals retained.
- [x] Stiff candidate and conservative-transfer prototype evaluated; RK45, SSP
      and deferred correction dispositions documented without invented timings.
- [x] One-box and heterogeneous actual CUDA feasibility, shared/independent
      controller work, synchronization, retained storage and measured timings.
- [x] Measured method/regime decision and supported-method guide.
- [x] Explicit continuous-rate/exact-map contracts; equilibrium/stochastic
      inputs rejected rather than inferred to be deterministic ODEs.
- [x] Detached trial/accepted states, clocks, physical tolerance weights,
      bounded steps/trials, failure preservation and numerical work counters.
- [x] Sequential and nested Strang schedules, unique component names,
      component-order admission and absolute stage clocks.
- [x] Native CPU RK23/ROS2 and bounded CUDA RK23 profile; independent accuracy
      tests, no hidden CPU fallback and explicit capture rejection.
- [x] Species/volume/inactive-slot budgets, nonautonomous forcing stage times,
      domain rejection and whole-cycle failure tests.
- [x] Complete cross-workflow forcing/output/restart compatibility with Stage 2B.
- [x] Final full validation, complete diff review, commit, push and separate PR.

## Owning implementations and durable evidence

`execution.ode` supplies pure ODE callbacks and RK3(2)/KPP ROS2 controllers.
`condensation_ode` owns scaled native CPU condensation; numerical Jacobian work
is explicit and latent heat is excluded. `splitting` owns deterministic complete
cycles, nested groups, exact dilution and continuous condensation adapters.
`gpu_condensation_ode` owns device-resident RK23 trial arrays, a shared maximum
error clock and explicit scalar readbacks. It reuses the mapped physical rate
kernel; ordinary steps do not evaluate CPU physical rates or download states.
Snapshots explicitly download accepted state. Full active-slot evaporation,
native GPU ROS2, PDF on GPU and adaptive CUDA capture are rejected/excluded.

The [method assessment](../docs/contribute/validation/numerics/adaptive-methods.md)
is the durable decision record. Its retained data include original preliminary
reports and the source-hashed native CUDA assessment. The
[method guide](../docs/guide/processes/numerical-integration.md) owns usage,
tolerance units, support matrix, forcing and custom-flow contracts.

The final native study completed 27 baseline runs, eight added observable-target
checks and two 32-box controller comparisons on RTX 5060 / Warp 1.11.1. Loose
RK23 error estimates are unreliable near depletion; stricter qualified settings
are recorded rather than masking those results. Native ROS2 meets the targets
but costs substantially more. No general end-to-end speedup is claimed.

## Current validation

Run from this worktree with `PYTHONPATH=.` and the selected `.venv`:

- Full package, warnings as errors and standard non-slow/non-performance
  selector: **10574 passed, 19 skipped, 57 deselected, 1 xfailed** in 217.16 s.
  Actual CUDA tests ran. Skips/deselections remain outside the evidence claim.
- Full package coverage: **91%**, normal 80% gate passed. Seven changed runtime
  targets: **86% aggregate**. Per-target coverage: ODE 93%, CPU adapter 93%, GPU
  owner 93%, splitting 91%, prepared condensation 88%, prepared registry 92%,
  Warp algebra 21% (compiled bodies do not execute under Python line tracing).
- Focused numerical/existing-condensation suite: 128 passed. After adding the
  near-depletion and local-oracle regressions, the affected subset: 26 passed.
- Full Ruff check and format check passed (734 files); full standard Mypy
  passed (654 source files). Checked-body checks for new GPU modules and the
  condensation planner pass with imported modules silent. A broader optional
  checked-body pass exposes existing imported-module issues; it is not reported
  as green. The standard full-repository check is green.
- All three numerical guide code blocks executed to 10 s, including native CUDA
  and coupled Strang simulation, using the declared native reference fixture.
- Documentation topic/structure checks passed. Astro built 540 pages; post-build
  verification checked 670 HTML pages, 308 API modules and 130 redirects with no
  errors. Rendered documentation structure also reported zero errors.
- Full `scripts/tests -Werror --no-cov`: **324 passed** in 58.58 s.

Coverage was collected with `coverage run --source=particula`, followed by the
normal gate and explicit per-target reports. Directory discovery avoids the
NumPy import/unload issue caused by dotted module discovery. No warning or
coverage gate was weakened. Logs and coverage files remain under the ignored
`.artifacts/numerics/`; scientific measurements are retained under `docs/data/`.

## Cross-workflow review and delivery

Stage 2B now supplies table forcing with one-sided knots, exact physical-time
recording, prepared fixed-grid observation, and adaptive CPU/GPU and split disk
restart. Its combined numerical/workflow tests passed 154 cases, including
actual CUDA, with independent forcing references and exact future continuation.
The complete combined package passed 10679 tests with 91.18% coverage. This
integration evidence belongs to the Stage 2B branch; the method measurements
above remain tied to the unchanged Stage 2A executable sources.

Review covered continuous-rate versus finite-step boundaries, RK/ROS coefficients,
scaled tolerance units, shared maximum-error control, species/volume coupling,
trial publication, split clocks, explicit GPU transfers and failure/capture
boundaries. The later discrete-bin review finding and its correction are
recorded below. Stage 2B
also closed a review finding in its archive writer: existing host checkpoints
must not be serialized inside active CUDA capture.

Stage 2A is delivered in PR #1659 against the verified upstream main. Stage 2B
targets this branch explicitly so its workflow changes can be reviewed
independently. Merging either PR remains outside this task.

## Discrete-bin review correction (2026-11-16)

PR #1659 identified that zero-density discrete bins may retain their defining
particle masses, unlike empty resolved-particle slots. The original CUDA trial
guard applied the resolved rule to both representations. The failure reproduced
on Warp CPU and actual CUDA with native-valid, unit-volume discrete fixtures.
The guard now receives the population interpretation and requires zero mass only
for resolved free slots. This changes admission of discrete states; resolved
physics, rates and adaptive controls are unchanged.

Regression coverage preserves zero-density bin masses, compares heterogeneous
three-box evolution against CPU condensation, checks an independent species
budget, and retains resolved free-slot rejection on both devices. All **27** GPU
ODE tests pass. Full package validation with warnings as errors passed **10,636
tests, 20 skips and one expected failure**, with **91.34% coverage** (80% gate).
Standard Mypy passed **654 sources**; Ruff and format passed **734 files**.

Commands: `PYTHONPATH=. .venv/bin/python -m pytest particula --cov=particula
--cov-report=term-missing --cov-fail-under=80 -Werror -q`, full standard Mypy,
and Ruff check/format over `particula/ examples_tests/ scripts/`. Logs are in
`.artifacts/discrete-bin-*.log`. Earlier scientific measurements retain their
original source hashes; this regression qualification does not relabel them.
The remote review thread remains unresolved until the correction is published.
