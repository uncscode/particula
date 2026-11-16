# Stage 2A execution target

Status: in progress; no completion claim.
Worktree: `/tmp/particula-stage-02a`.
Branch: `codex/stage-02a-numerical-integration`.
Verified upstream base: `356fc054cd9a4f4ececc3c64c7144e889a789494`.
PR target: `uncscode/particula:main`. No open matching PR at start.

## Completion boundary

All six work packages and every acceptance item in
[Stage 2A](stage-02a-numerical-integration.md) remain required. The first
assessment is a milestone, not completion of the stage.

The shared scientific case is Stage 1 monodisperse closed-box growth with
common dilution. Supplement with exact decay, reversible closed exchange,
near-depletion exchange, noncommuting deterministic operators and a small stiff
reaction system. Keep physical species budgets distinct from observable error.
Predeclare dimensionless trajectory error targets of 1e-3, 1e-5 and 1e-7;
scale radius by initial radius and each inventory by its own declared initial
or characteristic inventory. Reference uncertainty must be below 1% of target.
Use non-unit resolved volume and inert gas lanes in native integration tests.

## Required evidence ledger

- [ ] Independent reference checks and fixed/adaptive method comparison.
- [ ] Current condensation methods compared with shortlisted methods on native
      growth, equilibrium and depletion cases; limiting activity retained.
- [ ] Stiff candidate and conservative transfer prototype evaluated; SSP and
      deferred correction screened with explicit reasons.
- [ ] One-box and heterogeneous ensemble CUDA feasibility and measured costs.
- [ ] Selected methods documented with support/regime matrix and uncertainty.
- [ ] Continuous-rate, exact-map, equilibrium and stochastic stepping contracts.
- [ ] Accepted/trial state, clocks, tolerance weights, step limits and diagnostics.
- [ ] Sequential and grouped Strang schedules with correct absolute stage times.
- [ ] Relevant CPU and actual CUDA implementations, independent accuracy tests.
- [ ] Rejection preserves state/clock/output; unsupported stochastic/adaptive and
      capture combinations reject explicitly without CPU fallback.
- [ ] Units, maps, volume, inactive slots, species budgets and forcing tests.
- [ ] Supported methods guide, full validation, diff review, commit and PR.

## Shared Stage 2B decisions

Record only accepted complete steps. Explicit output times initially land on
step boundaries by shortening the proposed step; this policy may change the
numerical trajectory and must be measured. Observers never execute at internal
stages or rejected trials. Restart must retain controller and scheduling state.
Existing exact dilution remains exact. Existing runnable rate() methods are not
assumed to be continuous derivatives. No new method becomes a default until
assessment supports it. CPU and CUDA support must be stated separately.

## Preliminary assessment completed

`scripts/assess_numerical_methods.py` ran 60 CPU screening combinations on
exact decay, smooth/stiff/near-depletion linear exchange and Robertson kinetics.
The raw report is `.artifacts/numerics/screening.json`; it records source hash,
versions, errors per species, positivity, budgets, timings and solver work.
Both explicit candidates reached the declared 200000-evaluation cap for
Robertson. Tight Radau/BDF references differ by 1.754e-11 or less.
This motivates retaining a stiff candidate; it does not select a production
method. Native condensation, conservative prototypes, splitting, CUDA and
controller diagnostics remain unqualified. Ruff check and formatting passed.
The shared environment was reused and import identity verified inside this
worktree. No runtime methods or default behavior have changed yet.

## Native and splitting assessment

Added native growth/equilibrium/depletion comparisons (66 combinations) with
non-unit resolved volume, inert gas and inactive slot checks. The reference
uses an independent radius ODE with Kelvin curvature, cross-checked with
DOP853/Radau and independently tested against inverse-radius quadrature.
Some loose-tolerance adaptive native trials leave the physical inventory
domain; failures are retained. Production stepping needs domain-aware trial
rejection, not clipping or acceptance of an invalid stage.

Exact noncommuting exchange/removal maps give finest-grid observed orders
0.9994 and 1.0004 for sequential orderings and 1.9998 for Strang. This only
qualifies the composition mathematics, not a production scheduler.

All preliminary reports are retained in
`docs/data/numerics/stage-02a-preliminary.json` with source hashes.
`PYTHONPATH=. .venv/bin/python -m pytest
scripts/tests/numerical_assessment_test.py -q -Werror --no-cov`: 3 passed.
Ruff and format checks passed for all new assessment modules and tests.

Next: conservative transfer and stiff prototypes, actual CUDA feasibility,
then supported method selection and the trial/accepted-state implementation.
The complete Stage 2A/2B goal remains unchanged and incomplete.

## Stepping core and remaining qualification

Implemented concrete `particula.execution.ode` with pure continuous-rate
contracts, detached read-only callback state, embedded RK3(2), ROS-2 with
explicit Jacobian/partial-time derivative, per-component absolute/relative
tolerances, max-norm error control, bounded steps/trials and work counters.
Domain-invalid stages are rejected before rate evaluation; failures preserve
accepted state/time. No stochastic or arbitrary runnable is adapted implicitly.
This core is not exported at package root and is not yet wired to aerosols.

Independent manufactured nonautonomous solutions establish orders three/two.
Closed exchange verifies tolerance refinement and material conservation.
Failure, invalid controls, callback mutation, final time and detached inspection
are covered. Combined focused run: 19 passed with -Werror. Mypy with
--check-untyped-defs and Ruff passed for the core.

The linear MPRK22(1) prototype is conservative and nonnegative including zero
lanes; smooth/depletion refinements approach second order. The stiff transient
is unresolved on its tested grid (maximum error 0.0591 at 256 steps), so this
prototype is not selected as a general production method. Zero-donor handling
is a linear-rate limit only, not a claim about arbitrary chemistry.

Actual CUDA access required execution outside the restricted environment.
RTX 5060/Warp 1.11.1 ran independent per-box RK3(2) and ROS-2 controllers for
one and 4096 heterogeneous exchange boxes. All boxes completed; endpoint
errors were below 4.3e-8. The 4096-box warmed kernel times were 0.0106 s and
0.316 s respectively, excluding compile/upload/download. This is one
feasibility timing, not a speedup claim or native aerosol qualification.
ROS-2's lower-order estimator required 7450--34456 accepted steps versus
104--1028 for RK3(2); stiff stability alone does not make it the default.
Device array sizes exclude compiler registers/spills and are not peak memory.

Next implementation work: native mapped condensation continuous adapter,
physical-domain-aware derivative/Jacobian construction, complete splitting
scheduler, then Stage 2B recording and disk restart on those contracts.
Native CPU/CUDA qualification, shared-versus-independent box control, method
assessment publication, full validation, review, commits and PRs remain open.

## Native adapter and complete-cycle scheduler

Implemented `execution.condensation_ode.CondensationODE`: detached native
state/configuration, explicit source/target maps, scaled kg and kg/m3
coordinates with separate absolute tolerances, population weights, shared
maximum-error control across boxes, optional absolute-time T/P forcing and
bounded one-sided/central numerical Jacobian. Latent heat is rejected; full
evaporation of active slots remains a domain boundary requiring explicit
future event handling. This does not silently clip a failed trial.

Implemented `execution.splitting`: declared continuous/exact Flow components,
explicit nested sequential/Strang groups, local component stage times, complete
cycle publication and failure rollback of the current cycle. Stochastic and
equilibrium contracts are explicitly excluded from symmetric composition; no
arbitrary runnable is inferred to be an ODE. Native condensation and exact
dilution adapters exercise the shared growth/loss case.

Independent growth quadrature, per-species budgets, non-unit batched volumes,
inert/empty lanes, forcing, caller configuration detachment, component order,
nested stage clocks and failed-cycle preservation pass. Coupled growth/dilution
Strang error decreases by at least the required factor under time refinement
and closes the species budget at rtol 3e-13. Focused core/native/scheduler/
assessment suite: 30 passed with -Werror. Ruff and checked-body Mypy passed.

The new CPU contracts are implemented but not yet publicly documented or
exported. Broad native edge coverage, actual native CUDA integration, performance
and full validation remain open. Stage 2B now has an independent recorder
implementation on its own branch; cross-branch integration is pending.
