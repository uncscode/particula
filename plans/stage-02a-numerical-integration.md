# Stage 2A: Numerical integration and process splitting

Status: proposed. Entry dependency: initial
[Stage 1 reference cases](stage-01-scientific-correctness.md).
Develop alongside [Stage 2B](stage-02b-simulation-workflow.md).

## Outcome

Give CPU and GPU simulations explicit numerical-method choices and useful error
control, with demonstrated accuracy for supported processes and combinations.
Separate the method that advances a process from the policy that schedules
different processes.

## Current behavior

[RunnableSequence](../particula/runnable.py) advances processes in order within
equal substeps. CPU process execution stages native state before publication.
[Condensation](../particula/dynamics/condensation/condensation_strategies.py)
includes an explicit update and a two-sweep staggered Gauss-Seidel method that
freezes particle radii/composition during its sweeps. This is not an existing
general Strang scheduler. Direct GPU condensation uses four fixed substeps.

## Work packages

1. **Specify the stepping contract.** Distinguish continuous-rate processes,
   exact finite-step maps, equilibrium operations, and stochastic events.
   Define trial state, accepted state, stage time, diagnostics, and failure
   semantics. Do not assume every runnable's `rate` is an ODE derivative.
2. **Evaluate integration methods.** Retain first-order explicit stepping as a
   characterized baseline. Evaluate a second-/third-order embedded Runge-Kutta
   pair for suitable continuous dynamics. Compare stiff condensation against a
   suitable implicit or specialized reference method before selecting defaults.
   Higher order, stability, positivity, and error estimation are separate goals.
3. **Add error control.** Define absolute/relative tolerances and error weights
   for gas and particle quantities, accepted/rejected trials, minimum/maximum
   step limits, and failure reporting. Advance gas and particle inventories
   consistently at every stage. Measure how inventory limiting affects formal
   order near depletion rather than assuming clipping preserves it.
4. **Add process splitting.** Evaluate sequential and symmetric Strang schedules
   in the sequence/scheduling layer. Preserve explicit groups and absolute
   stage times. Require suitable component accuracy before claiming overall
   second-order behavior. Keep internal condensation batching separately
   configurable; do not force saving or arbitrary custom runnables through RK.
5. **Implement supported CPU/GPU paths.** Share numerical specifications and
   independent oracles, not just a backend parity test. Resolve shared versus
   per-box adaptive timesteps, resident trial buffers, and CUDA capture limits.
   Preserve explicit transfers and reject unsupported combinations without
   hidden CPU fallback.
6. **Publish the supported methods.** Explain accuracy regimes, defaults, cost,
   and compatibility with custom processes, forcing, snapshots, and restart.

## Decisions that require evidence

- Which error norm controls the selected scientific observable without hiding
  small-species errors behind large reservoirs?
- Is an embedded explicit pair adequate for the target condensation regimes,
  or is a stiff solver needed in the first supported workflow?
- How are stochastic updates scheduled and their distributions preserved when
  continuous trials are rejected? Copying RNG state alone does not establish
  an unbiased adaptive stochastic algorithm.
- Which adaptive policies can stay device resident, and which cannot support
  the existing fixed CUDA capture/replay contract?

## Owning code

- [Runnable scheduling](../particula/runnable.py) and
  [CPU batch transactions](../particula/_cpu_batch.py).
- Process strategies under [dynamics](../particula/dynamics/) and their tests.
- [Prepared execution](../particula/execution/prepared.py),
  [prepared condensation](../particula/execution/prepared_condensation.py),
  and [direct kernels](../particula/gpu/kernels/).
- [Condensation stiffness evidence](../docs/contribute/validation/numerics/condensation-stiffness.md).

## Acceptance and validation

- Demonstrate observed order on smooth independent reference cases and report
  departures at depletion, discontinuities, or stochastic transitions.
- Tightening error tolerances improves the selected observables against an
  independent reference over a declared range.
- Trial rejection does not publish partial state, advance the accepted clock,
  emit output, or corrupt supported RNG continuation. Preserve documented
  boundaries for failures after already accepted work.
- Gas/particle species budgets, positivity requirements, non-unit volumes, and
  inactive slots remain valid. Time-varying forcing uses correct stage times.
- Actual CUDA tests qualify advertised GPU methods. Unsupported adaptive,
  stochastic, or capture combinations fail clearly.

Follow [repository validation](../plugins/particula-helper/references/repository-validation.md).
Method references: [SciPy integration guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and [SUNDIALS splitting methods](https://sundials.readthedocs.io/en/v7.8.0/arkode/Usage/SplittingStep/SplittingStepCoefficients.html).
