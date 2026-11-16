# Stage 2A: Numerical integration and process splitting

Status: in progress; see the [delivery record](stage-02a-delivery.md).
Entry dependency: initial
[Stage 1 reference cases](stage-01-scientific-correctness.md).
Develop alongside [Stage 2B](stage-02b-simulation-workflow.md).

## Execution and delivery

Create a dedicated worktree and `codex/stage-02a-numerical-integration` branch
when starting this plan. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

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

## Initial method assessment

Assessment date: 2026-11-15. The following is a source- and literature-based
shortlist, not measured Particula performance or an approved default solver.
Perform the bounded comparison below before committing to a production method.

Here, error correction includes predictor/corrector or deferred-correction
methods that improve a trial solution. Error control estimates local error and
chooses whether to accept the step and how large the next step should be.
Neither conservation nor a higher formal order alone establishes accuracy.

| Candidate | Potential fit | Main question or limitation | Initial disposition |
| --- | --- | --- | --- |
| Euler/Heun predictor-corrector and step doubling | A transparent first-/second-order baseline and a way to assess an existing deterministic step | Compare a full step with two half steps only where deterministic consistency/order assumptions apply; extrapolation can violate positivity | Use as a diagnostic baseline; do not assume it corrects the stochastic sampler |
| Embedded RK3(2), with RK5(4) as a comparator | Nonstiff continuous gas/particle rates; an embedded estimate supports adaptive control | Shared stage inventories, extra rate evaluations, depletion, and stability limits; higher order does not remove stiffness | First explicit candidate; select order by cost at a target accuracy, not by order alone. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) |
| SSPRK(2,2) / SSPRK(3,3) | Deterministic operators with a demonstrated forward-Euler positivity or other convex stability property | The inherited property requires a timestep restriction and a suitable base operator; these schemes alone do not supply error control | Conditional candidate when the base operator supports the required property. [Gottlieb, Shu, and Tadmor](https://www.math.umd.edu/~tadmor/pub/linear-stability/Gottlieb-Shu-Tadmor.SIREV-01.pdf) |
| Rosenbrock 2(1)/3(2); implicit Radau/BDF references | Stiff condensation or later chemical kinetics | Jacobians, linear solves, stage memory, and batched GPU cost; stiff stability does not itself guarantee nonnegative inventories | Assess a small Rosenbrock candidate; use tight-tolerance Radau/BDF solutions as cross-checked numerical references. [KPP](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html), [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) |
| Modified Patankar-Runge-Kutta | Conservative production/destruction exchange between reservoirs | Establish the required production/destruction form, zero-species treatment, order conditions, and all relevant invariants; positivity alone is not accuracy | Explore a second-order candidate for closed transfer cases. Published adaptive variants provide a starting point, not a drop-in proof for arbitrary chemistry. [Order conditions](https://arxiv.org/abs/1702.04589), [adaptive study](https://d-nb.info/1247659747/34) |
| Spectral deferred correction | Raising accuracy by iterative correction of a quadrature-based solution | Additional nodes, sweeps, and stored stages; stiff/depletion behavior and convergence must be assessed | Limited exploratory comparison if simpler candidates leave a clear accuracy/cost gap. [Causley and Seal](https://arxiv.org/abs/1706.06245) |

**Working recommendation:** compare the current methods, embedded RK3(2), and
one stiff candidate first. Add a modified Patankar prototype for a suitable
closed transfer problem. Screen SSP and deferred correction for specific needs
before expanding implementation. Exact updates such as dilution should remain
available rather than being replaced by a less accurate general solver.

These priorities are inferences about Particula's requirements, not conclusions
from new runs. An embedded component error estimate does not include all
process-splitting error or bound global solution error. Increasing component
order above two also does not make a generic Strang-composed solution exceed
second order; measure the complete trajectory separately.

## Bounded exploration before production implementation

1. **Declare cases and observables.** Include exact decay; a smooth coupled
   gas/particle exchange with an independent solution; condensation approaching
   equilibrium and near depletion; and a deterministic two-process case that
   exposes splitting error. Include a small stiff reaction system to inform
   [later chemistry](stage-06-chemistry-model.md), without building that feature
   in Stage 2. Specify units, physical ranges, and reference uncertainty.
2. **Compare numerical work at equal accuracy.** Sweep fixed timesteps and
   adaptive tolerances. Record global observable error, observed order, local
   estimate reliability, conservation, positivity, accepted/rejected steps,
   rate/Jacobian evaluations, linear solves, memory, and wall time. Report
   failures and limiting/clipping activity rather than dropping difficult cases.
3. **Check the reference itself.** Use analytical cases where possible. For
   cases without them, tighten an independent solver's tolerances and compare
   another suitable method until reference uncertainty is below the study's
   error targets. A single fine-step production run is not an independent oracle.
4. **Probe CPU/GPU feasibility.** Compare one box and heterogeneous ensembles;
   assess shared versus per-box step control, divergence, scratch storage, and
   synchronization. Qualify actual CUDA only for shortlisted prototypes; do not
   infer GPU efficiency from CPU timings or require a full port of every candidate.
5. **Record the decision.** Publish a compact method/regime matrix and measured
   recommendation under the numerical validation records. Identify supported,
   deferred, and rejected candidates with reasons, default candidates, and
   remaining uncertainty. Production implementation follows this decision.

The exploration is complete when it supports a method choice for the selected
cases, including whether a stiff branch is necessary. It does not need to
implement every method in the shortlist. Finite local errors, positivity, and
formal order must not be substituted for the stated observable accuracy target.

## Work packages

1. **Specify the stepping contract.** Distinguish continuous-rate processes,
   exact finite-step maps, equilibrium operations, and stochastic events.
   Define trial state, accepted state, stage time, diagnostics, and failure
   semantics. Do not assume every runnable's `rate` is an ODE derivative.
2. **Complete the method assessment.** Run the bounded exploration above and
   select the smallest justified set of methods. Keep the stepping contract
   usable by the later chemistry process, including a coupled rate/Jacobian
   interface where the chosen method needs one. Do not force chemistry into an
   explicit method chosen only for a nonstiff condensation example.
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

- A measured method-assessment record supports the selected methods and their
  supported regimes; the initial literature shortlist is not treated as runtime
  qualification or a requirement to ship every candidate.
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
