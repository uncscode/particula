# Stage 3A: Thermodynamics and additional physics

Updated: 2026-11-17. Status: revised proposal; implementation not started.
Stages 1, 2A and 2B are merged; reuse their scientific references, numerical
contracts and recording/restart infrastructure. Their completion does not
qualify the new thermodynamic coupling.

## Execution and delivery

Create a dedicated worktree and `codex/stage-03a-thermodynamics` branch when
starting implementation. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome and initial scope

Deliver one conservative native particle/gas workflow using existing activity
and equilibrium calculations. The first milestone connects existing models;
AIOMFAC activity is a separate follow-on, and viscosity remains deferred.
Selectable models have distinct required inputs and mixture domains; sharing
an interface does not make BAT, kappa and AIOMFAC interchangeable.

Stage 3A owns library coupling and its scientific qualification.
[Stage 3B](stage-03b-scientific-comparisons.md) owns a reproducible published-study
comparison. Its early case screening should inform this milestone, but neither
plan requires the other to be fully completed before work begins. If external
study inputs are unavailable, 3A can proceed with a specified thermodynamic
reference case and independent published/reference-model evidence; it must not
claim that the experimental comparison is complete.

## Starting point

Reviewed against checkout `afd4f65856`:

- `Equilibria` obtains auxiliary `partitioning_inputs` and attaches an
  `EquilibriumResult`. The built-in partitioning result path does not update
  native particle masses or gas concentrations, and the runnable rejects CPU
  box contexts. Existing attachment tests do not establish native conservation.
- `LiquidVaporPartitioningStrategy` takes concentrations in micrograms per
  cubic metre and molar masses in grams per mole. Native gas concentrations are
  kg/m³, particle masses are kg per represented particle, and native molar
  masses are kg/mol. Conversion and population weighting need explicit owners.
- `ActivityNonIdealBinary` accepts `[water, organic]` and returns organic
  activity only. It has no temperature argument. A water/organic activity
  vector for coupled condensation must be specified and tested explicitly.
- The activity guide claims temperature-dependent BAT parameters and a
  250–310 K range. Those claims are not established by the inspected strategy
  interface; audit the underlying model and primary references before retaining
  them. Do not infer a physical validity range from an accepted input range.
- The current GPU support guide documents adaptive RK23/ROS3 with ideal/kappa
  activity. That evidence does not establish BAT or equilibrium GPU support.

## BAT activity in GPU-resident condensation

For one water/organic pair with fixed BAT parameters, organic mole fraction
determines the binary composition. Generate a one-dimensional table on the CPU
with the existing BAT calculation, retaining both water and organic activity.
Identify each table by its organic properties, functional group, model version,
phase convention and supported conditions. Upload it once during preparation.
At each condensation evaluation, compute organic mole fraction from the current
particle masses and molar masses on device, interpolate both activities, and
pass them to the surface vapor pressure calculation. This keeps BAT evaluation
on the CPU and all per-step particle data and lookups on the GPU.

Set an error tolerance for each activity before choosing the grid. Check
interpolation against direct CPU BAT at independent compositions between nodes,
including dilute and near-pure limits. Specify the pure-component limits and
skip empty slots before computing mole fractions. If the case includes phase
separation, qualify the lookup on each supported phase branch and specify how
the caller supplies phase composition. A lookup on bulk composition alone
cannot determine coexisting phase amounts.

On actual CUDA, compare the resident trajectory with direct CPU activities
under matched physical and numerical settings. Check particle growth, gas
depletion and species budgets under grid and timestep refinement. For ROS3,
qualify the interpolant derivatives used by the Jacobian, including grid knots,
before listing support. For supported capture and restart, verify stable device
buffers on replay and verify table identity when restoring a checkpoint.

## Decisions before implementation

Record a bounded case specification in this plan before coding. Resolve:

| Decision | Required record |
| --- | --- |
| Physical problem | Named species, model, phases, composition and temperature range, and target observable. Prefer a small organic/water partitioning case if its inputs and references are adequate; no study is selected yet. |
| Water boundary | Prescribed water activity with an accounted external reservoir, or finite closed water inventory with a qualified coupled solve. Fixed water activity alone does not implement the latter. |
| Population and phases | Supported native distribution, boxes and slot layout; rule for distributing bulk equilibrium mass among particles; treatment of alpha/beta phases and curvature. |
| Coupling | Which species are equilibrated, which evolve kinetically, and the accepted-step/process-order boundary at which updates occur. |
| Evidence | Independent reference, exact observables, absolute/relative tolerances and numerical refinement criteria set before evaluation. |
| Execution | CPU baseline and explicit GPU, solver and restart support matrix; unsupported combinations must reject before mutation. |

Use the smallest scientifically adequate population model for the first case.
A bulk equilibrium result does not determine size-resolved composition or phase
morphology. Either supply and qualify that allocation rule or reject the layout.
Do not silently collapse separated phases or assign identical composition to a
heterogeneous population. A representation change needs its own explicit scope.

## Work packages

1. **Qualify model inputs and outputs.** Audit activity reference states, units,
   water/organic output ordering, phase selection, domain limits and solver
   convergence reporting. Keep existing standalone calculations compatible;
   decide explicitly whether native application is an opt-in adapter or a new
   process contract. Deliver the case specification, independent reference
   fixtures and any corrections to misleading thermodynamics guidance.
2. **Connect equilibrium to native inventories.** Build explicit particle-to-gas
   species maps, including inert/unmapped lanes. Compute bulk inputs using
   population weights and represented volume, solve into detached candidate
   state, and validate residuals, nonnegativity, phase allocation and budgets
   before publishing coordinated particle/gas changes. Specify failure
   atomicity, ownership and batched behavior; retain slot capacity and counts
   unless a separately justified process changes them. Deliver a native-state
   regression that fails with result attachment alone.
3. **Integrate kinetics and workflow.** Apply equilibrium only at the declared
   process boundary, avoid duplicate kinetic/equilibrium transfer, and account
   for imposed water exchange as a source/sink. Qualify the selected timestep
   and splitting policies. Record physical outputs and reservoir budgets using
   Stage 2B; test restart with matching thermodynamic configuration or explicitly
   reject unsupported restart profiles. Repeated equilibrium applications must
   not introduce drift beyond the declared tolerance.
4. **Qualify and publish the bounded case.** Demonstrate ideal/limiting behavior,
   agreement with an independent equilibrium reference, and numerical
   refinement. Compare long-time kinetics only where activities, phases,
   curvature and reservoir assumptions match. Publish a runnable example,
   compact outputs, CPU/GPU support table and durable validation record. Link
   the Stage 3B comparison when available without duplicating its study report.

## Owning code and evidence

- [Activity calculations](../particula/activity/),
  [particle activity strategies](../particula/particles/activity_strategies.py),
  and [equilibria](../particula/equilibria/).
- [Condensation](../particula/dynamics/condensation/),
  [custom box processes](../particula/box_process.py), and
  [process extension support](../particula/execution/process_extensions.py).
- [Activity guide](../docs/guide/processes/activity.md) and
  [activity/equilibrium example tests](../examples_tests/native_activity_equilibria_examples_test.py).
- [Equilibrium tests](../particula/equilibria/tests/),
  [activity strategy tests](../particula/particles/tests/activity_strategies_test.py),
  [native ownership](../docs/guide/data-model/ownership.md), and
  [GPU support](../docs/guide/gpu/support.md).

## Acceptance and validation

- The selected case and numerical tolerances are recorded before implementation;
  physical assumptions and reference provenance remain visible beside results.
- The integrated example changes native particle/gas state, preserves closed
  species inventories independently of the production conversion helpers, and
  reports external water exchange separately. Check gas and particle errors
  separately so a large reservoir cannot hide a bad particle result.
- Independent ideal/limiting results and a relevant published/reference-model
  comparison support the claimed range. Numerical convergence and experimental
  validity remain distinct; using the same BAT helper twice is not independent
  validation.
- Unequal particle/gas species widths, reordered and inert gas lanes, non-unit
  resolved volumes, heterogeneous boxes, inactive slots, phase-boundary cases,
  invalid inputs and failed solves have explicit support or rejection tests.
  Unsupported population/phase layouts fail before mutation. No silent clipping
  of negative inventories or activity values substitutes for a valid solve.
- Failure tests verify the promised state/identity boundary; deterministic
  repeat application and any supported restart retain the specified budgets.
- A support table identifies CPU and GPU capabilities. Unsupported GPU
  thermodynamics are rejected explicitly; CPU support does not imply CUDA
  support. New GPU claims require real CUDA evidence.
- Adjacent activity/equilibrium/process tests and the affected published example
  pass under the repository validation policy. Any new CUDA behavior also needs
  parity and lifecycle checks for the supported execution profiles.

## Completion boundary

The first milestone is complete when the selected existing-model case meets
the acceptance criteria and its implementation, example, evidence and support
boundaries are delivered in a PR. A case specification alone is planning
completion, not completion of Stage 3A implementation. No version bump or
release is part of this milestone.

AIOMFAC remains a separately scoped follow-on: preserve its own reference,
packaging/licensing, supported-speciation, CPU/CUDA resident and bounded-table
qualification decisions when drafting that plan. No AIOMFAC implementation or
table support is claimed here; viscosity remains outside its activity scope.

Further vapor wall loss, inorganic thermodynamics, activation/parcel feedback,
aggregate properties, composition-conserving binned coagulation and nucleation
laws require a demonstrated comparison gap and separate acceptance case.
Chemistry belongs to [Stage 6](stage-06-chemistry-model.md). The handoff documents
available activities, reference states, phase volumes and reservoir ownership;
it does not supply a reaction mechanism or infer reverse reaction rates.

Follow the [validation reference](../plugins/particula-helper/references/repository-validation.md)
for adjacent tests and affected example refreshes.
