# Stage 3A: Thermodynamics and additional physics

Status: proposed. Entry dependencies: relevant
[scientific references](stage-01-scientific-correctness.md) and
[numerical contracts](stage-02a-numerical-integration.md).
Coordinate with [Stage 3B comparisons](stage-03b-scientific-comparisons.md).

## Execution and delivery

Create a dedicated worktree and `codex/stage-03a-thermodynamics` branch when
starting this plan. Develop and validate there, then commit, push, and create
a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome and initial scope

Make existing activity and equilibrium capabilities usable in a scientifically
consistent native-state workflow. Start with the connection needed by the first
selected comparison. The configurable chemistry model has its own later
[Stage 6 plan](stage-06-chemistry-model.md). Further thermodynamics and other
physics remain separately scoped extensions, not prerequisites for every stage.

## Starting point

Particula already contains activity models, BAT-related thermodynamics, and
equilibrium calculations. The current `Equilibria` runnable extracts auxiliary
partitioning inputs and attaches result objects; that alone does not establish
conservative integration with native particle/gas inventories. Existing custom
process interfaces provide extension points without establishing a complete
chemistry mechanism or thermodynamic coupling.

## Work packages

1. **Specify the first thermodynamic problem.** Identify species, phases,
   composition range, temperature range, water treatment, and required
   observable. Audit the model's actual assumptions and documentation before
   claiming temperature dependence or a broader mixture domain.
2. **Connect existing calculations to native state.** Define explicit species
   maps, molar/mass conversions, representative-volume accounting, and ownership
   of equilibrium results. Apply validated changes to particle and gas state
   together, preserving unmapped/inert species and failure boundaries.
3. **Couple equilibrium and kinetics deliberately.** State which species evolve
   kinetically, which are equilibrated, and when equilibrium is applied. Avoid
   counting the same transfer twice. Treat externally prescribed humidity or
   vapor reservoirs as explicit sources/sinks, not closed-system conservation.
4. **Validate the integrated case.** Compare against ideal and limiting cases,
   independent equilibrium results, and the selected published study. Evaluate
   timestep/splitting sensitivity and consistency of long-time kinetic limits
   where the physical assumptions permit that comparison.
5. **Define the thermodynamic handoff to chemistry.** Identify which activities,
   phase volumes, and equilibrium relations a later reaction mechanism may use.
   Preserve distinct kinetic and equilibrium responsibilities and explicit
   reservoir accounting. Species/reaction configuration and chemical integration
   belong to Stage 6; connecting existing activity models does not claim a
   chemistry mechanism or automatically supply reverse reaction rates.
6. **Select further physics.** Candidate extensions include vapor wall loss,
   inorganic thermodynamics, activation/parcel feedback, aggregate properties,
   composition-conserving binned coagulation, and new-particle-formation laws.
   Select each from a demonstrated comparison gap with its own scientific
   acceptance case. These are candidates, not an automatic implementation list.

## Owning code and evidence

- [Activity calculations](../particula/activity/),
  [particle activity strategies](../particula/particles/activity_strategies.py),
  and [equilibria](../particula/equilibria/).
- [Condensation](../particula/dynamics/condensation/),
  [custom box processes](../particula/box_process.py), and
  [process extension support](../particula/execution/process_extensions.py).
- [Activity guide](../docs/guide/processes/activity.md) and
  [activity/equilibrium example tests](../examples_tests/native_activity_equilibria_examples_test.py).

## Acceptance and validation

- The initial integrated example changes native state consistently, conserves
  the appropriate closed-system species inventories, and accounts explicitly
  for any external reservoir or reaction source/sink.
- Independent ideal/limiting results and a relevant external comparison support
  the documented physical range. Numerical convergence is assessed separately.
- Mixed species widths, non-unit resolved volumes, inactive slots, and invalid
  thermodynamic inputs have explicit behavior and tests.
- A support table identifies CPU and GPU capabilities. Unsupported GPU
  thermodynamics are rejected explicitly; CPU support does not imply CUDA
  support. New GPU claims require real CUDA evidence.
- Each later physics extension names its reference, model assumptions,
  conservation expectations, coupling policy, and reproducible case
  before being treated as part of the framework's supported science.

## Completion boundary

The first Stage 3A milestone is the existing-model connection needed by the
selected case. Chemistry proceeds through [Stage 6](stage-06-chemistry-model.md),
and additional physics can continue in later increments without blocking
qualified GPU studies or inference on that case.
Follow the [validation reference](../plugins/particula-helper/references/repository-validation.md)
for adjacent tests and affected example refreshes.
