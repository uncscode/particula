# Stage 3A: Thermodynamics, chemistry, and additional physics

Status: proposed. Entry dependencies: relevant
[scientific references](stage-01-scientific-correctness.md) and
[numerical contracts](stage-02a-numerical-integration.md).
Coordinate with [Stage 3B comparisons](stage-03b-scientific-comparisons.md).

## Outcome and initial scope

Make existing activity and equilibrium capabilities usable in a scientifically
consistent native-state workflow. Start with the connection needed by the first
selected comparison. Subsequent chemistry and additional thermodynamics are
separately scoped extensions, not prerequisites for every later stage.

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
5. **Define chemistry coupling from a research requirement.** Select a named
   mechanism or external solver only after specifying species, reaction rates,
   environmental forcing, and exchange intervals. Use existing custom-process
   boundaries where suitable; track elemental, mass, and charge budgets as
   appropriate. Do not assume a previously discussed mechanism is approved by
   this broad plan.
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
- Each later chemistry/physics extension names its reference, selected
  mechanism, conservation expectations, coupling policy, and reproducible case
  before being treated as part of the framework's supported science.

## Completion boundary

The first Stage 3A milestone is the existing-model connection needed by the
selected case. Broader chemistry and additional physics can continue in later
increments without blocking qualified GPU studies or inference on that case.
Follow the [validation reference](../plugins/particula-helper/references/repository-validation.md)
for adjacent tests and affected example refreshes.
