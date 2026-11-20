# Stage 6: Configurable chemistry on CPU and GPU

Status: proposed later-stage development, based on the supplied chemistry
design discussion. API names and solver defaults remain open; no mechanism
implementation or numerical qualification is claimed by this plan.

Dependencies: [Stage 1 accounting and verification](stage-01-scientific-correctness.md),
[Stage 2A solver assessment](stage-02a-numerical-integration.md),
[Stage 2B recording/restart](stage-02b-simulation-workflow.md), and the
[Stage 3A thermodynamics](stage-03a-thermodynamics.md) needed by the chosen case.
Apply [Stage 4](stage-04-gpu-qualification.md) to GPU qualification. Stage 5
autodiff is not a prerequisite, and chemistry gradients are not an initial goal.

## Execution and delivery

Create a dedicated worktree and `codex/stage-06-chemistry-model` branch when
starting this plan. Develop and validate there, then commit, push, and create
a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome

Let users define reactions among their existing gas and particle species using
one mechanism configuration shared by CPU and GPU. Species identify material
reservoirs, reactions define transformations, and rate laws define reaction
progress. The execution backend must not silently change those scientific
definitions.

Reuse existing execution extension points. CPU `BoxProcess` supports staged
native updates; explicit GPU adapters support prepared resources, a clock,
failure signaling, and declared continuation state. These are integration
points, not an existing general chemistry engine.

## Initial scope and exclusions

- Prescribed temperature, fixed species layouts, and unchanged particle counts.
- Gas-only and particle-only reactions, followed by explicitly modeled
  heterogeneous reactions sharing a gas reservoir.
- Begin particle chemistry with resolved populations. Products have existing
  storage lanes, which may initially contain zero mass. Do not create species,
  resize arrays, or turn inactive slots into particles during chemistry.
- Start with bounded first-order/mass-action and Arrhenius rate families.
  Rate coefficients may be constant; zero-order sources still need explicit
  reservoir/budget semantics. Choose a small qualification mechanism before
  claiming support for atmospheric chemistry generally.
- Defer reaction heat feedback, aqueous subphases, surface-site chemistry,
  charged chemistry, particle retirement, and PMF/PDF chemistry until their
  additional state and scientific contracts are specified.
- External mechanism/solver integration can be evaluated without committing to
  a new mandatory dependency or rebuilding a large established mechanism.

## Shared configuration and species binding

| Record | Required responsibilities |
| --- | --- |
| Species catalog/binding | Stable phase-qualified IDs; gas names or explicit particle-material mappings; molar masses; phase and concentration basis; optional elemental composition and relevant thermodynamic references |
| Reaction mechanism | Reactants/products and signed stoichiometry; reaction orders; rate law and units; rate dependencies; reversible-reaction policy; explicitly declared external reservoirs |
| Chemistry process | Mechanism, supported representation, solver settings/tolerances, process identity, declared state reads/writes, diagnostics, and continuation requirements |

Use phase-qualified references such as `gas:water` and `particle:water` to
distinguish reservoirs. Gas molar masses come from validated gas metadata;
particle bindings need explicit molar masses because material columns and
densities alone do not identify chemical species. Validate names and mappings
once at construction/preparation, retain their identity, and require rebinding
after a layout/order change. Reject ambiguous aliases and conflicting metadata.

Put reaction orders and rate laws on reactions, separately from stoichiometry.
A catalyst, light intensity, or other rate dependency is not automatically a
consumed species. A consumed oxidant must appear in the material balance.
The condensation partitioning mask must not act as a chemistry reactivity mask.
Two independent reaction sets in different phases are not cross-phase coupling.

Start with Python configuration objects and a versioned serializable dictionary
schema; YAML may later use the same schema. Compile supported mechanisms into
species indices, sparse stoichiometry, orders, rate identifiers, and parameters.
Keep solver choice replaceable. Arbitrary Python rate callbacks remain CPU-only
unless an explicit supported GPU implementation is supplied.

Cantera provides a precedent for separating phase/species/reaction definitions
and explicit units; this plan does not adopt Cantera's entire thermodynamic
model or imply file-format compatibility. See its
[mechanism guide](https://www.cantera.org/stable/userguide/creating-mechanisms.html)
and [reaction-rate definitions](https://www.cantera.org/stable/reference/kinetics/reaction-rates.html).

## Units, budgets, and shared gas availability

Native gas concentrations are kg/m³ of air; particle species masses are kg per
physical particle. Use explicit molar variables internally. Gas rates may use
mol/m³ of air, while bulk particle rates need the specified material-phase
volume. Surface/site or aqueous concentration bases require additional models;
their rate coefficients are not interchangeable with a bulk concentration law.

For bulk particle reactions, specify how phase volume is computed from the
current composition. Convert concentration-based rates to reaction extents
using the volume appropriate to each trial stage; do not silently freeze it
while reaction products change material volume. Check this conversion with
unequal material densities as well as constant-volume limiting cases.

For a heterogeneous reaction, let `delta_xi[n]` be its integrated extent in
mol per physical particle, `w[n]` the represented count, `V` the box volume in
m³, `M[i]` molar mass in kg/mol, and `nu[i]` signed stoichiometry. Apply the same
extent to both reservoirs:

```text
delta_gas_concentration[i] = M[i] * nu_gas[i]
                             * sum_n(w[n] / V * delta_xi[n])
delta_particle_mass[n, j] = M[j] * nu_particle[j] * delta_xi[n]
```

This is an accounting relationship, not a proposed integration algorithm.
Competing reactions and particles share each box's available gas. A coupled
solve or conservative shared-allocation method must prevent each particle
from independently spending that reservoir. Species-by-species clipping after
integration cannot be used to hide violated reaction stoichiometry.

Closed reaction validation checks mass consistency using molar masses and
checks elemental balance when compositions are supplied. Missing elemental
metadata limits the claim; mass balance is not proof of elemental balance.
Declare source/sink exchanges for open reservoirs. Total molecular count need
not be conserved by a reaction. Charge-changing mechanisms are outside the
initial scope and must be rejected rather than leaving inconsistent charge.

Heterogeneous uptake additionally needs a selected physical law for surface
area, transport/accommodation, and reaction progress. A cross-phase equation
alone does not supply it. Coordinate with condensation/equilibrium to avoid
counting a gas transfer twice. Fixed-count chemistry must preserve positive
total mass in active particles; complete particle removal needs a later
retirement contract.

## Reviewable implementation increments

1. **Species/schema and compilation.** Resolve bindings and units, validate
   reaction balances and dependencies, and compile one canonical mechanism.
   Reject missing species, invalid layouts, unsupported phase laws, and
   unbalanced closed reactions before any state changes. Demonstrate schema
   round trips and binding invalidation independently of a numerical solver.
2. **Gas-only and particle-only chemistry.** Add the CPU process and explicit
   CUDA adapter for the bounded rate families. Use the Stage 2A assessment to
   select methods suitable for the case's stiffness, with error control,
   positivity, and budget checks. First-order transformations and a small
   coupled reaction system provide independent analytical/reference cases.
3. **Coupled heterogeneous chemistry.** Implement one specified uptake/reaction
   law and shared-reservoir accounting. Test competition between particles and
   reactions, distinct box volumes, depletion, and interaction with condensation.
   Adding one uptake law must not imply general surface chemistry support.
4. **Continuation and scientific example.** Integrate diagnostics and exact
   supported restart, publish a worked mechanism and comparison, and qualify
   CUDA capture only for compatible execution policies. Measure complete
   workload cost; preserve unsupported combinations as explicit rejections.

## CPU/GPU execution and numerical boundaries

- CPU uses staged native updates through `BoxProcess` and the common numerical
  contracts. Custom callbacks must not hide uncheckpointed solver state.
- GPU preparation owns fixed mechanism/scratch arrays and the declared device
  operation. Validate support before upload; no automatic CPU fallback.
  An explicitly selected host-transfer path is a separate documented option.
- Independent per-box solves must still couple all reactions sharing that
  box's gas. Jacobian construction, linear solves, divergence, and memory cost
  require measurement for a stiff GPU solver.
- Use Stage 2A's error and failure policies. Invalid/rejected trials must not
  publish state or output; failures after accepted work retain the declared
  execution boundary rather than promising whole-run rollback.
- Checkpoints include mechanism/schema identity, bindings, configuration,
  simulation time, and persistent solver state required for exact continuation.
  Do not serialize executable callbacks or infer incompatible mechanism changes.
- Prescribed temperature avoids a thermal-feedback claim. Reverse rates need
  explicit laws or a qualified thermodynamic derivation; equilibrium constants
  must not be invented from forward kinetics alone.

[KPP Rosenbrock methods](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html)
are one reference for stiff kinetics and embedded error estimates. Selecting
one for Particula remains conditional on the Stage 2A study and the mechanism's
invariants; stiff stability alone is not a positivity or conservation proof.

## Owning implementation and evidence

- [Gas metadata](../particula/gas/gas_data.py),
  [particle state](../particula/particles/particle_data.py), and a proposed
  chemistry module whose public exports will be decided during schema design.
- [CPU box process](../particula/box_process.py),
  [GPU process adapters](../particula/execution/process_extensions.py), and
  [prepared checkpoint integration](../particula/execution/prepared_checkpoint.py).
- [Custom-process guide](../docs/guide/processes/index.md),
  [timed chemistry example](../docs/studies/examples/custom-processes/timed_box_chemistry.py),
  and [CPU/GPU extension example](../docs/studies/examples/custom-processes/custom_process_backends.py).
- [Extension tests](../particula/execution/tests/process_extensions_test.py),
  new adjacent mechanism tests, and
  [scientific validation records](../docs/studies/validation/index.md).

## Acceptance

- One mechanism configuration drives supported CPU and actual CUDA execution;
  independent references verify rates, conversion factors, trajectories, and
  error estimates rather than relying only on backend agreement.
- Analytical transformations, a stiff reference system, and a bounded
  heterogeneous case establish convergence and appropriate mass/elemental
  budgets. Include zero product species, inactive slots, near depletion,
  multiple reactions competing for a reactant, and non-unit box volumes.
- Unmapped/inert species and particle counts remain unchanged. Particle and
  full-gas species widths may differ without dropping a reservoir from budgets.
- A supported run resumes with the declared continuation guarantee. Incompatible
  mechanisms, layouts, unsupported callbacks, and capture combinations reject
  explicitly before unsupported execution.
- A worked example supplies mechanism data, units, inputs, numerical settings,
  independent comparison, and clear physical limits. No full atmospheric or
  multiphase chemistry claim is inferred from a simple demonstration.

Use the [repository validation reference](../plugins/particula-helper/references/repository-validation.md)
for focused CPU/CUDA, lifecycle, example, and packaging checks when implemented.
