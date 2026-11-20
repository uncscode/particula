# Stage 6: Configurable chemistry on CPU and GPU

The Stage 6 delivery is implemented and qualified on the branch in
[PR #1687](https://github.com/uncscode/particula/pull/1687). The milestones
below refer to implemented behavior and its independent reference.

| Milestone | Delivered behavior | Evidence |
| --- | --- | --- |
| Species and mechanisms | Load Python, JSON and YAML. Bind gas names and explicit material columns, validate units and closed budgets, and compile signed reaction entries for resident execution. | [Gas schema](../docs/guide/processes/gas-chemistry.md) · [Resolved schema](../docs/guide/processes/resolved-chemistry.md) |
| Gas reactions and interchange | Run RK23 and stiff ROS3 on NumPy and CUDA. Import the closed NCAR 1.3.0 Arrhenius subset. | [Analytical trajectories, source rates and stiff references](../docs/contribute/validation/numerics/gas-chemistry.md) |
| Bulk particle reactions | Integrate molar mass action using the changing material volume of each resolved particle. | [Equal- and unequal-density volume integrals](../docs/contribute/validation/numerics/resolved-chemistry.md#bulk-material-conversion) |
| Coupled uptake | Share each gas reservoir among particles and pathways using a prescribed collision probability. Preserve represented counts and track both reservoirs through condensation and near depletion. | [Independent radial growth and inventory checks](../docs/contribute/validation/numerics/resolved-chemistry.md#competition-and-near-depletion) |
| Continuation, cost and example | Resume exact same-backend histories, capture a compatible resolved CUDA sequence, measure complete workloads and publish executed cases. | [Work and continuation](../docs/contribute/validation/numerics/resolved-chemistry.md) · [Uptake and aging example](../docs/examples/processes/customization/resolved_reactive_uptake.py) |

Dependencies: [Stage 1 accounting and verification](stage-01-scientific-correctness.md),
[Stage 2A solver assessment](stage-02a-numerical-integration.md),
[Stage 2B recording/restart](stage-02b-simulation-workflow.md), and the
[Stage 3A thermodynamics](stage-03a-thermodynamics.md) needed by the chosen case.
Apply [Stage 4](stage-04-gpu-qualification.md) to GPU qualification. Stage 5
autodiff is not a prerequisite, and chemistry gradients are not an initial goal.

## Execution and delivery

The Stage 6 worktree is `/tmp/particula-stage-06` on branch
`codex/stage-06-chemistry-model`, based on `ebabce042e`. Follow the shared
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
points for the gas and resolved chemistry implementations.

## First delivery design and acceptance criteria

Use the existing two-box example to specify A → B at 0.3 s⁻¹. A and B each
have molar mass 0.1 kg/mol. The boxes have volumes 1 and 2 m³, initial A
concentrations of 2 × 10⁻⁹ and 5 × 10⁻⁹ kg/m³ respectively, zero B, and an inert gas
at 10⁻⁶ kg/m³. The example implements this reaction in a custom CPU callback
using SciPy. Start the configurable mechanism with the reaction alone. For each
box check A(t) = A(0) exp(−kt), B(t) = B(0) + A(0) − A(t), the inert species,
and total A+B inventory in kg. Then restore the example's 4 s pulse and
dilution, including the external mass ledger, and compare all 41 recorded
times with its analytical solution. Retain its trajectory tolerances
(`rtol=2e-9`, `atol=1e-20` kg/m³) and inventory tolerances
(`rtol=1e-12`, `atol=1e-22` kg). [Timed chemistry example](../docs/examples/processes/customization/timed_box_chemistry.py)

| Starting point | Gas milestone acceptance criterion |
| --- | --- |
| CPU `BoxProcess` stages changes and publishes declared writes after all boxes succeed. | Compile phase-qualified names against the gas layout, run the same mechanism after an explicit rebind to reordered columns, and reject stale bindings before mutation. [Box process](../particula/box_process.py) |
| `ODEProblem` and `AdaptiveSolver` accept continuous rates and admissibility checks. CPU ROS2/ROS3 also require a Jacobian and an explicit time derivative. | Reuse these contracts for a numerical CPU implementation. Compare to the exponential trajectory before selecting a solver default. Use the dimensionless Robertson case for a separate stiffness test. [Solver](../particula/execution/ode.py) · [Assessment cases](../scripts/assess_numerical_methods.py) |
| The custom GPU example runs prescribed warming through a prepared adapter. Resident ROS3 device kernels implement condensation-specific algebra. | Declare the mechanism's immutable configuration, supported device operation and any persistent state through the adapter contract. Compare the new chemistry path on actual CUDA with the analytical time series, then check same-device restart. [Adapter contract](../particula/execution/process_extensions.py) · [Warming example](../docs/examples/gpu/prepared/custom_process_backends.py) · [Condensation solver support](../docs/guide/processes/numerical-integration.md) |

The gas milestone passed binding, analytical trajectory and continuation
checks on NumPy and resident CUDA. The [qualification record](../docs/contribute/validation/numerics/gas-chemistry.md) gives the measured errors.
The resolved milestones below extend this qualification to changing material
volume and a shared gas reservoir.

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

Python objects, JSON and YAML share one versioned native schema. Compile supported mechanisms into
species indices, sparse stoichiometry, orders, rate identifiers, and parameters.
Keep solver choice replaceable. Arbitrary Python rate callbacks remain CPU-only
unless an explicit supported GPU implementation is supplied.

### Import an external gas mechanism

The converter translates **NCAR Mechanism Configuration
1.3.0** to native Particula v1 JSON or YAML. The output goes through
`ReactionMechanism.from_dict`, `save`, and the existing gas binding. Users can
inspect it and run it on either backend without the NCAR parser or a compiler.
The source fixture is pinned at NCAR revision `2069f9c97f`.
[Pinned example](https://github.com/NCAR/MechanismConfiguration/blob/2069f9c97f08399b34ca3dcb104d3a5bf596ce40/examples/v1/full_configuration.yaml)
· [Source rate law and defaults](https://github.com/NCAR/MechanismConfiguration/blob/2069f9c97f08399b34ca3dcb104d3a5bf596ce40/docs/source/v1/reactions/arrhenius.rst)

1. **Define the supported input.** Use a minimal, single-gas-phase fixture in
   both JSON and YAML, with ordinary evolving species and closed, irreversible
   `ARRHENIUS` reactions. Map species names to existing gas columns, checking
   any supplied molecular weights against simulation metadata. Start with
   unit reactant coefficients and positive integer product stoichiometry.
   Treat the full upstream example as a rejection fixture. It includes fixed
   reservoirs, nonzero pressure dependence and other reaction types.
2. **Translate and validate before writing.** Keep the reader in the chemistry
   package, separate from the solver. Map NCAR's temperature exponent and
   reference temperature directly. Convert activation energy in joules per
   molecule to J/mol by multiplying by Avogadro's constant. For the alternative
   temperature parameter `C` in kelvin, use molar activation energy `−R C`.
   Apply the source defaults and require compatible native parameter values.
   Units of the prefactor are s⁻¹ for one reactant and m³/(mol s) for two.
   Reject both activation parameters together, a nonzero pressure coefficient,
   fixed reservoirs, unsupported versions and unknown scientific fields. An
   error identifies the source field and reaction or species and leaves the
   requested output untouched.
3. **Verify conversion independently.** Hand-author native counterparts for
   A → B and A + B → C. Evaluate both source equations directly at 298 and
   318 K with nonzero activation and exponent. Require relative rate agreement
   within 10⁻¹². Bind the imported mechanisms to reordered gas columns and
   compare NumPy and actual CUDA to the analytical exponential and equal-molar
   bimolecular solutions using the existing gas qualification tolerances. Check
   every species, inert gas and total kg per box. Verify each unsupported
   fixture fails before export. Publish one source file, its generated native
   file and the conversion command alongside the worked example.

The converter targets the gas mechanism. Resolved chemistry uses the native
version 2 mechanism and explicit particle bindings.

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

## Delivery criteria by increment

1. **Species/schema and compilation.** Resolve bindings and units, validate
   reaction balances and dependencies, and compile one canonical mechanism.
   Reject missing species, invalid layouts, unsupported phase laws, and
   unbalanced closed reactions before any state changes. Demonstrate schema
   round trips and binding invalidation independently of a numerical solver.
2. **Gas-only, then particle-only chemistry.** Complete the gas delivery above
   before extending the CPU process and CUDA adapter to particle material lanes.
   Use the Stage 2A assessment to select methods suitable for the case's stiffness, with error control,
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
are one reference for stiff kinetics and embedded error estimates. Stage 2A
already assessed these methods. Select a chemistry method by measuring
its trajectory errors, admissibility, conservation and cost for the compiled
mechanism. [Stage 2A assessment](../docs/contribute/validation/numerics/adaptive-methods.md)

## Owning implementation and evidence

- [Gas metadata](../particula/gas/gas_data.py),
  [particle state](../particula/particles/particle_data.py), and the
  [native chemistry package](../particula/dynamics/chemistry/).
- [CPU box process](../particula/box_process.py),
  [GPU process adapters](../particula/execution/process_extensions.py), and
  [prepared checkpoint integration](../particula/execution/prepared_checkpoint.py).
- [Custom-process guide](../docs/guide/processes/index.md),
  [timed chemistry example](../docs/examples/processes/customization/timed_box_chemistry.py),
  and [CPU/GPU extension example](../docs/examples/gpu/prepared/custom_process_backends.py).
- [Extension tests](../particula/execution/tests/process_extensions_test.py),
  new adjacent mechanism tests, and
  [scientific validation records](../docs/contribute/validation/).

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
for focused CPU/CUDA, lifecycle, example, and packaging checks.
