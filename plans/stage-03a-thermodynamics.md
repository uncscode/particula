# Stage 3A: Thermodynamics and additional physics

Status: implemented on the Stage 3A branch. The [validation record](../docs/studies/validation/material-properties/binary-bat.md) lists the supported homogeneous profile, CPU and CUDA evidence, and limits of the lower-O:C comparison.
Stages 1, 2A and 2B are merged; reuse their scientific references, numerical
contracts and recording/restart infrastructure. Their completion does not
qualify the new thermodynamic coupling.

## Execution and delivery

Create a dedicated worktree and `codex/stage-03a-thermodynamics` branch when
starting implementation. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## First milestone

Add BAT alongside κ as a particle activity option for kinetic condensation and evaporation. Start with one internally homogeneous water–organic liquid in each active particle. Compute both activities from current particle masses at every rate evaluation. Existing transport, Kelvin correction, gas inventories and external vapor forcing determine the subsequent trajectory. All simulated mass transfer remains kinetic.

Use the miscible case in Fig. 9 of Gorkowski et al. as a reference configuration (generic organic, 200 g/mol, O:C 0.5). The caption specifies 298.15 K and the archived Kelvin calculation uses 293 K. The [binary reference checks](../docs/studies/validation/material-properties/binary-bat.md) record the descriptors and the archived phase flag. Reproduce direct activities with the published implementation and record its revision. This is a model verification case. Stage 3B owns the choice of independent measurements. [Theory and primary references](../docs/theory/thermodynamics/activity-theory.md)

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

## Implementation and acceptance

1. Extend the CPU condensation configuration to select BAT and map water and organic activities to the intended species. Preserve the scalar organic strategy for existing callers. Test reordered gas species, inert lanes, inactive slots, the pure limits and the documented units. Reject unsupported mixtures before changing state.
2. Generate and qualify the resident table as described above. As initial engineering targets, require absolute error at most 0.0001 for each activity on held-out compositions and relative error at most 0.1% in final wet radius against a refined direct-BAT kinetic run. Check near-zero flux with an absolute pressure residual, and tighten the grid where errors change the growth direction outside that residual. Record separate timestep, grid and per-species budget checks using the existing conservation tolerances. These targets measure approximation error relative to BAT.
3. Publish a matched CPU and actual CUDA uptake/evaporation example. Record gas and particle water separately, test a change in ambient humidity with finite-time relaxation and qualify the advertised solver, capture and restart profiles.
4. Keep standalone fixed-water-activity and VBS partitioning calculations functional. Clean up input units, naming, return values and examples while preserving existing callers and numerical results. This work does not require applying an equilibrium result to simulation state.
5. Evaluate a phase-separation extension with the published Fig. 9 lower-O:C cases. Include it if a binary coexistence lookup reproduces both phase endpoints, equal activities and lever-rule amounts under an explicit rapid internal equilibration assumption using the existing state. Otherwise record the unsupported interval and deliver the homogeneous case. Internal diffusion and evolving morphology require a separate model.

AIOMFAC remains a separate follow-on. The evidence for this milestone belongs in a runnable example and the repository validation records. Follow the [validation guidance](../plugins/particula-helper/references/repository-validation.md).
