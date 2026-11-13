# Current roadmap

## Differentiability and initial-state optimization (Epic I)

**Status: pending, after the v0.3.0 container migration.** No differentiable
process path or release approval is implied by this roadmap. Open release gates
remain in [v0.3.0 validation](validation/releases/v0.3.0.md); missing CUDA evidence
remains in [graph-capture qualification](validation/gpu/graph-capture-qualification.md).

Start from the [native data model](../guide/data-model/index.md) and the
[autodiff design constraints](architecture/warp/autodiff-limitations.md).

**The optimization targets are initial state, not process parameters.** Physical
coagulation and condensation parameters (accommodation coefficient, diffusivity,
and similar) are prescribed and held fixed. The inverse problem fits the initial
aerosol state to a measured final state. Examples: initial versus final size
distribution, initial versus final hygroscopicity, or initial versus final
mixing state. Coagulation and condensation are the size-dependent forward
operators that connect initial to final state; their parameters are not tuned.

The consequence for differentiability is important: because the loss is on the
final state and the unknowns are the initial state, gradients must flow
**through** each process operator with respect to the state it acts on (per-bin
or per-particle masses, concentrations, and composition), not with respect to
its parameters. Prescribed parameters mean no parameter adjoints are needed, but
the operators themselves must still be differentiable in state.

Planned features:

1. Differentiable-friendly variants of `apply_mass_transfer_kernel` and
   `apply_coagulation_kernel` compatible with `wp.Tape` (no gradient-breaking
   in-place mutation or control flow on the optimization path).
2. End-to-end gradient from a final-state loss back to the initial state
   through a multi-step deterministic condensation loop, with recorded tape
   memory usage.
3. Tape memory budgeting and gradient-checkpointing evaluation for long
   differentiable loops, feeding the resident memory-budget model.
4. Global-optimization example that recovers an initial size distribution
   from a synthetic final distribution across multiple boxes, with process
   parameters held fixed.
5. Decision record for differentiable coagulation (deterministic
   binned/sectional operator versus relaxed particle-resolved formulation).
6. Decision record for the mixing-state and hygroscopicity state
   representation (composition-resolved sectional versus particle-resolved
   with a differentiable surrogate).
7. Combined validation of autodiff, graph capture, and per-box RNG streams
   operating together.
8. Loss-function definitions on state (size distribution, hygroscopicity,
   mixing-state metrics), each with a confirmed differentiable path back to
   the initial state.

Design constraints and open decisions:

- **Author optimization-path kernels against Warp's `wp.Tape` requirements.**
  In-place mutation and some control-flow patterns can break or zero gradients.
  The current `apply_mass_transfer_kernel` and `apply_coagulation_kernel` use
  in-place updates and would need differentiable-friendly variants for the
  optimization path.
- **Start with deterministic condensation.** Isothermal and latent-heat
  condensation are natural first differentiable targets because mass transfer is
  deterministic. Prove an end-to-end gradient and optimization loop here before
  extending to other processes.
- **Budget tape memory and plan checkpointing.** Backpropagating through a
  multi-step loop stores intermediates per step, so tape memory scales with
  timesteps times `n_boxes × n_particles × n_species` fp64 state and can
  dominate the memory budget. Evaluate gradient checkpointing (recompute
  segments of the forward pass during the backward pass) before committing to
  long differentiable loops, and include tape storage in the resident
  memory-budget model.
- **Stochastic coagulation blocks state gradients as currently implemented.** The
  GPU Brownian coagulation kernel uses stochastic acceptance-rejection and
  discrete pair selection, so gradients do not flow through it back to the
  initial state, even though its parameters are fixed. How coagulation
  participates in the initial-state inversion is an **open decision**. Candidate
  approaches:
  - A deterministic binned or sectional coagulation operator (Smoluchowski) for
    the gradient path, distinct from the particle-resolved stochastic forward
    model, validated to agree in the mean.
  - A relaxed or differentiable particle-resolved formulation if per-particle
    fidelity is required.
- **Mixing state and hygroscopicity favor composition-resolved state.** Fitting
  initial versus final mixing state or hygroscopicity requires tracking
  per-species composition, which coagulation redistributes when particles of
  different composition merge. Decide the representation that is both
  physically adequate and differentiable, for example composition-resolved
  sectional bins (size by composition) versus particle-resolved with a
  differentiable surrogate. The limited direct-kernel ideal/kappa water-
  activity support shipped in [direct condensation](../guide/gpu/direct/condensation.md)
  is not yet a differentiable or high-level runtime path; a suitable
  differentiable hygroscopicity implementation remains required.
- **Autodiff, graph capture, and RNG state interact.** Differentiable execution,
  graph-captured timestep loops, and per-box RNG streams must coexist for
  gradient-based fitting. Validate them together, not only in isolation.
- **Multi-box supports optimization directly.** Evaluating many boxes per
  optimization step provides the batch of conditions used to match experiments
  or observations.
- Define the loss functions on state, for example distance between initial and
  final size distributions, hygroscopicity distributions, or mixing-state
  metrics, and confirm each has a differentiable path back to the initial state.

**Exit bar:** A gradient-based optimizer recovers a known initial state from
synthetic multi-box final-state data within recorded tolerance, using GPU
autodiff, without fitting any process parameters, and the differentiable
coagulation and state-representation decisions are recorded.

