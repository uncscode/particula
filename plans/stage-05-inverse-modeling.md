# Stage 5: Inverse modeling and automatic differentiation

Status: proposed. Entry dependency: a scientifically supported forward case
with validated [numerics](stage-02a-numerical-integration.md), reproducible
[outputs](stage-02b-simulation-workflow.md), and a defined
[observation comparison](stage-03b-scientific-comparisons.md).
GPU-specific claims also require relevant [Stage 4 evidence](stage-04-gpu-qualification.md).

## Outcome

Infer supported aerosol initial-state quantities from observations, quantify
what the observations can constrain, and demonstrate when automatic
differentiation improves the inverse workflow.

The existing [initial-state roadmap](../docs/contribute/roadmap.md) holds process
parameters fixed. Preserve that initial target. Parameter inference, if later
requested, is a distinct scope decision rather than an implicit expansion.

## Work packages

1. **Define the inverse problem.** Select initial size distribution, composition,
   mixing state, or another explicitly supported initial-state quantity.
   Specify observations, noise, uncertainty, constraints, regularization, and
   losses. Test identifiability before promising unique recovery.
2. **Establish inference baselines.** Use a manageable synthetic case with known
   initial conditions, plus a suitable search/optimization baseline. Numerical
   error and stochastic noise must be controlled relative to the inference
   objective. Gradient-free fitting is a comparison, not an assumption that all
   high-dimensional problems can be solved efficiently without derivatives.
3. **Qualify deterministic gradients.** Start with a validated condensation
   workflow. Verify gradients through multiple steps against independent
   finite-difference or analytical checks over a stable perturbation range.
   Characterize depletion limits, in-place state updates, and adaptive stepping;
   raw rate derivatives do not establish a differentiable simulation.
4. **Resolve coagulation and state representation.** Compare a deterministic
   composition-resolved operator, an appropriate surrogate, or another justified
   approach when stochastic coagulation is needed. Demonstrate forward
   adequacy and gradient meaning; do not silently replace the physical model
   solely to obtain nonzero gradients.
5. **Measure inference cost.** Record optimization accuracy, evaluations,
   convergence failures, wall time, and memory. Evaluate tape storage and
   recomputation. Distinguish differentiation checkpoints from restart files.
   Validate combinations with CUDA capture and RNG only when actually supported.
6. **Demonstrate recoverability and limits.** Test withheld synthetic cases and
   noise levels before an experimental inference example. Report ambiguous
   solutions and uncertainty as well as successful fits.

## Owning code and evidence

- [Dynamics](../particula/dynamics/), [GPU kernels](../particula/gpu/kernels/),
  and [prepared execution](../particula/execution/prepared.py).
- Existing [autodiff design notes](../docs/contribute/architecture/warp/autodiff-limitations.md)
  and direct condensation derivative tests. Recheck installed Warp behavior
  against current primary documentation before choosing implementation details.
- [Scientific examples](../docs/examples/), [example tests](../examples_tests/),
  and [validation records](../docs/contribute/validation/).

## Acceptance and validation

- A declared, identifiable synthetic initial-state problem is recovered within
  predeclared tolerances, with process parameters kept fixed.
- Gradients are checked for the complete supported forward path and compared
  with an independent estimate. Boundary/discrete cases are excluded explicitly
  or handled with a validated method, rather than hidden by loose tolerances.
- Forward accuracy and conservation remain valid on the optimization path;
  lowering fidelity to improve fitting speed changes the claim and is measured.
- Baseline comparisons demonstrate the cost/accuracy tradeoff of autodiff.
  Recovery uncertainty and non-identifiability are reported where present.
- GPU/autodiff/capture/restart combinations have their own actual evidence;
  passing any one of them alone does not qualify the combination.

Use the [validation reference](../plugins/particula-helper/references/repository-validation.md).
This stage does not depend on implementing the entire future chemistry catalog,
but it does depend on the physics required by its selected inverse problem.
