# Stage 5: Inverse modeling and automatic differentiation

The closed-box delivery used `/tmp/particula-stage-05` on
`codex/stage-05-inverse-modeling`, based on upstream `c76cfb5993`.
The executed implementation revision is `9fa5880745`.
[Validation and representation decisions](../docs/studies/validation/condensation/initial-size/index.md)
· [Worked example](../docs/studies/validation/condensation/initial-size/infer_initial_radius.py)

Planning baseline reviewed at `5ec711e600` on 2026-11-19. Entry dependency: a scientifically supported forward case
with validated [numerics](stage-02a-numerical-integration.md), reproducible
[outputs](stage-02b-simulation-workflow.md), and a defined
[observation comparison](stage-03b-scientific-comparisons.md).
GPU-specific claims also require relevant [Stage 4 evidence](stage-04-gpu-qualification.md).

## Execution and delivery

Create a dedicated worktree and `codex/stage-05-inverse-modeling` branch when
starting this plan. Develop and validate there, then commit, push, and create
a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome

Infer supported aerosol initial-state quantities from observations, quantify
what the observations can constrain, and demonstrate when automatic
differentiation improves the inverse workflow.

The existing [initial-state roadmap](../docs/contribute/roadmap.md) holds process
parameters fixed. Preserve that initial target. Parameter inference, if later
requested, is a distinct scope decision rather than an implicit expansion.

## First delivery — recover initial radius from a final size observation

Start with the Stage 2A smooth-growth case. In a closed 0.37 m³ box, one
resolved slot represents a monodisperse population at 10⁹ m⁻³. Hold the
initial vapor at 10⁻⁵ kg/m³, flat-interface equilibrium vapor at zero,
temperature at 298.15 K and pressure at 101325 Pa. Retain the inert gas and
inactive slot. Vary only initial particle radius over 0.8–1.2 µm, converting
it to mass per physical particle with the case density of 1000 kg/m³. Use
the Sutherland viscosity from the adaptive assessment. The proposed observable
is final wet radius at 1.4 s. Reserve vapor concentration and intermediate
radii for independent trajectory checks. [Case inputs](../scripts/assess_native_condensation.py)
· [Independent radius reference](../scripts/assess_native_adaptive.py)
· [Forward validation](../docs/studies/validation/condensation/adaptive/index.md)

A planning calculation with the independent radius equation gave the following
values. In `local_reference`, pass initial radius divided by 1 µm, the fixed
initial vapor, duration 1.4 s and equilibrium zero. Convert the returned mass
to radius at the stated density. The slopes are centered differences with
initial-radius perturbations of 10⁻⁴ µm. Increasing the perturbation to
10⁻³ µm changed each slope by less than 2 × 10⁻⁷.

| Initial radius (µm) | Final radius (µm) | Final vapor (kg/m³) | d(final radius)/d(initial radius) |
| ---: | ---: | ---: | ---: |
| 0.8 | 1.045893 | 7.35229 × 10⁻⁶ | 0.74648 |
| 1.0 | 1.202174 | 6.91114 × 10⁻⁶ | 0.81215 |
| 1.2 | 1.369394 | 6.48165 × 10⁻⁶ | 0.85731 |

The three slopes support trying a scalar inversion. At the center, a synthetic
10 nm standard deviation in measured final radius corresponds locally to about
12 nm in inferred initial radius. Check sensitivity across the interval and
test noisy recovery directly. This observation model applies to a synthetic
monodisperse population. The existing CCN study fits an effective κ from
activation data and illustrates how to retain a held-out observable if the
study later moves to measurements. [CCN retrieval study](../docs/studies/literature/particle-water-uptake/ccn-retrieval/index.md)

## Reviewable work packages

1. **Recover a scalar with the CPU reference.** Fit the initial radius to
   final-radius residuals scaled by the declared synthetic measurement error.
   Scan the interval for monotonicity and ambiguous minima. Compare a bracketed
   scalar search with bounded least squares on the same observations, bounds
   and forward accuracy. Generate withheld truths using the independent radius
   equation. Proposed gates are at most 1 nm of forward radius error and 2 nm
   of noiseless initial-radius error on withheld interior cases. Fix the grid,
   starts, noise seeds and gates before fitting. For noisy replicates report
   bias, spread, interval coverage and the frequency of fits at either bound.
   [SciPy bounded least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
2. **Differentiate the complete discrete trajectory.** On the fixed-step
   isothermal path, record the chain from initial radius to mass, successive
   gas and particle states, final radius and scalar loss. Preserve forward
   values needed by the reverse pass using distinct time levels or verified
   recomputation. Match the mapped species, transport coefficients and fixed
   substeps to the ordinary forward run. Refine the timestep against the
   independent reference. Check separate vapor error and the closed inventory
   (gas concentration plus represented particle mass per box volume).
   Track inactive slots, inert gas, and margins to evaporation or gas limiting.
   Depletion, clipping and adaptive step decisions require separate derivative
   tests before extending the supported interval.
3. **Verify gradients on Warp CPU and actual CUDA.** The existing fp64 test
   differentiates a single raw mass-transfer rate with respect to gas
   concentration. It passed on both backends on this checkout with Warp
   1.11.1. Add full-trajectory centered differences over a perturbation sweep,
   both for final radius and for the loss away from its minimum. Compare AD
   to the same discrete forward calculation first, then check convergence of
   the observable derivative toward the independent radius-equation slope.
   Start with a proposed relative agreement of 10⁻³ for the nonzero slopes
   and declare a dimensioned absolute tolerance for near-zero derivatives.
   Retain the existing clamp and mutation tests. [Raw-rate and boundary tests](../particula/gpu/kernels/tests/condensation_autodiff_test.py)
   · [Condensation kernels](../particula/gpu/kernels/condensation.py)
   · [Warp differentiation guide](https://nvidia.github.io/warp/stable/user_guide/differentiability.html)
4. **Measure the value of gradients.** Repeat held-out recoveries across
   independent boxes on actual CUDA, then move to a size distribution sharing
   a common vapor reservoir within each box. Specify which bins the observation
   resolves and examine the singular values of the noise-scaled sensitivity
   matrix before fitting additional unknowns. Report indistinguishable
   distributions and any regularization. Include a synthetic empty size band
   and measure how much population the inversion assigns to it as observation
   channels broaden. Cai et al. documented false small-particle counts in
   some particle size magnifier inversions. This motivates a resolution test
   for the later distribution study. [Cai et al. (2018)](https://doi.org/10.5194/amt-11-4477-2018)

   Compare gradient-free, finite-difference and AD fits at matched recovery
   accuracy. Count every forward
   and backward evaluation, including failed starts. Measure synchronized
   end-to-end time and peak tape memory versus steps and boxes, with a
   segment-recomputation comparison. A scalar fit alone closes only the first
   milestone. [Initial-state roadmap and exit criteria](../docs/contribute/roadmap.md)
5. **Resolve the broader roadmap.** Record a representation decision for
   composition and mixing state. Before using a deterministic sectional
   coagulation surrogate, compare its predicted observables with resolved
   ensembles over independent seeds and quantify sampling uncertainty.
   Qualify capture, per-box RNG and autodiff together in a dedicated test
   before claiming that combination. Restart files serve forward continuation.
   [Autodiff design constraints](../docs/contribute/architecture/warp/autodiff-limitations.md)
   · [Prepared execution](../particula/execution/prepared.py)

Publish the case, reference, derivative sweep, recovery failures, timed
comparators and software/device revisions with a percent-format Python
example and a [validation record](../docs/studies/validation/index.md). The Stage 4
campaign supplies the timing protocol. Its BAT tables, humidity reset and
captured forward timings need their own gradient qualification if used later.

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
This stage does not depend on implementing the
[later chemistry model](stage-06-chemistry-model.md), but it does depend on the
physics required by its selected inverse problem. Chemistry differentiability
would need its own qualification rather than being implied by either plan.
