# Stage 1: Scientific correctness and verification

Status: proposed. Entry dependency: none.

## Execution and delivery

Create a dedicated worktree and `codex/stage-01-scientific-correctness` branch
when starting this plan. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome

Establish a trustworthy reference baseline for the existing physics and resolve
confirmed defects that affect the selected research workflows. Every scientific
claim should identify its tested regime, numerical settings, and limitations.
Verification continues as subsequent stages change the implementation.

## Starting evidence and boundaries

The review discussion identified useful audit leads, not a fully reproduced
defect inventory. Current source inspection confirms that particle radii use
additive material volumes while effective density uses a mass-weighted
arithmetic mean. It also confirms inventory-limited explicit condensation,
fixed-composition binned coagulation, and a first tutorial with exaggerated
vapor concentration. Determine their physical consequences in specific cases.

Do not inherit categorical review claims as facts. CPU batched runnables,
exact exponential dilution, analytical checks, and qualified examples already
exist. Existing evidence must be assessed for its actual scope and revision.

## Work packages

1. **Reproduce and classify findings.** Distinguish physical defects, numerical
   resolution limits, unsupported configurations, documentation errors, and
   convenience issues. Prioritize effects on predicted observables. Investigate
   density/radius consistency, liquid versus ice vapor-pressure assumptions,
   condensation timestep sensitivity, coagulation sampling limits, and species
   accounting. Record the result even when a review claim is disproved.
2. **Build independent reference cases.** Start with exact dilution, simple
   condensation limits, and an analytically solvable coagulation case where its
   assumptions match the chosen representation. Add relevant activity/Kelvin
   limits and source/sink balances. Derive or cite the oracle independently of
   production code and specify tolerances before interpreting results.
3. **Correct confirmed errors.** Make bounded changes with a regression that
   exposes the original error. Inspect dependent CPU and GPU paths, update
   scientific assumptions, and refresh affected example outputs.
4. **Characterize numerical uncertainty.** Vary timestep, size resolution,
   represented particle count, and stochastic sample count separately. Check
   coupled process order as well as isolated processes. Report each observable
   and species budget rather than only an aggregate mass residual.
5. **Repair research reproducibility gaps.** Audit installed-package identity,
   documentation/version alignment, import side effects, dependency scope, test
   collection, and publication gates. Confirm review claims against current
   packaging before changing them. Make CPU-only installation policy an
   explicit decision rather than assuming optional GPU dependencies already exist.

## Owning code and evidence

- [Particle properties](../particula/particles/particle_data.py),
  [gas properties](../particula/gas/properties/), and their adjacent tests.
- [Dynamics](../particula/dynamics/), including condensation, coagulation,
  dilution, wall loss, nucleation, and their independent tests.
- [GPU kernels](../particula/gpu/kernels/) and
  [integration tests](../particula/integration_tests/) for shared defects.
- [Package configuration](../pyproject.toml),
  [CI workflows](../.github/workflows/), and
  [release evidence](../docs/contribute/validation/releases/v0.3.0.md).

## Acceptance and validation

- A small reference set runs reproducibly and distinguishes conservation from
  numerical accuracy. Resolved counts, PMF number densities, and PDF quadrature
  use the correct representative volume and units.
- Confirmed blockers for the selected case are corrected, or the unsupported
  regime is explicitly rejected and excluded from the scientific claim.
- Each fix has an independent regression and relevant CPU/GPU coverage.
  Stochastic evidence uses ensembles and uncertainty bands rather than exact
  cross-device random trajectories.
- The case has documented convergence results, inputs, revision, tolerances,
  and remaining limitations in the maintained validation records.
- Relevant tests actually collect, including concrete wall-loss tests when
  affected. Package/install changes receive installed-artifact checks.

Use the [repository validation reference](../plugins/particula-helper/references/repository-validation.md)
to choose focused checks and any required broader suite. Do not weaken numerical
or coverage thresholds to close a finding.

## Handoff

The initial reference set enables both Stage 2 plans. Independent experiments
belong to [Stage 3B](stage-03b-scientific-comparisons.md); solver design belongs
to [Stage 2A](stage-02a-numerical-integration.md). Stage 1 does not require
implementing every missing physical process before these plans can proceed.
