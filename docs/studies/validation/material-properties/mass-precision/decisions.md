# Mass Precision Recommendation Report

This page is the final `mass-precision study` acceptance artifact for particle mass precision
policy. It records the shipped baseline-candidate evaluation evidence, publishes the current
recommendation for downstream dtype/schema work, and keeps measured evidence
separate from follow-up constraints.

Production particle storage remains absolute per-species `np.float64` on CPU
and `wp.float64` on GPU. This report does not ship a runtime dtype or schema
change.

## Final recommendation

- Keep absolute per-species `np.float64` particle masses on CPU.
- Keep absolute per-species `wp.float64` particle masses on GPU/Warp mirrors.
- Treat this baseline as the accepted production policy until a later proposal
  proves that an alternative representation preserves the documented fidelity,
  conservation-sensitive behavior, and workflow constraints in this report.

## Recommendation boundaries

- This report summarizes only evidence already executed in the shipped baseline-candidate evaluation
  test surface.
- The recommendation is an approval gate for downstream dtype/schema proposals,
  not approval to change production defaults now.
- Optional benchmark evidence remains supplemental and opt-in; it does not by
  itself authorize a production migration.

## Deterministic cases

[Deterministic cases in the study](../mass-precision.md#deterministic-cases)

## Density assumptions

[Density assumptions in the study](../mass-precision.md#density-assumptions)

## Volume-fraction assumptions

[Volume-fraction assumptions in the study](../mass-precision.md#volume-fraction-assumptions)

## Construction rule

[Construction rule in the study](../mass-precision.md#construction-rule)

## Shipped evidence summary

<a id="baseline-coverage"></a>
<a id="precision-comparison-executed-candidates"></a>
<a id="candidate-evaluation-thresholds-and-conservation-sensitive-checks"></a>
<a id="candidate-evaluation-mixed-scale-fidelity-finding"></a>
<a id="candidate-evaluation-zero-mass-and-zero-volume-edge-handling"></a>
<a id="candidate-evaluation-clamp-accounting"></a>
<a id="candidate-evaluation-memory-footprint-examples"></a>
<a id="candidate-evaluation-throughput-evidence-availability"></a>
[Shipped evidence summary in the study](../mass-precision.md#shipped-evidence-summary)

## Executed but not recommended candidates

- `fp32_absolute_mass`
- `mixed_precision_mass_plus_density`
- `fp32_total_mass_fp32_mass_fraction`

These candidates were executed to measure reconstruction fidelity,
conservation-sensitive deltas, mixed-scale behavior, clamp accounting, and
storage tradeoffs. They are not recommended for production defaults in this
phase because the shipped evidence is used to approve the current `fp64`
baseline, not to authorize a runtime migration.

## Unsupported candidates

Candidates that need new runtime schema fields, extra production metadata, or
public API expansion remain unsupported in this phase. They should be recorded
as documentation-only ideas rather than added to executable runtime code or the
focused test matrix.

## Deferred investigation areas

- Broader alternative mass representations such as log-mass or new reference-
  mass schemas.
- Production schema expansion that introduces new stored helper fields or new
  public migration obligations.
- Wider throughput campaigns beyond the focused opt-in benchmark surface.

These follow-up areas are out of scope for this issue.

## Downstream constraints for future dtype/schema proposals

Any future proposal that changes production defaults must, at minimum:

- preserve the deterministic baseline case coverage from NPF to droplet scale,
- satisfy the shipped candidate evaluation reconstruction and mixed-scale thresholds with exact
  candidate ids and warning-clean validation,
- demonstrate acceptable conservation-sensitive mass-transfer deltas,
- account for clamp behavior explicitly rather than hiding it inside aggregate
  error metrics,
- document storage/memory tradeoffs truthfully, and
- update downstream roadmap and migration guidance in the same change so the
  canonical policy reference stays consistent.

## Reproducibility

[Reproducibility in the study](../mass-precision.md#reproducibility)

## Documentation cross-check checklist

- Candidate names in this page match the executable candidate ids.
- Threshold values match the fast metric assertions.
- Memory examples use the documented formulas and concrete shapes.
- Reproduction commands cover both fast checks and optional benchmark evidence.

## Publication-readiness validation

Rerun the focused commands below before treating this report as the backing
reference for downstream dtype/schema work:

- `pytest particula/gpu/tests/mass_precision_cases_test.py -q`
- `pytest particula/gpu/tests/mass_precision_metrics_test.py -q`
- `pytest particula/gpu/tests/benchmark_helpers_test.py -q`
- `pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q`

Optional throughput evidence remains opt-in only:

- `pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s`

Check every Markdown link and anchor referenced by this report update directly
and run `npm --prefix website run build` when the docs toolchain is available.

## Reproduction commands

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_helpers_test.py -q
pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s
```
