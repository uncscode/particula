# Scientific correctness reference baseline

Status: validation in progress on the Stage 1 branch based on `894f60350`.
Measurements below use Python 3.12, float64 CPU calculations; CUDA checks use
an NVIDIA RTX 5060 with driver 615.78.08. They verify implementation and
numerical limits, not agreement with independent laboratory measurements.

## Findings and decisions

| Finding | Classification and disposition |
| --- | --- |
| Mixture density inconsistent with additive-volume radius | Confirmed defect. Equal material volumes at 1000 and 2000 kg/m³ gave 1666.67 instead of 1500 kg/m³. CPU density now uses total mass/total material volume, matching existing GPU physics. Single-material results are unchanged. |
| Negative coagulation births in sparse tails | Confirmed interpolation defect. Cubic interpolation of positive collision densities produced negative values; decreasing dt from 2000 to 0.002 s did not resolve negative updates. Both PMF-converted and PDF gain quadratures now constrain interpolated births to nonnegative values. Time-step rejection remains intact. |
| Subfreezing water vapor pressure | Phase-model boundary. Buck selects ice below 273.15 K and liquid above it; it is not a supercooled-liquid or freezing model. The condensation guide now states this boundary. |
| Inventory-limited condensation | Numerical-resolution limit. Positive state and species closure do not establish trajectory accuracy. The independent growth reference below measures time refinement. |
| Fixed-composition binned coagulation | Representation limitation. Variable material fractions across sizes retain individual-species defects even as total mass converges. Use resolved particles for the selected coupled species-budget claim. |
| First tutorial's 100 kg/m³ vapor | Already explicitly labeled exaggerated, not calibrated. It is not a research reference; no atmospheric accuracy claim is made for it. |
| Import logging | Confirmed reproducibility defect. Import previously created package-local files and changed application logging. Import now adds a NullHandler only; explicit legacy setup remains opt-in. |
| Version alignment | Source is 0.3.1; stale AGENTS wording corrected. No release or version bump is implied. |
| CPU installation | Retain the declared Warp dependency. CPU execution needs no CUDA device, but this delivery does not offer a Warp-free package. |

## Independent reference definitions

[Independent reference definitions in the study](../process-references.md#independent-reference-definitions)

## Reproduction and remaining gates

Run the reference modules with the selected checkout's Python:

```bash
python -m pytest particula/integration_tests/condensation_reference_test.py \
  particula/integration_tests/coagulation_reference_test.py \
  particula/integration_tests/sampling_reference_test.py \
  --no-cov -q -Werror -o junit_family=legacy --junitxml=reference-results.xml
```

Ten cases passed. JUnit properties retain convergence errors and durations.
The mixture regression independently checks total mass equals density times
spherical volume, including an empty slot. Existing mapped-species integration
checks cover inert lanes, resolved representative volumes and PDF quadrature.

## Final validation

The unfiltered package run with warnings treated as errors passed 10583 tests,
with 20 skips and one expected failure. Coverage was 91.34%, above the unchanged
80% gate. The source/example/tooling commands are those in the repository
validation reference, using the existing Python 3.12 environment. The full run
includes available CUDA tests; skipped capabilities remain outside the claim.
Explicit concrete wall-loss checks were also included in the focused CPU run.

| Gate | Observed result |
| --- | --- |
| Focused CPU density/consumer tests | 1611 passed |
| Warp properties and wall-loss checks | 103 passed with RTX 5060 access |
| Independent growth/coagulation/sampling references | 10 passed |
| Installed wheel CPU release selection | 5086 passed, 25 skipped, 1130 deselected |
| Example regressions | 241 passed, one skipped |
| Tooling/publication tests | 319 passed |
| Ruff and formatting | Passed |
| Mypy | No issues in 645 source files |
| Documentation build | 533 generated pages; rendered structural check passed |

The installed runner deliberately excludes GPU-heavy selections and is not a
replacement for the full source suite. Its isolated environment validates
installed package identity and dependency consistency. Wheel/sdist bundle
checks also verified source hashes, separate installs and offline rebuilding;
those artifacts are verification outputs, not an approved release.

Affected CPU coagulation, wall-loss and organic-partitioning examples were
executed in full. Refreshed output bundles retain their input/revision metadata;
changed plots were visually inspected. Import-log removal changed three
wall-loss tutorial figure filenames; manifests and files were refreshed together.

Publication previously tested documentation bundles on a release event without
rerunning source suites for that event. The release workflow now requires source,
example and tooling tests before build/upload, retaining exact bundle validation.
No release was published by this work.

Independent experimental validation, stiff/adaptive solver design, unrestricted
custom-kernel sampling and new thermodynamics remain outside this baseline.
Their absence does not support a claim of accuracy beyond the cases above.

## Scalable native coagulation: issue 1661

<a id="cuda-population-boundaries-and-resource-scaling"></a>
[Scalable native coagulation: issue 1661 in the study](../../coagulation/native-sampling/convergence.md#scalable-native-coagulation-issue-1661)
