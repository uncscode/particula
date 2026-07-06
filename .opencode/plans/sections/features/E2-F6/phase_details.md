# E2-F6 Phase Details

- [x] **E2-F6-P1:** Define NPF-to-droplet numerical cases and fp64 baseline with validation checks
  - Issue: #1208 | Size: S | Status: Shipped
  - Goal: Build deterministic study cases covering NPF clusters, small
    particles, accumulation-mode particles, and cloud-droplet-scale masses.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `docs/Features/Roadmap/index.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, and
    `particula/gpu/tests/mass_precision_cases_test.py`.
  - Delivered: Added deterministic baseline cases for `npf_cluster`,
    `five_to_ten_nm`, `accumulation_mode`, and `cloud_droplet`; recorded the
    current `np.float64` / `wp.float64` baseline policy; and linked the new
    roadmap page from existing roadmap artifacts.
  - Tests: `mass_precision_cases_test.py` now covers deterministic rebuilds,
    canonical shapes, finiteness, nonnegative masses, `ParticleData`
    construction, derived-radius ordering, malformed-input rejection, Warp
    dtype assertions, and exact Warp CPU-device round trips.

- [x] **E2-F6-P2:** Compare absolute fp64 mass storage with fp32 and mixed-precision candidates
  - Issue: #1209 | Size: S | Status: Shipped
  - Goal: Evaluate current absolute `fp64` storage against bounded study-only
    candidates without changing production defaults.
  - Files: `docs/Features/Roadmap/mass-precision-study.md` and
    `particula/gpu/tests/mass_precision_metrics_test.py`.
  - Delivered: Added study-only candidate projection/reconstruction helpers,
    mass/radius tolerance checks for three executed candidates, invalid-input
    coverage, zero-total-mass handling, unsupported-candidate doc-only
    coverage, and explicit CPU/Warp dtype regression checks while leaving
    production dtype defaults unchanged.
  - Tests: `mass_precision_metrics_test.py` now covers candidate mass/radius
    fidelity, invalid candidate ids, zero-total-mass reconstruction behavior,
    unsupported-candidate boundaries, and CPU/Warp `float64` regression checks.

- [x] **E2-F6-P3:** Evaluate conservation small-particle fidelity memory and throughput tradeoffs
  - Issue: #1210 | Size: S | Status: Shipped
  - Goal: Quantify conservation error, small-particle mass/radius fidelity,
    clamping behavior, memory footprint, and GPU/CPU throughput where available.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `.opencode/plans/sections/features/E2-F6/documentation_updates.md`,
    `particula/gpu/tests/mass_precision_metrics_test.py`,
    `particula/gpu/tests/benchmark_test.py`, and
    `particula/gpu/tests/benchmark_helpers_test.py`.
  - Delivered: Added cached P3 reconstruction and CPU-reference
    mass-transfer metrics, mixed-scale smallest-particle thresholds,
    zero-total-mass and zero-volume warning-clean coverage, explicit clamp
    accounting, bounded opt-in projection benchmarks, fast helper tests for the
    benchmark path, and roadmap updates that record thresholds, memory examples,
    and throughput availability.
  - Tests: `mass_precision_metrics_test.py` now covers conservation-sensitive
    candidate deltas, mixed-scale mass/radius thresholds, edge-path stability,
    and clamp metrics; `benchmark_helpers_test.py` covers skip gates, helper
    behavior, and bounded benchmark result recording; `benchmark_test.py`
    retains the optional `--benchmark` throughput entry point.

- [x] **E2-F6-P4:** Publish precision recommendation report with validation evidence
  - Issue: #1211 | Size: XS | Status: Shipped
  - Goal: Publish the final recommendation artifact for downstream dtype/schema
    work without changing production runtime defaults.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/index.md`, and
    `docs/Features/particle-data-migration.md`.
  - Delivered: Promoted the roadmap study page into the canonical Mass
    Precision Recommendation Report, published the accepted policy to keep
    absolute per-species `np.float64` / `wp.float64` particle mass storage,
    separated measured evidence from rejected/unsupported/deferred paths,
    recorded downstream constraints for future dtype/schema proposals, and
    updated the GPU roadmap, roadmap index, and migration guide to cite the
    report as the canonical reference.
  - Tests: Publication guidance now names focused reruns of
    `mass_precision_cases_test.py`, `mass_precision_metrics_test.py`,
    `benchmark_helpers_test.py`, and `pytest -Werror`
    `mass_precision_metrics_test.py`, plus direct Markdown-link checks and
    optional `mkdocs build --strict` / benchmark validation when available.

## Phase Ordering Notes

- P1 defines the reproducible cases that every later comparison reuses.
- P2 should evaluate candidate representations only against the P1 baseline so
  dtype conclusions are comparable across cases.
- P3 should follow P2 because conservation and throughput tradeoffs only matter
  after the candidate precision set is fixed.
- P4 is the acceptance gate for downstream dtype work and should publish only the
  evidence produced by P1 through P3.
