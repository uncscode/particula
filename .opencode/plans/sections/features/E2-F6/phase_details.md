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

- [ ] **E2-F6-P2:** Compare absolute fp64 mass storage with fp32 and mixed-precision candidates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Evaluate current absolute `fp64` storage against `fp32`, mixed
    precision, and representation alternatives without changing production
    defaults.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `particula/particles/particle_data.py`,
    `particula/particles/particle_data_builder.py`,
    `particula/gpu/warp_types.py`, `particula/gpu/conversion.py`, and focused
    study helpers/tests under `particula/gpu/tests/`.
  - Tests: Candidate conversion checks, round-trip tolerances, and explicit
    assertions that production defaults remain `fp64`.

- [ ] **E2-F6-P3:** Evaluate conservation small-particle fidelity memory and throughput tradeoffs
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Quantify conservation error, small-particle mass/radius fidelity,
    clamping behavior, memory footprint, and GPU/CPU throughput where available.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `particula/gpu/tests/mass_precision_metrics_test.py`, and any benchmark
    helper kept adjacent to existing GPU condensation tests to avoid a new tool
    surface.
  - Tests: Conservation tolerance checks, small-particle fidelity checks,
    memory-budget calculations, and skip-safe GPU benchmark validation.

- [ ] **E2-F6-P4:** Publish precision recommendation report with validation evidence
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record the recommended mass representation and precision policy and
    update development docs for downstream schema/dtype work.
  - Files: `docs/Features/Roadmap/mass-precision-study.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, and any short cross-link added
    to migration docs.
  - Tests: Documentation link checks plus any lightweight validation command
    needed to regenerate or verify report evidence.

## Phase Ordering Notes

- P1 defines the reproducible cases that every later comparison reuses.
- P2 should evaluate candidate representations only against the P1 baseline so
  dtype conclusions are comparable across cases.
- P3 should follow P2 because conservation and throughput tradeoffs only matter
  after the candidate precision set is fixed.
- P4 is the acceptance gate for downstream dtype work and should publish only the
  evidence produced by P1 through P3.
