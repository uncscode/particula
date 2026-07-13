# Phase Details

- [ ] **E4-F4-P1:** Warp thermal-resistance helpers and validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Port conductivity/resistance and validate latent sidecars before work.
  - Files: `particula/gpu/dynamics/condensation_funcs.py`, `particula/gpu/kernels/condensation.py`
  - Tests: CPU formula parity; value, shape, dtype, and device validation.

- [ ] **E4-F4-P2:** Per-substep latent-heat correction with parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Correct the common E4-F2 surface-pressure rate in all four substeps.
  - Files: `particula/gpu/kernels/condensation.py`
  - Tests: Corrected-rate parity, deterministic substeps, fallback, scratch reuse.

- [ ] **E4-F4-P3:** Signed whole-call energy bookkeeping with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Accumulate per-box/species energy from bounded applied transfer.
  - Files: `particula/gpu/kernels/condensation.py`
  - Tests: Sign, identity, clamp, aggregation, isolation, pre-mutation failure.

- [ ] **E4-F4-P4:** Warp integration regressions and documentation updates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compose sibling contracts and document issue #1272 behavior.
  - Files: GPU condensation tests, `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: Warp CPU integration, optional CUDA parity, docs validation.
