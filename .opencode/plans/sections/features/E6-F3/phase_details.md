# Phase Details

- [ ] **E6-F3-P1:** Port and validate neutral wall-loss transport primitives with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Inventory reusable Warp physics and implement only missing fp64 device functions needed by both geometries.
  - Files: `particula/gpu/properties/`, `particula/gpu/dynamics/`, co-located `tests/`
  - Tests: CPU parity for viscosity/slip/diffusion/settling/Debye or coth over representative particle scales and edge domains.

- [ ] **E6-F3-P2:** Implement spherical and rectangular coefficient device functions with CPU parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Port the authoritative neutral Crump-Seinfeld coefficient equations without changing CPU behavior.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Tests: One-particle/vector spherical and rectangular comparisons to independent CPU functions at explicit fp64 tolerances.

- [ ] **E6-F3-P3:** Define neutral wall-loss configuration and atomic preflight tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze geometry, environment, time, particle schema, and caller-sidecar contracts before any mutation path exists.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Valid spherical/rectangular forms plus malformed geometry, dimensions, values, shapes, dtypes, devices, and failure immutability.

- [ ] **E6-F3-P4:** Implement fixed-shape coefficient and stochastic removal kernels with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Evaluate active-slot coefficients, sample survival, and clear mass/concentration/charge for lost slots in place.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Inactive gaps, all-survive/all-remove bounds, zero-time no-op, multi-species clearing, identity and shape preservation.

- [ ] **E6-F3-P5:** Add caller-owned persistent RNG lifecycle with repeated-step tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Reuse validated per-box RNG buffers without hidden reseeding and provide an explicit initialization/reset path.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Omitted convenience state, initialize-once reuse, explicit reset, per-box advancement, identity retention, and invalid-call non-advancement.

- [ ] **E6-F3-P6:** Add deterministic coefficient and statistical CPU-Warp parity matrix
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Demonstrate both geometries match CPU coefficients and expected survival distributions on Warp CPU, with optional CUDA evidence.
  - Files: `particula/gpu/kernels/tests/wall_loss_parity_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: Single/multi-box, particle-scale, sparse/inactive, repeated-step, deterministic coefficient, survival confidence bounds, and import smoke coverage.

- [ ] **E6-F3-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document direct use, persistent RNG ownership, fixed-slot removal, support boundaries, and focused validation commands.
  - Files: `AGENTS.md`, `docs/Features/`, `.opencode/guides/`, E6 plan sections as needed
  - Tests: Markdown links, import snippets, support/deferred tables, and focused commands.
