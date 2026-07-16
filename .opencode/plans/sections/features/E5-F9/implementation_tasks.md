# Implementation Tasks

### Documentation and Example

- [ ] Audit the final E5-F1-F7 API/evidence before writing support claims.
- [ ] Update the foundations and coagulation strategy guides with supported
  mechanisms, combinations, distribution boundary, ownership, and imports.
- [ ] Add focused Warp CPU and optional CUDA reproduction commands.
- [ ] Implement `docs/Examples/gpu_coagulation_direct.py` using public transfer
  helpers, lazy kernel imports, fixed-shape caller buffers, and persistent RNG.
- [ ] Add the example and guide links to `docs/Examples/index.md`.

### Roadmap and Closeout

- [ ] Replace Epic E's `not scheduled` placeholder with plan `E5`.
- [ ] Add a one-to-one E5-F1 through E5-F9 track table with shipped artifacts.
- [ ] Reconcile completed scope and E5-F8 deferred owners in both roadmap files.
- [ ] Encode and execute the closeout checklist; preserve E5 active/Epic F
  pending on any child, test, example, artifact, or link failure.
- [ ] After a clean gate only, mark E5 shipped and Epic F active consistently.

### Tooling / Tests

- [ ] Add `particula/gpu/tests/gpu_coagulation_direct_example_test.py`.
- [ ] Add `particula/tests/gpu_coagulation_docs_test.py` for claims, IDs, status,
  commands, and links.
- [ ] Run E5-F7's required Warp CPU parity/stochastic matrix and standard docs,
  lint, and fast test checks; keep CUDA optional and skip-safe.
