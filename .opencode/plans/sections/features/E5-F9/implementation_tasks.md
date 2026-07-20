# Implementation Tasks

### Documentation and Example

- [x] Audit the final E5-F1-F7 API/evidence before writing support claims.
- [x] Update the foundations and coagulation strategy guides with supported
  mechanisms, combinations, distribution boundary, ownership, and imports.
- [x] Add focused Warp CPU and optional CUDA reproduction commands.
- [x] Implement `docs/Examples/gpu_coagulation_direct.py` using public transfer
  helpers, lazy kernel imports, fixed-shape caller buffers, and persistent RNG.
- [x] Add the direct-example reference to `docs/index.md`.

### Roadmap and Closeout

- [x] P3: Reconciled Epic E's historical `not scheduled` placeholder with
  plan `E5`.
- [x] P3: Added a one-to-one E5-F1 through E5-F9 track table with canonical
  statuses and artifact links in both roadmap records.
- [x] P3: Reconciled E5-F7/E5-F8 artifact ownership and historical
  active/pre-closeout wording in both roadmap files without duplicating their
  detailed evidence.
- [x] Encoded and executed the closeout checklist; failed prerequisites would
  have preserved the pre-closeout state.
- [x] After a clean gate, marked E5 shipped and Epic F active consistently.

### Tooling / Tests

- [x] Add `particula/gpu/tests/gpu_coagulation_direct_example_test.py`.
- [x] Add `particula/tests/gpu_coagulation_docs_test.py` for direct-contract
  claims, commands, and links using only the Python standard library.
- [x] Ran E5-F7's required Warp CPU parity/stochastic matrix and standard docs,
  lint, and fast test checks; keep CUDA optional and skip-safe.
