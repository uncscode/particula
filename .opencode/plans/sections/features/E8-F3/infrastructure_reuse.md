# Infrastructure Reuse

- `ManifestEntry`, `ResourceManifest`, `_shape()`, `_item_size()`,
  `_checked_product()`, and `_allocate()` in
  `particula/execution/gpu_resources.py` express fixed schemas and overflow-safe
  allocation formulas; extend these rather than introducing a second schema
  system.
- `GPUResourceRegistry._session_signature()` and `validate_pinned_session()`
  pin exact lifecycle, dimensions, device, container, and primary-array
  identities.
- `_validate_array()`, `_contiguous_range()`, `_validate_nonalias()`, and
  `_array_range()` provide metadata-only schema, storage-capacity, and byte-range
  alias checks. P4 reuses these checks while comparing transaction-local staged
  ranges without publishing candidate bindings.
- `_acquire()` supplies the established stage-then-publish family-acquisition
  pattern. P4 reuses compatible existing publications and uses separate private
  staging for its atomic whole-set transaction rather than public acquisition.
- `acquire_condensation()`, `acquire_coagulation()`, `acquire_wall_loss()`,
  `acquire_nucleation()`, and `acquire_communication()` define existing native
  views and remain compatible with P4 preparation.
- `_enumerate_resources()` and `_enumerate_published_rng_streams()` establish
  deterministic manifest/process order for checkpoint consumers; byte reports
  use the same ordering.
- Native records from `particula.gpu.kernels.condensation`, `.nucleation`,
  `.exhaustion`, and `.communication` remain the authoritative kernel-facing
  carriers; do not duplicate their schemas.
- `particula/execution/tests/gpu_resources_test.py` already covers stable
  repeated acquisition, zero dimensions, identity drift, cross-family aliasing,
  transactional failure, persistent RNG, and manifest ordering. Extend that
  suite using Warp CPU as the installed-Warp baseline.
- E8-F1 compatibility signatures and E8-F2 prepared launch records are the
  authoritative consumers. E8-F3 should bind their resource requirements, not
  create another scheduler or process-selection model.
