# Infrastructure Reuse

- `ManifestEntry`, `ResourceManifest`, `_shape()`, `_item_size()`,
  `_checked_product()`, and `_allocate()` in
  `particula/execution/gpu_resources.py:100-117,488-494,1229-1377` already
  express fixed schemas and overflow-safe allocation formulas; extend these
  rather than introducing a second schema system.
- `GPUResourceRegistry._session_signature()` and
  `validate_pinned_session()` at `gpu_resources.py:559-604` already pin exact
  lifecycle, dimensions, device, container, and primary-array identities.
- `_validate_array()`, `_contiguous_range()`, `_validate_nonalias()`, and
  `_array_range()` at `gpu_resources.py:1272-1507` provide metadata-only schema,
  storage-capacity, and byte-range alias checks.
- `_acquire()` at `gpu_resources.py:1509-1562` provides the stage-then-publish
  pattern for transactional family acquisition and stable reacquisition.
- `acquire_condensation()`, `acquire_coagulation()`,
  `acquire_wall_loss()`, `acquire_nucleation()`, and
  `acquire_communication()` at `gpu_resources.py:1564-2185` define existing
  native views and must remain compatible.
- `_enumerate_resources()` and `_enumerate_published_rng_streams()` at
  `gpu_resources.py:1030-1123` establish deterministic manifest/process order
  for checkpoint consumers; byte reports should use the same ordering.
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
