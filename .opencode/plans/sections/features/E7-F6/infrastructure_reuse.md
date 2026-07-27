# Infrastructure Reuse

- E7-F1's planned `particula.execution` request, capability matrix, context, and
  CPU adapter are the required extension points; preserve the architecture in
  `.opencode/plans/sections/features/E7-F1/architecture_design.md:5-59`.
- `RunnableABC.execute()` in `particula/runnable.py:83-106` remains the CPU
  reference path. Explicit fallback delegates to the E7-F1 CPU adapter rather
  than reimplementing process physics.
- `particula/gpu/__init__.py:36-68` records deferred Warp import availability,
  and `particula/gpu/__init__.py:103-118` supplies lazy optional-symbol behavior.
  Normalize this information at the execution boundary without forcing a Warp
  import from backend-neutral modules.
- `particula/gpu/__init__.py:81-100` is the existing deliberate transfer/type
  export list. Do not add process steps or concrete sidecars there.
- `particula/gpu/kernels/__init__.py:24-64` maps the narrow lazy direct-step
  surface; keep helper kernels and configuration records in concrete modules.
- `particula/gpu/tests/kernel_exports_test.py:17-111` provides positive and
  negative export assertions to mirror for `particula.execution`.
- `particula/gpu/conversion.py:35-51` provides prior unavailable-Warp handling,
  while explicit conversion helpers in `particula/gpu/conversion.py:120-625`
  establish the transfer boundary that fallback must never hide.
- `particula/gpu/tests/conversion_test.py:480-512` exercises imports without
  Warp, and `particula/gpu/tests/cuda_availability.py:17-40` demonstrates
  non-launching CUDA availability checks used by tests.
- Issue scope is anchored in
  `docs/Features/Roadmap/data-oriented-gpu.md:1494-1497` and
  `docs/Features/Roadmap/data-oriented-gpu.md:1522-1542`.
