# Infrastructure Reuse

- `particula/gpu/tests/benchmark_test.py:1-13` already documents opt-in CUDA
  execution and the `WARP_PROFILE=1` external-profiler hook; extend this policy
  rather than adding another collection switch.
- `particula/gpu/tests/benchmark_test.py:131-145` supplies benchmark markers,
  warmup/step defaults, and the controlled `.artifacts/benchmarks` root.
- `particula/gpu/tests/benchmark_test.py:228-292` provides artifact-safe output
  names and bounded device/version metadata. Reuse or extract these helpers for
  profiling records instead of embedding pointers or payload data.
- `particula/gpu/tests/benchmark_test.py:438-466` demonstrates incremental,
  JSON-serializable result persistence so interrupted runs retain evidence.
- `particula/gpu/tests/benchmark_test.py:469-477` and
  `particula/gpu/tests/cuda_availability.py` define clean CUDA skips with no Warp
  CPU fallback.
- `particula/gpu/tests/benchmark_test.py:550-603` contains the current optional
  profiler region and synchronized GPU timer. Keep warmup outside measurement,
  but replace ambiguous capture-as-profiling use with explicit host, end-to-end,
  and vendor-profiler categories.
- `particula/gpu/tests/benchmark_helpers_test.py:56-89` shows how fast tests
  load opt-in benchmark code in isolation; follow this for schema and analysis
  tests that do not need CUDA.
- `particula/gpu/tests/benchmark_safety_test.py:12-73` covers path and memory
  preflight guardrails that profiling artifacts must preserve.
- `particula/execution/resident_scheduler.py:69-85` is the authoritative closed
  twelve-node sequence; labels in profiler output should map to these canonical
  node IDs rather than inventing another process order.
- `particula/execution/resident_scheduler.py:196-211` forbids ordinary scheduler
  synchronization, transfer, resource acquisition, and fallback. Profiling
  synchronization stays outside the timed resident operation at sample bounds.
- E8-F3 owns stable resource identities and logical-byte accounting, E8-F4 owns
  captured replay, E8-F5 owns correctness fixtures, and E8-F6 owns scaling,
  raw timing, and memory evidence. Consume those contracts without duplicating
  setup or redefining their schemas.
- Follow `.opencode/guides/testing_guide.md:167-203` and `237-252`: performance
  evidence is opt-in, Warp CPU remains non-profile parity evidence, and optional
  CUDA rows pass or cleanly skip.
