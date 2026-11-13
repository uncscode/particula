# Documentation Updates

## P1 status

Issue #1581 made no public documentation change. The delivered host-only test
support module is concrete implementation support, not a user-facing benchmark
or publication surface. Documentation and artifact publication remain P6 work.

## P2 status

Issue #1582 likewise makes no public documentation change. It adds test-only
CUDA supplemental evidence and an isolated machine-generated artifact path;
the implementation docstrings define the timing boundary, CUDA-only clean-skip
behavior, artifact isolation, and absence of a speed claim. Public benchmark
publication and documentation remain P6 work.

## P3 status

Issue #1583 makes no public documentation change. It adds test-only host
matrix/preflight and opt-in artifact-consumer evidence, including structured
budget and preconstruction-unavailable rows. P4/P5 allocator/byte accounting
and P6 publication/documentation remain future work.

## P4 status

Issue #1584 makes no public documentation change. The analytical memory model
and its docstrings are private co-located execution test support; it creates no
artifact or public API.

## P6 delivered publication (Issue #1586)

- Added `docs/Features/resident_benchmark_memory_budget.md`, an explicit
  unavailable-state record for the resident benchmark schema, fixed matrix,
  planning inputs, timing/memory vocabulary, tape projections, and limitations.
- Added one roadmap link in `docs/Features/Roadmap/data-oriented-gpu.md`.
- Both documents identify
  `.artifacts/benchmarks/resident_capture_comparison.json` as the only resident
  source of record and reject legacy `gpu_benchmark_results.json` as
  coagulation-only.
- No reviewed source artifact exists in this revision: all evidence is stated as
  unavailable and not measured, with no timing, allocator, provenance, or zero
  values fabricated.
- Added `particula/tests/resident_benchmark_docs_test.py`, a hardware-free,
  stdlib-only contract test that reads only the roadmap and report; it neither
  reads the absent artifact nor imports Warp or runs benchmarks.
