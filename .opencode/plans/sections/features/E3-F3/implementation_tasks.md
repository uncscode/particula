# Implementation Tasks

## E3-F3-P1: Benchmark Reproduction

- Confirm E3-F2 mixed-scale sampling decision is reflected in the benchmark
  target before interpreting results.
- Run the existing coagulation benchmark matrix from
  `particula/gpu/tests/benchmark_test.py` on CUDA hardware when available;
  otherwise record the exact skip outcome instead of inventing substitute CPU
  timing evidence.
- Record the exact command, hardware/GPU name, Warp version context if printed,
  skip status, and a compact single-box versus multi-box timing summary in the
  roadmap note that will ship with the decision.
- Add or update fast tests only if benchmark metadata helpers or result parsing
  in `benchmark_test.py` change; keep raw benchmark execution out of normal CI.

## E3-F3-P2: Scaling Limit Record

- Summarize measured single-box behavior in
  `docs/Features/Roadmap/data-oriented-gpu.md` and name the particle-count band
  where one-thread-per-box stops being a recommended path.
- Summarize measured multi-box behavior in the same section and state where
  independent-box parallelism remains effective.
- Include traceable numbers, benchmark date, and the exact command that produced
  them; keep this as a docs-only update unless benchmark helper code changed.
- Avoid overstating one-machine results: label hardware context clearly and keep
  the decision language to accepted bounds, not universal GPU guarantees.

## E3-F3-P3: Usage Boundary Documentation

- Update the Epic C roadmap status with the accepted or caveated decision and
  cross-reference the benchmark evidence section added in P2.
- Add user-facing guidance for appropriate current uses in
  `docs/Features/data-containers-and-gpu-foundations.md`: many independent
  boxes, low-level experiments, and CUDA-available benchmark studies.
- Document inappropriate or caveated uses in the same doc: large single-box
  production workloads needing parallel pair selection, non-CUDA environments,
  and any workflow expecting hidden transfers.
- Sync paired benchmark notebook artifacts only if a notebook-backed source file
  is actually edited; do not create a new notebook just to hold timing notes.

## E3-F3-P4: Follow-up Scope

- If measurements require future work, draft a small follow-up scope in the
  roadmap/open-questions material for a parallel-within-box coagulation variant.
- Capture open design questions such as collision-pair write ownership, RNG
  state partitioning, conservation validation, and stochastic parity targets.
- Keep the follow-up to planning/documentation only; do not modify kernel launch
  code, Warp buffers, or collision application logic in this phase.
