# Implementation Tasks

## E3-F3-P1: Benchmark Reproduction

- Confirm the E3-F2 mixed-scale choice is reflected in the coagulation benchmark
  case that `particula/gpu/tests/benchmark_test.py` actually exercises before
  treating new timings as decision evidence.
- If benchmark setup plumbing changes, keep the code edit inside named helpers
  such as `_sanitize_benchmark_output_name()` or `_get_benchmark_output_path()`
  rather than refactoring unrelated benchmark families in the same phase.
- Run the existing coagulation benchmark entrypoint from
  `particula/gpu/tests/benchmark_test.py` on CUDA hardware when available;
  otherwise copy the exact skip outcome into
  `docs/Features/Roadmap/data-oriented-gpu.md` instead of inventing substitute
  CPU timing evidence.
- Record one compact evidence block with the exact pytest command, GPU/hardware
  name, Warp version context if printed, and single-box versus multi-box timing
  summary; keep the shipped docs edit to roughly one roadmap subsection plus any
  truly required benchmark helper delta so the phase stays near the 100-LOC
  review target.

## E3-F3-P2: Scaling Limit Record

- Summarize measured single-box behavior in a dedicated roadmap subsection of
  `docs/Features/Roadmap/data-oriented-gpu.md` and name the particle-count band
  where one-thread-per-box stops being a recommended path.
- Summarize measured multi-box behavior in the same subsection and state where
  independent-box parallelism remains effective.
- If `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py` is updated, limit the
  code change to reproducing the shipped command/output context rather than
  redesigning the theory example.
- Include traceable numbers, benchmark date, and the exact command that produced
  them, and label the evidence as one-machine bounded guidance rather than a
  universal GPU guarantee.

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

- If measurements require future work, draft the follow-up in exactly two plan
  locations: `docs/Features/Roadmap/data-oriented-gpu.md` for the accepted
  boundary and `.opencode/plans/sections/features/E3-F3/open_questions.md` for
  unresolved design choices.
- Capture concrete design questions such as collision-pair write ownership, RNG
  state partitioning, conservation validation, and stochastic parity targets.
- Keep the follow-up to planning/documentation only; do not modify kernel launch
  code, Warp buffers, collision application logic, or benchmark kernels in this
  phase.
