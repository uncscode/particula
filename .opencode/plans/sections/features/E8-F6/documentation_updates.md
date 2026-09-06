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
artifact or public API. P5 observed-memory evidence and P6 publication remain
future work.

- Update `docs/Features/Roadmap/data-oriented-gpu.md` under Epic H Performance
  and Memory with the exact plain reproduction command, date, Warp/Python
  versions, qualified device, matrix, raw artifact path, summary table, memory
  categories, analytical/observed comparison, and machine-bounded caveats.
- Add or update a focused feature report under `docs/Features/` describing the
  benchmark schema, fair captured/uncaptured timing boundary, configured budget,
  structured unavailable rows, and how to interpret logical versus observed
  versus projected bytes.
- Hand the published evidence to E8-F7 for profiling and link E8-F8's
  graph-capture example/limits page to it; leave runnable lifecycle ownership
  in E8-F8.
- Update `.opencode/guides/testing_guide.md` only if the concrete resident
  benchmark command or artifact convention adds a reusable repository policy;
  preserve `--benchmark` as the only collection-affecting option.
- Update `AGENTS.md` with the focused reproduction command and evidence location
  only when useful for future contributors; do not paste machine-specific
  timings into general quick-start text.
- Keep `.artifacts/benchmarks/` results machine-generated and identify the
  reviewed source-of-record artifact explicitly. Never present unavailable rows
  as zero time or zero memory.
- Reconcile these plan sections and phase states after implementation and run
  documentation contract tests plus `mkdocs build --strict`.
