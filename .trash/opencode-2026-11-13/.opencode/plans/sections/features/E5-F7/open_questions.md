# Open Questions

- [x] What is the exact final list of approved two-way masks emitted by E5-F6?
  - Resolved 2026-07-16: validate masks `3`, `5`, `9`, `6`, `10`, and `12`, plus
    singleton masks `1`, `2`, `4`, `8` and full mask `15`. Assert exact equality
    between the test case table and the executable capability table. Three-way
    masks remain unsupported.
- [x] Should the cross-mechanism cases remain in
  `coagulation_validation_test.py` or be split into deterministic and stochastic
  modules?
  - Resolved 2026-07-16: use `coagulation_validation_test.py` for deterministic,
    majorant, conservation, and edge cases; use
    `coagulation_stochastic_validation_test.py` for repeated-seed checks; keep
    one canonical case table in `_coagulation_validation_support.py`.
- [x] What sample count and confidence/sigma threshold gives useful power for
  every mechanism while keeping the required suite fast?
  - Resolved 2026-07-16: use 100 fresh independent seeds and a predeclared
    `3 * sqrt(expected_mean)` aggregate bound. Author uncapped fixtures with an
    expected aggregate count of at least 100; test zero-rate rows deterministically.
- [x] Which existing mechanism-specific fixtures can be imported safely without
  coupling expected values to implementation internals?
  - Resolved 2026-07-16: reuse public CPU formulas and shared device helpers, but
    do not import private fixtures from another `*_test.py` or use Warp helpers
    as the oracle. Copy compact explicit fp64 physical records into the neutral
    support module and calculate additive expectations independently.
- [x] Where should E5-F9 publish the final user-facing evidence table?
  - Resolved 2026-07-16: publish the canonical detailed report at
    `docs/Features/Roadmap/coagulation-validation.md`. E5-F9 links it from both
    roadmap files, the coagulation/foundations guides, and the direct example.

Resolved constraints: parent is E5; classifier diagnostics are none; Warp CPU is
required when Warp is installed; CUDA is optional; exact CPU/Warp pair replay
and performance claims are out of scope.
