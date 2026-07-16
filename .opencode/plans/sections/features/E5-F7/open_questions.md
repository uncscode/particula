# Open Questions

- [ ] What is the exact final list of approved two-way masks emitted by E5-F6?
  - Resolve before P1 implementation by reading the shipped configuration; the
    matrix must fail closed for any executable row without evidence.
- [ ] Should the cross-mechanism cases remain in
  `coagulation_validation_test.py` or be split into deterministic and stochastic
  modules?
  - Decide in P1 based on collection clarity; preserve one canonical case table
    either way.
- [ ] What sample count and confidence/sigma threshold gives useful power for
  every mechanism while keeping the required suite fast?
  - Resolve in P3 before observing pass/fail outcomes; document derivation and
    measured runtime.
- [ ] Which existing mechanism-specific fixtures can be imported safely without
  coupling expected values to implementation internals?
  - Prefer neutral test-support records; copy small explicit fixtures when
    importing would create circular or private dependencies.
- [ ] Where should E5-F9 publish the final user-facing evidence table?
  - Coordinate during P4 with the roadmap document selected by E5-F9; E5-F7
    must still retain focused commands and evidence references.

Resolved constraints: parent is E5; classifier diagnostics are none; Warp CPU is
required when Warp is installed; CUDA is optional; exact CPU/Warp pair replay
and performance claims are out of scope.
