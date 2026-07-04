# E2-F6 Success Criteria

- A precision/mass representation report exists and is linked from the relevant
  roadmap or migration documentation.
- The report defines NPF-to-droplet numerical cases and records how to reproduce
  them.
- Current absolute per-species `fp64` mass storage is evaluated as the reference
  baseline.
- At least `fp32` and mixed-precision candidates are compared; representation
  alternatives are evaluated or explicitly ruled out with rationale.
- Conservation, small-particle fidelity, nonnegative/clamping behavior, memory
  budget, and runtime/throughput tradeoffs are reported.
- Production schema and dtype defaults remain unchanged unless a later feature
  explicitly implements an accepted recommendation.
- The recommendation clearly states whether to keep absolute `fp64`, pursue a
  mixed-precision path, or investigate a different mass representation.
- Fast validation tests pass, and slow/GPU evidence is either reproducible or
  skip-safe with documented environment requirements.
