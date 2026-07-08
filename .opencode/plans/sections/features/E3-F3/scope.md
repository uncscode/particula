# E3-F3 Scope

E3-F3 reproduces or refreshes coagulation GPU benchmark evidence, records the
measured scaling limit of the current one-thread-per-box implementation, and
updates documentation to state whether this design is accepted for Epic C or
requires a follow-up parallel-within-box variant.

## In Scope

- Run or refresh the existing opt-in coagulation benchmark matrix for
  single-box and multi-box configurations where CUDA hardware is available.
- Preserve CUDA optionality by keeping benchmark tests skipped when CUDA or the
  `--benchmark` flag is unavailable.
- Record measured single-box and multi-box scaling limits in roadmap or benchmark
  documentation.
- Document that the current low-level coagulation GPU path is best suited to
  many independent boxes and direct experimental use unless measurements justify
  broader claims.
- Decide whether one-thread-per-box is accepted for Epic C or whether a future
  parallel-within-box track should be created.
- Add fast regression coverage for any benchmark metadata or documentation
  helper changes.

## Out of Scope

- Production graph capture implementation or optimization.
- Rewriting the coagulation kernel for parallel pair selection within a box.
- Changing CPU/GPU container transfer contracts or hidden synchronization
  behavior.
- Replacing E3-F1 RNG work or E3-F2 mixed-scale sampling decisions.
- Making CUDA mandatory in normal development, CI, or documentation builds.
