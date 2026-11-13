# Dependencies

## Upstream

E7-F1 has no active E7 child-plan dependency and is the root of issue #1451's
authoritative chain. It relies on shipped infrastructure rather than unfinished
feature plans:

- E2 fixed-shape CPU/Warp container schemas, ownership, and explicit conversion
  boundaries.
- Shipped direct GPU process contracts, especially condensation and coagulation,
  to describe capabilities without changing their physics.
- E6-F1 through E6-F9 process primitives and integrated fixtures.
- CPU `RunnableABC` and `RunnableSequence` behavior as the reference path.
- Python 3.12+, NumPy, and optional Warp; CUDA hardware is not required.

## Downstream

- **E7-F6** immediately depends on E7-F1 to define fallback, availability,
  error, export, and API-stability policy.
- **E7-F2, E7-F3, and E7-F4** depend on both E7-F1 and E7-F6 for condensation,
  coagulation, and resident-session implementations.
- **E7-F5** transitively consumes the context through process adapters and
  resident state; E7-F7/E7-F8 extend the resulting loop.
- **E7-F9** validates the complete public contract and depends on all E7 tracks.
- Epics H and I depend on the eventual stable resident execution boundary; no
  graph-capture, performance, or autodiff requirement enters E7-F1.

Authoritative sequence:
`E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5 -> {E7-F7, E7-F8} -> E7-F9`.

## Phase Ordering

P1 capability vocabulary precedes P2 context resolution. P3 freezes state and
result semantics before P4 implements the CPU adapter. P5 publishes only the
surface proven by P1-P4. P6 documents the finalized contract. Every production
phase includes its own tests; no standalone testing phase is permitted.
