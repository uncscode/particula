# Overview

## Problem Statement

E6-F1 through E6-F8 define the missing CPU references, direct Warp processes,
and fixed-slot services needed by Epic E6, but the epic cannot close without
integrated evidence that those contracts compose. The repository also needs a
canonical example and roadmap record that demonstrate explicit transfers while
keeping Epic G's scheduler and backend-selection responsibilities separate.

## Value Proposition

E6-F9 provides one auditable exit gate for GPU process completeness: shared
fixtures and tests exercise condensation, coagulation, dilution, neutral or
charged wall loss, and nucleation on fixed-shape device state; a runnable
example shows exactly one setup transfer and one checkpoint restore; and public
documentation links E6 and every E6 child plan to the resulting evidence.

P1 is now complete as a private test-fixture foundation in
`particula/gpu/tests/process_sequence_test.py`. It supplies deterministic fp64
one- and multi-box sparse fixtures, snapshots and ownership assertions,
independent accounting and exhaustion expectations, and optional runtime Warp
mirror checks. It deliberately does not execute the integrated process sequence
or change production or public API behavior; those remain later E6-F9 phases.

P2 is now complete for private test-only resident composition in the same
module. It exercises the five existing direct GPU boundaries on stable
same-device containers and persistent sidecars/RNGs, with final-inspection-only
conversion. It adds neither a production coordinator nor an export, hidden
transfer/fallback, runnable, or public API.

P3 is now complete for the public illustrative boundary. The runnable
`docs/Examples/gpu_complete_process_sequence.py` converts each CPU container
once, invokes condensation, coagulation, dilution, wall loss, and nucleation in
that order, synchronizes once, and restores each container once. Its focused
regression suite verifies lazy deterministic no-Warp behavior, caller-owned
sidecars/RNG, identities, transfer ordering, and visible error propagation with
no CPU fallback.

## User Stories

- As a GPU contributor, I want an integrated Warp CPU regression so that an
  individual process change cannot silently break the complete direct-call
  sequence.
- As a user, I want an explicit-transfer example so that I can compose direct
  kernels without assuming a hidden high-level runtime.
- As a maintainer, I want a dependency-gated closeout checklist and roadmap
  cross-links so that E6 closes only after E6-F1 through E6-F8 satisfy their
  contracts and Epic G's scope remains clearly deferred.
