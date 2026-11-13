# Overview

## Problem Statement

E4-F1 through E4-F5 add the issue 1272 condensation physics, fixed four-step
integration, thermal feedback, and gas coupling. Without a device-aware
evidence suite, correctness can still regress by backend, box, species, buffer
reuse, or execution mode. Graph and autodiff claims also need bounded evidence
rather than inference from kernel shape.

## Value Proposition

E4-F6 establishes an independent, reproducible acceptance gate: Warp CPU is
mandatory, CUDA runs when available, parity and strict conservation are tested
separately, and graph/autodiff probes establish explicit capability boundaries
without claiming unsupported capture or differentiability.

P1 is complete for issue #1308. Its deterministic fp64 matrix compares final
particle mass and coupled gas concentration independently against a NumPy
four-substep, P2-finalized, gas-coupled oracle on Warp CPU and optional CUDA.

P2 is complete for issue #1309. It adds Warp-CPU contract regressions for
per-box/per-species concentration-weighted particle-plus-gas conservation,
P2-finalized transfer and unweighted latent-energy accounting, immutable
caller inputs, atomic invalid-buffer rejection, and deterministic fresh runs.
The delivered work is test-only and does not change production APIs or physics.

P3 is complete for issue #1310 as a documented limitation. Warp CPU capture is
capability-skipped because Warp capture requires CUDA. CUDA public-step replay
is strict-xfailed because host validation readbacks are not capture-safe. These
guarded outcomes document unsupported capture, not graph-replay readiness; no
production API or behavior changed.

P4 is complete for issue #1311. Its dedicated autodiff test records a bounded
one-box out-of-place raw-rate Warp Tape derivative against a centered-fp64
reference, enables and restores Warp array-access verification, and adds
optional CUDA evidence. Separate P2 tests prove evaporation clamping,
inventory-limited uptake, and in-place mass mutation only as forward semantics;
they make no backward claim.

P5 is complete for issue #1312. The shipped documentation matrix distinguishes
P1 parity, P2 strict conservation/contracts, P3's unsupported capture limit,
and P4's raw-rate-only interior derivative evidence. It records Warp CPU as
the normal supported-probe backend, CUDA as optional, and preserves all direct
kernel, graph, and differentiability non-claims.

## User Stories

- As a scientist, I want each box and species compared with an independent CPU
  reference so that vectorization errors cannot hide in aggregate results.
- As a maintainer, I want strict inventory and energy invariants on every
  available device so backend changes remain safe.
- As a GPU developer, I want explicit graph/autodiff evidence and documented
  boundaries so later optimization starts from verified facts.
