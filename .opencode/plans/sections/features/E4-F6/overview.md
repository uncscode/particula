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
separately, and fixed-loop/buffer behavior is evaluated for graph capture and
autodiff readiness without claiming unsupported differentiability.

P1 is complete for issue #1308. Its deterministic fp64 matrix compares final
particle mass and coupled gas concentration independently against a NumPy
four-substep, P2-finalized, gas-coupled oracle on Warp CPU and optional CUDA.

## User Stories

- As a scientist, I want each box and species compared with an independent CPU
  reference so that vectorization errors cannot hide in aggregate results.
- As a maintainer, I want strict inventory and energy invariants on every
  available device so backend changes remain safe.
- As a GPU developer, I want explicit graph/autodiff readiness evidence and
  documented boundaries so later optimization starts from verified facts.
