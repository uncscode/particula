# Overview

## Problem Statement

Epic E closes only after its GPU coagulation mechanisms, validation matrix,
and Epic D carry-forward walkthrough have shipped, and the two GPU roadmap
files provide a traceable E5/E5-F1 through E5-F9 closeout record with
resolving artifact links.

## Value Proposition

P1 shipped the documentation-only portion of the release and P2 shipped the
direct example. Issue #1374 shipped P3 reconciliation for the two roadmap
records, including one canonical E5/E5-F1--E5-F9 inventory and three resolving
artifact links. The canonical guides publish the bounded direct
particle-resolved contract; `docs/Examples/gpu_coagulation_direct.py`
demonstrates two explicit Brownian calls on optional Warp CPU with caller-owned
collision and persistent RNG sidecars, and co-located regression coverage
protects the disabled, lazy-loading, error, ownership, and invariant paths.
The hardware-free documentation regression coverage protects the documented
roadmap reconciliation. The #1375 P4 closeout record is dated 2026-07-19,
whereas P3 is recorded complete on 2026-07-20. The records therefore support
the documented status projection, but not a claim that P4 occurred after P3 or
completed a dependency-ordered gate.

## User Stories

- As a user, I want a canonical GPU coagulation support guide so that I can
  call the supported low-level API without inferring a
  high-level `Aerosol`/`Runnable`, fallback, CUDA, or performance guarantee.
- As a contributor, I want focused reproduction commands and stable artifact
  links so that I can verify each supported mechanism and its validation
  evidence.
- As a roadmap maintainer, I want E5 closeout conditioned on E5-F1 through
  E5-F9 and their release gates so that Epic F is activated only after all
  dependencies pass.

**Parent:** E5 — GPU Coagulation Physics Coverage  
**Track:** T9 — Support Documentation, Example, and Roadmap Closeout  
**Classifier diagnostics:** none
