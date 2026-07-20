# Overview

## Problem Statement

Epic E can only close after its GPU coagulation mechanisms, validation matrix,
and Epic D carry-forward walkthrough have shipped, but users still need one
accurate publication boundary that explains what is supported, how to import
and run the low-level step, and which capabilities remain deferred. The two GPU
roadmap files must also stop using unassigned placeholders and provide a
traceable E5/E5-F1 through E5-F9 closeout record with resolving artifact links.

## Value Proposition

P1 shipped the documentation-only portion of the release and P2 shipped the
direct example. Issue #1374 documents P3 reconciliation work for the two
roadmap records, including one canonical E5/E5-F1--E5-F9 inventory and three
resolving artifact links, but E5-F9-P3 remains Not Started in authoritative
metadata. The canonical
guides publish the bounded direct
particle-resolved contract; `docs/Examples/gpu_coagulation_direct.py` now
demonstrates two explicit Brownian calls on optional Warp CPU with caller-owned
collision and persistent RNG sidecars, and co-located regression coverage
protects the disabled, lazy-loading, error, ownership, and invariant paths.
The hardware-free documentation regression coverage protects the documented
roadmap reconciliation, while P3 and the dependency-gated E5/Epic F transition
remain unfinished; P4 is the sole closeout phase.

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
