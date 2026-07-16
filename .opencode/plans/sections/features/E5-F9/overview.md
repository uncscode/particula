# Overview

## Problem Statement

Epic E can only close after its GPU coagulation mechanisms, validation matrix,
and Epic D carry-forward walkthrough have shipped, but users still need one
accurate publication boundary that explains what is supported, how to import
and run the low-level step, and which capabilities remain deferred. The two GPU
roadmap files must also stop using unassigned placeholders and provide a
traceable E5/E5-F1 through E5-F9 closeout record with resolving artifact links.

## Value Proposition

E5-F9 turns the dependency-complete implementation into a discoverable,
runnable, and auditable release. Users receive a Warp-CPU-default direct
coagulation example and explicit support caveats; maintainers receive a
dependency-gated roadmap transition that cannot mark E5 shipped or Epic F
active while any required child, test gate, or artifact remains incomplete.

## User Stories

- As a user, I want a canonical GPU coagulation support guide and runnable
  example so that I can call the supported low-level API without inferring a
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
