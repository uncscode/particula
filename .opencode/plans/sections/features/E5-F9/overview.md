# Overview

## Problem Statement

Epic E can only close after its GPU coagulation mechanisms, validation matrix,
and Epic D carry-forward walkthrough have shipped, but users still need one
accurate publication boundary that explains what is supported, how to import
and run the low-level step, and which capabilities remain deferred. The two GPU
roadmap files must also stop using unassigned placeholders and provide a
traceable E5/E5-F1 through E5-F9 closeout record with resolving artifact links.

## Value Proposition

P1 has shipped the documentation-only portion of that release: the two
canonical guides now publish the bounded direct particle-resolved contract,
and a stdlib-only regression test protects the wording, commands, and links.
The direct example and dependency-gated roadmap transition remain later phases;
they are not implied by the published documentation contract.

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
