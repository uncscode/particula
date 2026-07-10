# E3-F3 Overview: One-thread-per-box coagulation decision

## Problem Statement

Epic C currently uses a low-level Warp coagulation kernel that launches one GPU
thread per simulation box and performs pair selection sequentially inside that
thread. This is simple, race-resistant, and aligned with many-independent-box
workloads, but it can serialize large single-box coagulation work. The project
needs recorded benchmark evidence and a clear design decision before presenting
this path as an accepted Epic C GPU capability.

## Value Proposition

This feature turns the known one-thread-per-box limitation into an explicit,
measured contract. The shipped P1 work refreshed the opt-in coagulation-only
benchmark path with a dedicated mixed-scale NPF/droplet fixture, added fast
helper coverage proving the coagulation path stayed isolated from condensation,
and recorded one compact benchmark-evidence block in the roadmap. Users and
maintainers will know which benchmark evidence supports current guidance and
whether a future parallel-within-box variant is needed.

## User Stories

- As a maintainer, I want current single-box and multi-box coagulation benchmark
  results recorded so that Epic C decisions rely on measured behavior rather
  than assumptions.
- As a GPU user, I want documentation explaining when the current low-level
  coagulation path is appropriate so that I do not mistake it for an optimized
  large-single-box production path.
- As a future implementer, I want any parallel-within-box follow-up scoped
  separately so that this documentation feature does not grow into production
  graph-capture or performance optimization work.

## Parent Epic Context

Parent epic: E3. This feature follows E3-F1, which addresses persisted RNG
semantics, and depends on E3-F2, which evaluates mixed-scale sampling
behavior. E3-F3 records the design boundary after those correctness-oriented
tracks provide a stable benchmark target.
