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
measured contract. The shipped P1/P2 work refreshed the benchmark evidence and
recorded the measured caution band versus many-box effective region. The
shipped P3 work then kept the implementation docs-only: it updated the Epic C
roadmap and the GPU foundations guide so users can see that the current path is
accepted with caveats for many-box and low-level direct-kernel use, while large
single-box workloads remain caveated.

## User Stories

- As a maintainer, I want current single-box and multi-box coagulation benchmark
  results recorded so that Epic C decisions rely on measured behavior rather
  than assumptions.
- As a GPU user, I want documentation explaining when the current low-level
  coagulation path is appropriate so that I do not mistake it for an optimized
  large-single-box production path.
- As a future implementer, I want any parallel-within-box follow-up scoped
  separately so that this documentation update does not grow into production
  graph-capture or performance optimization work.

## Parent Epic Context

Parent epic: E3. This feature follows E3-F1, which addresses persisted RNG
semantics, and depends on E3-F2, which evaluates mixed-scale sampling
behavior. E3-F3 records the design boundary after those correctness-oriented
tracks provide a stable benchmark target.
