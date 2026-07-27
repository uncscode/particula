# Overview

## Problem Statement

E7-F1 through E7-F8 define the backend-selection, resident-session, scheduling,
fallback, multi-box, and random-stream contracts, but Epic G cannot close on
component contracts alone. The system needs device-side diagnostics, independent
full-loop regressions, a checkpoint-only transfer example, and reproducible
evidence that the issue #1451 validation matrix and roadmap exit bar are met.

## Value Proposition

E7-F9 turns the integrated execution system into a supportable user contract. It
publishes bounded diagnostics, validates complete resident CPU/Warp workflows,
records optional CUDA evidence without making CUDA mandatory, and gives users a
canonical multi-timestep example whose transfers, limitations, and restart
semantics are explicit.

## User Stories

- As a simulation user, I want a complete multi-timestep example so that I can
  use backend selection without directly orchestrating GPU kernels.
- As a maintainer, I want independent full-loop and multi-box regressions so
  that ordering, conservation, transfer, checkpoint, and RNG contracts cannot
  regress silently.
- As an Epic G reviewer, I want a reproducible validation matrix and dated
  closeout evidence so that issue #1451 can close against its declared exit bar.

Parent context: E7 (Epic G). Scope authority: issue #1451 Track T9 and
`docs/Features/Roadmap/data-oriented-gpu.md:1472-1606`.
