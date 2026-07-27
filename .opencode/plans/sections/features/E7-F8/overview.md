# Overview

## Problem Statement

Issue #1451 Track T8 requires stochastic GPU processes to behave as parts of a
restartable simulation rather than unrelated kernel calls. Coagulation and wall
loss currently accept mutable per-box `uint32` sidecars, but positional seeding,
implicit one-shot allocation, or incomplete checkpoints can make a box's random
sequence depend on neighboring boxes, scheduling changes, or restarts.

## Value Proposition

E7-F8 gives every stochastic process and logical box a stable stream identity,
seeds it only by explicit intent, retains it in the E7-F4 resident session, and
continues it exactly from an explicit checkpoint. E7-F3 coagulation and E7-F5
scheduling gain reproducible behavior without hidden transfers, synchronization,
fallback, or claims of exact CPU/CUDA trajectory equality.

## User Stories

- As a simulation user, I want a logical box to reproduce its stochastic path
  from the same seed even when unrelated boxes are added, disabled, or reordered.
- As an operator, I want checkpoint/restart to continue each process stream from
  its saved state so that a split run matches an uninterrupted run on the same
  backend and device class.
- As a maintainer, I want initialization and reset to be explicit so that normal
  scheduler steps cannot silently reseed persistent state.

Parent epic: E7. Scope authority: issue #1451 Track T8. Upstream feature plans:
E7-F3, E7-F4, and E7-F5; E7-F9 consumes the resulting closeout evidence.
