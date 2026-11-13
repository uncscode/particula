# Overview

## Problem Statement

Particle creation cannot be added to fixed-shape CPU and Warp containers until
the repository has one explicit definition of active and inactive slots, a
deterministic way to discover free capacity, and mutation-safe activation.
Ad-hoc predicates would let condensation, coagulation, wall loss, resampling,
and nucleation disagree about whether a slot participates in physics.

## Value Proposition

E6-F5 establishes matching CPU and GPU discovery and activation primitives that
preserve every array shape and identity, enumerate free slots in ascending
index order, and populate exact per-box diagnostics. P4 ships the supported
package-exported direct-Warp `activate_slots_gpu` boundary with complete
preflight and caller-owned `int32` sidecars; P3 diagnostics remain the
concrete-module-only `get_slot_diagnostics_gpu` helper. The contract becomes
the stable foundation for E6-F6 exhaustion handling, E6-F7 CPU nucleation,
E6-F8 GPU nucleation, and E6-F9 integration.

## User Stories

- As a process author, I want one active/inactive predicate so every process
  treats sparse fixed-slot particle state consistently.
- As a GPU simulation developer, I want deterministic activation without
  allocation or resizing so resident arrays remain graph-friendly.
- As a diagnostics consumer, I want exact per-box integer counts in buffers I
  own so I can detect capacity pressure without hidden synchronization.

## Parent Context

This is parent epic E6 track T5. It may start independently of the dilution and
wall-loss tracks. E6-F6, E6-F7, E6-F8, and E6-F9 depend on this contract.
