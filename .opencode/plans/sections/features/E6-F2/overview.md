# Overview

## Problem Statement

Epic F needs dilution to run directly over fixed-shape Warp particle and gas
containers. Today dilution is CPU-only, and adding host-side orchestration around
each step would violate the explicit-transfer and GPU-residency boundaries needed
by the later integrated process sequence.

## Value Proposition

E6-F2 ports the E6-F1/T1 dilution contract to a deterministic low-level Warp
step. It gives GPU callers scalar and per-box dilution inputs, updates only
particle number concentration and gas mass concentration in place, and provides
CPU/Warp parity without hidden transfers, fallback, resizing, or a high-level
GPU runnable.

## User Stories

- As a GPU simulation developer, I want dilution to mutate caller-owned Warp
  concentrations directly so that timesteps remain device resident.
- As a scientific maintainer, I want single- and multi-box results compared with
  the T1 CPU oracle so that finite-step semantics and units remain consistent.
- As an API user, I want malformed calls rejected before mutation so that failed
  steps cannot leave particle and gas state inconsistent.

**Parent context:** E6 (GPU Process Completeness). This is issue track T2 and
depends on E6-F1/T1. E6-F9 consumes this feature in integrated direct-step
validation; wall-loss and slot-management sibling tracks remain independent.
