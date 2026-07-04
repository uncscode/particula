# Overview

## Problem Statement

Issues #1188 and #1189 shipped E2-F2 phases P1 and P2 by adding a CPU-side
`EnvironmentData` container in `particula/gas/environment_data.py`, exposing it
through `particula.gas`, and completing the convenience API with `n_boxes` and
independent `copy()` semantics. Existing `GasData` still owns gas-species state
and existing dynamics APIs still pass `temperature` and `pressure` as scalars,
so the remaining feature work is documentation and downstream process
migration rather than container API completion.

## Value Proposition

`EnvironmentData` now exists as the third CPU data container identified by
parent epic E2: `ParticleData` for particles, `GasData` for gas species, and
`EnvironmentData` for per-box thermodynamic state. With P2 shipped,
downstream phases can rely on the canonical package import path,
box-count convenience property, and copy-safe array ownership while preserving
the deterministic coercion and validation rules established in P1.

## User Stories

- As a dynamics implementer, I want per-box temperature and pressure stored in
  a validated container so future processes can read state without scalar-only
  assumptions.
- As a model author, I want humidity or saturation state shaped by box so I can
  represent single-box and multi-box simulations consistently.
- As a maintainer, I want invalid environment shapes and values rejected early
  so downstream GPU and process migrations have a reliable CPU baseline.

## Parent Epic Context

- Parent epic: E2, issue #1172, Data-Model and Numerical Foundations v2.
- Dependency: E2-F1 schema foundation should define the common container
  conventions this feature follows.
- Sibling tracks include GPU mirror/conversion work and kernel/process
  migrations; this feature only establishes the CPU environment container.
