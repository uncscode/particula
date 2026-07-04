# Overview

## Problem Statement

Issues #1188, #1189, and #1190 shipped E2-F2 phases P1, P2, and P3 by adding a
CPU-side `EnvironmentData` container in `particula/gas/environment_data.py`,
exposing it through `particula.gas`, completing the convenience API with
`n_boxes` and independent `copy()` semantics, and documenting the shipped
ownership boundary in the feature guide and GPU roadmap. Existing `GasData`
still owns gas-species state, simulation volume remains under
`ParticleData.volume`, and current dynamics APIs may still pass `temperature`
and `pressure` as scalars until downstream migration tracks update them.

## Value Proposition

`EnvironmentData` now exists as the third CPU data container identified by
parent epic E2: `ParticleData` for particles, `GasData` for gas species, and
`EnvironmentData` for per-box thermodynamic state. With P3 shipped, downstream
phases can rely on a documented CPU contract: `EnvironmentData` owns
`temperature`, `pressure`, and `saturation_ratio`; `ParticleData.volume`
remains the authoritative simulation-volume owner; and GPU mirrors,
CPU↔GPU conversion helpers, and runtime/kernel integration remain downstream.

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
  migrations; this feature establishes and documents the CPU environment
  contract only.
