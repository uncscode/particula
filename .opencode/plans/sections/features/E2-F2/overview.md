# Overview

## Problem Statement

Issue #1188 shipped E2-F2 phase P1 by adding a CPU-side `EnvironmentData`
container for per-box thermodynamic state in
`particula/gas/environment_data.py`. Existing `GasData` still owns gas-species
state and existing dynamics APIs still pass `temperature` and `pressure` as
scalars, so the shipped work establishes the validated container baseline but
does not yet migrate process call sites.

## Value Proposition

`EnvironmentData` now exists as the third CPU data container identified by
parent epic E2: `ParticleData` for particles, `GasData` for gas species, and
`EnvironmentData` for per-box thermodynamic state. P1 gives downstream phases a
validated direct-module container with deterministic coercion and validation
rules before they add exports, copy helpers, or process/GPU integrations.

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
