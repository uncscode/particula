# Overview

## Problem Statement

Issue #1172 feature E2-F2 needs a CPU-side `EnvironmentData` container for
per-box thermodynamic state. Existing `GasData` owns gas-species state and
existing dynamics APIs pass `temperature` and `pressure` as scalars, leaving
multi-box temperature, pressure, humidity, and saturation state without a
consistent home.

## Value Proposition

`EnvironmentData` creates the third CPU data container identified by parent
epic E2: `ParticleData` for particles, `GasData` for gas species, and
`EnvironmentData` for per-box thermodynamic state. This keeps schema ownership
clear before downstream tracks add GPU mirrors and migrate process kernels.

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
