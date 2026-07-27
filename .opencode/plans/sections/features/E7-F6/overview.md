# Overview

## Problem Statement

Issue #1451 Track T6 requires the backend-selection layer to behave predictably
when Warp, a requested device, or a requested GPU process is unavailable. Today
the direct GPU surface uses several import and runtime error patterns, while no
single policy defines when CPU fallback is permitted, what users may import, or
which APIs are stable.

## Value Proposition

E7-F6 gives E7 adapters and sessions one typed error taxonomy, fail-closed
availability checks, and an opt-in CPU transition that is observable and occurs
only before mutation or at an explicit restore boundary. It also freezes a
narrow public execution surface while marking low-level `particula.gpu.*` APIs
experimental until backend selection and full-loop validation are complete.

## User Stories

- As a simulation author, I want unsupported GPU requests to fail with a clear,
  inspectable capability error so I can correct configuration safely.
- As an operator, I want CPU fallback to require an explicit request so a long
  resident run never moves data or changes backend silently.
- As a library consumer, I want documented stable and experimental import paths
  so upgrades do not unexpectedly bind me to internal sidecars or kernels.
