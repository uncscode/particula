# Overview

## Problem Statement

Existing low-level GPU kernel APIs accept scalar `temperature` and `pressure`
values. That keeps single-box and uniform-environment workflows simple, but it
blocks E2's broader multi-box roadmap because physics kernels cannot consume a
distinct thermodynamic environment per box. Feature E2-F5 defines the migration path
from those scalar inputs to per-box environment state while preserving current
scalar API compatibility.

## Value Proposition

- Current callers of `condensation_step_gpu` and `coagulation_step_gpu` keep
  working with scalar temperature/pressure inputs.
- Multi-box GPU callers gain a validated path for per-box environment arrays,
  aligned with E2-F2 environment containers and E2-F3 GPU environment-transfer
  conventions.
- Later GPU physics work can depend on a consistent environment feed point
  instead of adding one-off temperature/pressure handling per kernel.

## User Stories

- As an existing GPU API caller, I want scalar temperature and pressure calls to
  continue passing so my current simulations do not break.
- As a multi-box model developer, I want per-box environment inputs validated
  against `n_boxes` so kernel launches fail early on shape mismatches.
- As a future physics kernel implementer, I want a stable environment state
  interface so latent heat, parcel expansion, and vapor-pressure updates can
  consume box-local thermodynamic values.

## Parent Epic Context

Parent epic: E2. This feature is E2-F5 for issue #1172. It depends on E2-F2
for the environment container schema and E2-F3 for GPU environment-transfer
conventions. Sibling tracks E2-F1 through E2-F4 establish schema,
environment-container, and vapor-pressure foundations; E2-F6 through E2-F9 can
build on the compatibility path created here.
