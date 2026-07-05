# Overview

## Problem Statement

Existing low-level GPU kernel APIs originally only accepted scalar
`temperature` and `pressure` values. That kept single-box and
uniform-environment workflows simple, but it blocked E2's broader multi-box
roadmap because physics kernels could not consume distinct thermodynamic state
per box. Issue #1203 shipped the compatibility contract, and issue #1204
implemented the shared normalization and validation path that turns supported
direct inputs into canonical `(n_boxes,)` Warp arrays before launch work.

## Value Proposition

- Current callers of `condensation_step_gpu` and `coagulation_step_gpu` still
  work with scalar temperature/pressure inputs.
- GPU callers can now pass direct `(n_boxes,)` Warp arrays or a
  `WarpEnvironmentData` container and get stable pre-launch validation.
- Later GPU physics work can depend on one shared environment normalization
  boundary instead of adding one-off temperature/pressure handling per kernel.

## User Stories

- As an existing GPU API caller, I want scalar temperature and pressure calls to
  continue passing so my current simulations do not break.
- As a multi-box model developer, I want explicit per-box environment inputs to
  execute through the existing entry points without hidden CPU transfers.
- As a future physics kernel implementer, I want a stable environment state
  interface so latent heat, parcel expansion, and vapor-pressure updates can
  consume box-local thermodynamic values.

## Parent Epic Context

Parent epic: E2. This feature is E2-F5. P1 shipped under issue #1203 and issue
#1204 implemented the shared helper plus the minimal condensation/coagulation
runtime migration needed to consume normalized per-box arrays. It depends on
E2-F2 for the environment container schema and E2-F3 for GPU
environment-transfer conventions. Sibling tracks E2-F1 through E2-F4 establish
schema, environment-container, and vapor-pressure foundations; E2-F6 through
E2-F9 can now build on a working shared environment feed path.
