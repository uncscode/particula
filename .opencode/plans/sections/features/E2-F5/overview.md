# Overview

## Problem Statement

Existing low-level GPU kernel APIs accept scalar `temperature` and `pressure`
values. That keeps single-box and uniform-environment workflows simple, but it
blocks E2's broader multi-box roadmap because physics kernels cannot consume a
distinct thermodynamic environment per box. Issue #1203 delivered E2-F5-P1 by
publishing the temporary compatibility contract for this migration without yet
implementing per-box environment execution.

## Value Proposition

- Current callers of `condensation_step_gpu` and `coagulation_step_gpu` keep
  working with scalar temperature/pressure inputs.
- Future GPU callers now have a documented keyword-only `environment=` entry
  point plus stable early `ValueError` behavior for mixed or not-yet-supported
  explicit-environment calls.
- Later GPU physics work can depend on a consistent environment feed point
  instead of adding one-off temperature/pressure handling per kernel.

## User Stories

- As an existing GPU API caller, I want scalar temperature and pressure calls to
  continue passing so my current simulations do not break.
- As a multi-box model developer, I want the reserved explicit-environment path
  and ambiguity rule documented now so later phases can add real per-box
  execution without breaking callers.
- As a future physics kernel implementer, I want a stable environment state
  interface so latent heat, parcel expansion, and vapor-pressure updates can
  consume box-local thermodynamic values.

## Parent Epic Context

Parent epic: E2. This feature is E2-F5 and P1 shipped under issue #1203. It
depends on E2-F2 for the environment container schema and E2-F3 for GPU
environment-transfer conventions. Sibling tracks E2-F1 through E2-F4 establish schema,
environment-container, and vapor-pressure foundations; E2-F6 through E2-F9 can
build on the compatibility path created here.
