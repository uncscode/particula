# Overview

## Problem Statement

E5-F3 through E5-F5 make charged, sedimentation, and turbulent-shear terms
individually executable, but running one stochastic selector per mechanism
would reuse the same particle population, double-count collision opportunities,
and advance RNG state inconsistently. E5-F6 completes parent epic E5's T6 by
composing enabled pair rates before selection.

## Value Proposition

`coagulation_step_gpu` will evaluate every enabled mechanism for a candidate
pair, sum those rates, and compare the sum once against a safe total majorant.
Users gain physically additive two-way and four-way particle-resolved GPU
coagulation while retaining one collision buffer, one per-box RNG stream, one
merge pass, and the existing direct low-level API ownership contract.

## User Stories

- As a simulation author, I want enabled collision mechanisms added for each
  candidate pair so combined physics is not biased by sequential selectors.
- As a GPU caller, I want one RNG acceptance pass and one merge pass so buffer
  identity, conservation, and deterministic seeded behavior remain reviewable.
- As a maintainer, I want an explicit executable combination matrix and proven
  total bound so unsupported requests fail before state or RNG mutation.

Parent context: E5, "GPU Coagulation Physics Expansion and Validation."
Classifier diagnostics: none.
