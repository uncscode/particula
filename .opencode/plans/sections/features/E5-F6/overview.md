# Overview

## Problem Statement

E5-F3 through E5-F5 make charged, sedimentation, and turbulent-shear terms
individually executable, but running one stochastic selector per mechanism
would reuse the same particle population, double-count collision opportunities,
and advance RNG state inconsistently. E5-F6 completes parent epic E5's T6 by
composing enabled pair rates before selection.

## Value Proposition

`coagulation_step_gpu` evaluates every enabled mechanism for a candidate
pair, sum those rates, and compare the sum once against a safe total majorant.
Users gain physically additive two-way and four-way particle-resolved GPU
coagulation while retaining one collision buffer, one per-box RNG stream, one
merge pass, and the existing direct low-level API ownership contract.

## Shipped P1--P3 Boundary

Issue #1357 delivered the recognition and atomic-preflight boundary. A private
immutable table recognizes the four singleton,
six unordered pair, and four-term masks. The four three-term masks reject
before particle access. The rejected three-way masks run only their enabled
turbulent, charged, and sedimentation read-only validation, then raise
`ValueError("Additive coagulation execution is deferred.")` before runtime
normalization, output/RNG work, kernels, or caller-state mutation.

## Implemented P2 Boundary

Issue #1358 delivered private fp64 aggregation primitives in
`particula/gpu/kernels/coagulation.py`. Enabled Brownian, charged hard-sphere,
SP2016 sedimentation, and ST1956 turbulent-shear component rates and majorants
are summed through checked, fail-closed addition for every P1-recognized
two-way and four-way mask. Invalid, nonpositive, or overflowed aggregates
become zero. Candidate acceptance now accepts only finite positive ratios and
permits a rate-over-majorant roundoff excess of at most eight fp64 ULPs, mapped
to exactly `1.0`; material violations cannot advance RNG or mutate selector
state. P3 shipped approved-mask execution through the shared selector/apply
path. Existing public-path tests cover approved masks, ownership, conservation,
selector behavior, deferred masks, and persistent RNG behavior.

## User Stories

- As a simulation author, I want enabled collision mechanisms added for each
  candidate pair so combined physics is not biased by sequential selectors.
- As a GPU caller, I want one RNG acceptance pass and one merge pass so buffer
  identity, conservation, and deterministic seeded behavior remain reviewable.
- As a maintainer, I want an explicit executable combination matrix and proven
  total bound so unsupported requests fail before state or RNG mutation.

Parent context: E5, "GPU Coagulation Physics Expansion and Validation."
Classifier diagnostics: none.
