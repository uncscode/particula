# Open Questions

1. Should caller-supplied `rng_states` always bypass initialization, or should
   the API provide an explicit initialize/reset helper for users who want repeat
   sequences?
2. Is documentation of `particula.gpu.kernels` sufficient, or should selected
   kernel entry points also be re-exported from `particula.gpu`?
3. What acceptance-rate or throughput threshold defines mixed-scale rejection
   sampling as "hardened enough" for Epic C?
4. Where should the one-thread-per-box decision live: roadmap section,
   architecture guide, ADR, or all of the above?
5. Should device-aware pytest policy be documentation-only, or should it add
   new markers/options in `particula/conftest.py`?
6. Which latent-heat scenario is the best long-term CPU baseline for future GPU
   parity: a minimal single-species case, a multi-species case, or both?

## Proposed Resolution Path

- Resolve questions 1, 2, and 5 before or during E3-F1/E3-F4/E3-F5 because
  they shape API and test contracts.
- Resolve question 3 during E3-F2 with measured evidence.
- Resolve question 4 during E3-F3 based on repository documentation norms.
- Resolve question 6 during E3-F6, then encode it in E3-F7 tests.

## Completeness Follow-up

- [ ] Who owns Epic C execution, and what `start_date`/`target_date` values
  should be recorded for E3 and child plans E3-F1 through E3-F7? (reviewer:
  plan-review-completeness)
  - Open: Current metadata leaves `owners`, `start_date`, and `target_date`
    empty across the in-scope plans, which blocks complete scheduling and
    accountability review.
