# Open Questions

Status: reviewed and answered on 2026-07-08 from repository code, GPU roadmap
guidance, and maintainer confirmation.

## Resolved Decisions

1. Caller-supplied `rng_states` should bypass automatic initialization by
   default. E3-F1 should add an explicit initializer/reset helper so users who
   need repeatable sequences can opt into reset behavior without changing the
   positional `coagulation_step_gpu` call contract.
2. The public quick-start should import low-level step functions from
   `particula.gpu.kernels`. Keep raw kernel internals out of top-level
   `particula.gpu` re-exports unless E3-F4 adds a narrow export contract with
   tests.
3. Mixed-scale rejection sampling is "hardened enough" only after E3-F2 records
   deterministic acceptance diagnostics and either improves acceptance or
   documents a bounded limitation. The threshold should be evidence-based in
   E3-F2 rather than fixed before measurements.
4. The one-thread-per-box decision belongs in the GPU roadmap and, if the
   measured decision is durable, an architecture/ADR note. The roadmap already
   owns GPU limitations and performance boundaries.
5. Device-aware pytest policy should add concrete markers/options in
   `particula/conftest.py`, not documentation only. Current config has only
   `slow`, `performance`, and `benchmark`, while Epic C needs explicit Warp,
   CUDA/parity, and stochastic-test guidance.
6. The latent-heat CPU baseline should include both a minimal deterministic
   single-species case and a small multi-species extension if it remains fast.
   E3-F6 should document the example path, and E3-F7 should encode the stable
   baseline as integration coverage.

## Scheduling And Ownership

- Owner for E3 and E3-F1 through E3-F7: `Gorkowski`.
- Start date for E3 and E3-F1 through E3-F7: `2026-07-08`.
- Target dates remain TBD until implementation issues are generated and sized.

## Residual Issue Placeholders

Phase-level `Issue: TBD` placeholders should remain unassigned until concrete
implementation issues are generated. The corresponding JSON `issue_number`
fields remain `null`; this is expected draft-plan state rather than an open
technical blocker.
