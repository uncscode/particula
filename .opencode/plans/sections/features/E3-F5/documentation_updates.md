# Documentation Updates

## Required Updates

- `.opencode/guides/testing_guide.md`
  - Added the shipped device-aware Warp pytest policy under the NVIDIA Warp
    Tests section.
  - Records the exact marker names, Warp CPU default behavior,
    CUDA-local/manual semantics, and clean-skip expectations.
  - Records the three validation classes: explicit deterministic
    `rtol`/`atol`, tight conservation checks, and aggregate stochastic
    expectations.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Now records E3-F5 policy as shipped rather than pending.
  - Repeats CUDA local/manual validation expectations and the same three
    parity/conservation/stochastic tolerance classes used in the testing guide.

## Shipped Status

- `E3-F5-P1` did not change external documentation files.
- `E3-F5-P2` standardized helper wording through shared skip-message coverage,
  but did not update plan-facing docs.
- `E3-F5-P3` shipped the intended documentation-only diff in exactly two files:
  `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- The shipped wording now explicitly rejects exact per-seed equality for
  stochastic GPU validation while keeping conservation tolerances tight.
- `E3-F5-P4` was test-only and did not add new external documentation files.
  Its documentation effect is that the policy from P3 is now reflected in the
  actual GPU test surface through module-level Warp marks plus targeted
  `gpu_parity`, `stochastic`, and `cuda` annotations.

## Conditional Updates

- `docs/Features/data-containers-and-gpu-foundations.md`
  - Update only if public user-facing GPU test/run guidance needs clarification.
- Contributor or release docs
  - Add focused release validation commands if a suitable document already
    exists.

## Documentation Content Requirements

- State that CUDA is optional and must skip cleanly when unavailable.
- State that Warp CPU is the default parity backend when Warp is installed.
- State that stochastic tests use aggregate statistical expectations, not exact
  per-seed equality.
- Keep deterministic parity guidance explicit with `rtol`/`atol` examples.
