# Documentation Updates

## Required Updates

- `.opencode/guides/testing_guide.md`
  - Added the shipped device-aware Warp pytest policy under the NVIDIA Warp
    Tests section plus the final release-validation command/troubleshooting
    guidance for CUDA-optional workflows.
  - Records the exact marker names, Warp CPU baseline behavior, optional/local/
    manual CUDA semantics, clean-skip expectations for missing Warp and missing
    CUDA, and the separation between standard validation and opt-in benchmark
    evidence.
  - Records the three validation classes: explicit deterministic
    `rtol`/`atol`, tight conservation checks, and aggregate stochastic
    expectations.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Now records E3-F5 policy as shipped rather than pending and mirrors the
    release-validation wording from the testing guide.
  - Repeats Warp CPU baseline validation, CUDA local/manual-only expectations,
    expected skip behavior in CPU-only or standard CI environments, and the
    same parity/conservation/stochastic tolerance classes used in the testing
    guide.

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
- `E3-F5-P5` shipped as a documentation-only release-validation clarification in
  those same two canonical files.
- `docs/contribute/CONTRIBUTING.md` was intentionally left unchanged because the
  testing guide and roadmap remain the canonical policy homes and this phase did
  not broaden into a contributor-doc refresh.

## Conditional Updates

- `docs/Features/data-containers-and-gpu-foundations.md`
  - Update only if public user-facing GPU test/run guidance needs clarification.
- Contributor or release docs
  - No broader contributor-doc update shipped for P5; keep the testing guide and
    roadmap as the canonical sources unless a later issue needs a short pointer.

## Documentation Content Requirements

- State that CUDA is optional and must skip cleanly when unavailable.
- State that Warp CPU is the default parity backend when Warp is installed.
- State that missing Warp is also an expected skip path for guarded suites.
- State that stochastic tests use aggregate statistical expectations, not exact
  per-seed equality.
- Keep deterministic parity guidance explicit with `rtol`/`atol` examples.
- Keep benchmark validation explicitly opt-in and separate from standard release
  validation commands.
