# Documentation Updates

## P4 Status

Issue #1345 completed the development documentation closeout for P1-P3 without
adding runtime behavior. The documentation records Brownian-only,
charged-only, and canonical Brownian-plus-charged particle-resolved support;
caller-owned fp64 charge, collision buffers, and RNG state; one-pass bounded
majorants; and Warp CPU baseline with optional cleanly skipped CUDA evidence.

The canonical direct coagulation example and final support table remain E5-F9
work. This closeout does not claim high-level runnable support, exact
stochastic parity, unsupported charged variants, or mandatory CUDA.

Validation included Markdown/link and import/reference verification plus the
focused warning-clean coagulation regression. No executable snippet was added.
