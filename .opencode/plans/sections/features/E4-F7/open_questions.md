# Open Questions

- [ ] What are the final public numeric mode values, argument names, scratch-buffer names, return fields, and diagnostic units after E4-F1 through E4-F6?
  - Resolve from merged implementation and tests before P1 wording is finalized; do not infer from pre-E4 code.
- [ ] Which exact physics, conservation, and energy tolerances did E4-F6 approve for Warp CPU and optional CUDA?
  - Publish each category separately and cite its focused test.
- [ ] Does E4-F6 define stable `warp`, `gpu_parity`, and `cuda` marker combinations for focused commands?
  - If not, publish direct file commands and describe optional CUDA through the repository's actual skip policy.
- [ ] Should the parity walkthrough remain inside the canonical feature page or become a separate runnable example?
  - Prefer extending the canonical script unless clarity or runtime justifies a second maintained artifact.
- [ ] Does on-device vapor-pressure refresh replace the current `WarpGasData.vapor_pressure` caller guidance or retain it as explicit scratch/helper state?
  - Preserve CPU container ownership and document the final operational distinction from E4-F1.
- [ ] Which graph-capture and autodiff boundaries from E4-F6 are stable enough for the support matrix?
  - Include only evidence-backed behavior and explicit clamp/in-place limitations; otherwise label it experimental or unsupported.
- [x] Should E4-F7 add new runtime diagnostics?
  - Resolved 2026-07-12: No. Diagnostics requested for this track are none; document only upstream-supported outputs.
