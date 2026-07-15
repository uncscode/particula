# Open Questions

- [x] What names, values, fields, and units does E4-F7 publish?
  - Resolved 2026-07-13: copy final public names and integer constants from the
    merged E4-F1 through E4-F6 APIs/tests; do not freeze speculative spellings.
    Document mass transfer in kg and whole-call energy in J with all array
    shapes and ownership.
- [x] Which tolerances does E4-F7 publish?
  - Resolved 2026-07-13: publish formula parity, coupled-physics parity,
    conservation, and energy identity as separate categories, including any
    F6-approved device/case exceptions and a focused test reference for each.
- [x] Which focused marker commands are stable?
  - Resolved 2026-07-15 by Issue #1316: publish the direct-file Warp CPU
    baseline commands and the optional/local CUDA command
    `pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror`.
    CUDA is additive and skips cleanly when unavailable.
- [x] Is an independent CPU/Warp parity walkthrough delivered?
  - Resolved 2026-07-15: No. The canonical direct-kernel quick-start documents
    explicit conversion and execution only; it is not an independent CPU/Warp
    parity comparison. The **CPU/Warp parity-walkthrough follow-up** remains
    deferred and must separately establish physics, conservation, and energy
    tolerances before it can be published.
- [x] How does on-device refresh affect `WarpGasData.vapor_pressure` guidance?
  - Resolved 2026-07-13: retain it as derived mutable GPU helper/scratch state.
    Supported E4 models refresh it on device; explicit caller values are only
    for a named static/legacy mode. CPU ownership remains unchanged.
- [x] Which graph-capture and autodiff boundaries are published?
  - Resolved 2026-07-13: publish graph capture only for named Warp/CUDA versions
    with successful F6 capture/replay evidence. Label autodiff experimental and
    smooth-interior-only; clamps, inventory gates, and in-place mutation remain
    unsupported.
- [x] Should E4-F7 add new runtime diagnostics?
  - Resolved 2026-07-12: No. Diagnostics requested for this track are none; document only upstream-supported outputs.
