# Open Questions

- [x] Should reusable scratch be exposed as individual keyword-only arrays or a
  dedicated operation-sidecar dataclass?
  - Resolved 2026-07-13: use one typed operation-sidecar dataclass. Do not
    modify physical `Warp*Data` container schemas.
- [x] Which omitted scratch arrays may be allocated for compatibility?
  - Resolved 2026-07-13: omission of the entire scratch sidecar keeps the
    convenience allocation path. A complete validated sidecar provides the
    zero-required-allocation path; reject partial sidecars rather than
    allocating unpredictably.
- [x] Should `mass_transfer` be both work storage and the total accumulator?
  - Resolved 2026-07-13: keep work and output separate. Zero `mass_transfer` as
    the whole-call applied-transfer accumulator and use a distinct substep work
    buffer.
- [x] Can E4-F1 refresh reuse the same `(n_boxes,)` environment-property
  buffers across all four substeps?
  - Resolved 2026-07-13: yes. Complete structural validation before the loop,
    then refill the fixed-shape viscosity/mean-free-path buffers after each
    temperature refresh.
- [x] What instrumentation proves no required allocations or hidden
  synchronization with complete scratch?
  - Resolved 2026-07-13: warm compilation, then count/monkeypatch Warp
    allocation and synchronization entry points during repeated calls while
    also asserting buffer identity and shape stability.
- [x] Which graph-capture/autodiff claims belong to E4-F3 versus E4-F6?
  - Resolved 2026-07-13: E4-F3 proves four fixed iterations, stable shapes,
    reset semantics, and repeated scratch reuse. E4-F6 owns actual graph
    capture/replay and bounded autodiff evidence.
- [x] Is adaptive substepping part of E4-F3?
  - Resolved 2026-07-12: No; exactly four fixed iterations are required.
- [x] Are new diagnostics required?
  - Resolved 2026-07-12: No diagnostics were requested.
