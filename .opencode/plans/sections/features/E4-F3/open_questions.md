# Open Questions

- [ ] Should reusable scratch be exposed as individual keyword-only arrays or a
  dedicated operation-sidecar dataclass?
  - Constraint: it must not modify physical `Warp*Data` container schemas.
- [ ] Which omitted scratch arrays may be allocated for backward compatibility,
  and how will the zero-required-allocation path be made unambiguous?
- [ ] Should the existing `mass_transfer` argument become the total-transfer
  accumulator directly, or should work and output remain separate?
  - Required outcome: returned transfer is the full-call applied total.
- [ ] Can E4-F1 refresh reuse the same `(n_boxes,)` environment-property buffers
  across all four substeps without obscuring validation ordering?
- [ ] What instrumentation is accepted for proving no required allocations and
  no hidden synchronization when all scratch is supplied?
- [ ] Which graph-capture/autodiff claims are proven in E4-F3 versus deferred to
  E4-F6?
- [x] Is adaptive substepping part of E4-F3?
  - Resolved 2026-07-12: No; exactly four fixed iterations are required.
- [x] Are new diagnostics required?
  - Resolved 2026-07-12: No diagnostics were requested.
