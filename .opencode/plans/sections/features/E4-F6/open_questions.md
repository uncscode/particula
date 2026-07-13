# Open Questions

- [ ] What final explicit `rtol`/`atol` applies to each combined-physics parity
  case after E4-F1 through E4-F5 land? The stiffness-study `5e-2` bound is not
  automatically acceptable for production qualification.
- [ ] Does the final E4-F3/E4-F5 API expose every scratch array needed to avoid
  allocation during graph capture, and how is completeness validated?
- [ ] Which Warp versions/devices support condensation graph capture in CI, and
  what precise skip reason is acceptable elsewhere?
- [ ] Which smooth-interior kernel slice can be differentiated without crossing
  inventory/mass clamps or unsupported in-place accesses?
- [ ] Should strict conservation use only `rtol=1e-12`, or pair it with a
  scale-derived `atol` for near-zero species inventories?
- [ ] Which issue 1272 document is canonical for the final evidence matrix?

Diagnostics are intentionally **none**; questions must not introduce a public
diagnostics surface merely to make tests convenient.
