# Open Questions

- [ ] Which effective surface-tension mode is the minimum supported E4-F2 contract?
- [ ] Should numeric thermodynamic configuration live in a dedicated Warp struct or validated parallel arrays?
- [ ] Does the public call return only whole-call energy, or also an explicitly named final-substep diagnostic?
- [ ] What deterministic allocation rule resolves simultaneous particle demand when gas inventory is limiting?
- [ ] Which quantities, if any, are differentiable across clamps and inventory gates in the bounded autodiff claim?
- [ ] What numerical tolerances are accepted for each physics mode on Warp CPU versus CUDA?

These questions do not alter the seven-track ordering. E4-F1 must resolve
configuration layout; E4-F4 must resolve energy semantics; E4-F5 must resolve
inventory allocation; E4-F6 records accepted tolerances and autodiff bounds.

- [ ] Should the M/L labels in `milestones_timeline.md` denote aggregate
  workstream scope rather than implementation-phase size? (reviewer: plan-review-sizing)
  - Open: E4-F1 through E4-F6 already contain XS/S implementation phases, so
    this distinction determines whether the milestone labels require splitting.
