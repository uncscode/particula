# Open Questions

## Resolved Answers

1. Use `saturation_ratio` as the exact humidity/saturation field name.
2. Require strictly positive pressure for environment state. A zero pressure is
   not valid for the physical per-box environment container.
3. `saturation_ratio` values above `1.0` must be allowed to support
   supersaturation. Validate finite, non-negative values rather than capping at
   unity.
4. Defer a builder until scalar-to-box broadcasting is needed by process
   migration tracks. Issue #1188 confirmed the first CPU container can stay a
   minimal explicit data API.
5. Issue #1189 shipped the package-level `particula.gas.EnvironmentData`
   export together with `n_boxes` and `copy()` semantics. Root-level package
   export remains deferred to downstream API-surface decisions.

## Resolution Path

- Implement the smallest documented API that satisfies E2-F2 and leave builder
  and root-export decisions to downstream migration tracks.
- Preserve the current validation order and error-message specificity as later
  phases document or migrate the shipped convenience API surface.
