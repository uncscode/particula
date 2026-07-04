# Open Questions

## Resolved Answers

1. Use `saturation_ratio` as the exact humidity/saturation field name.
2. Require strictly positive pressure for environment state. A zero pressure is
   not valid for the physical per-box environment container.
3. `saturation_ratio` values above `1.0` must be allowed to support
   supersaturation. Validate finite, non-negative values rather than capping at
   unity.
4. Defer a builder until scalar-to-box broadcasting is needed by process
   migration tracks. The first CPU container should be a minimal explicit data
   API.
5. Export `EnvironmentData` from `particula.gas` first. Defer root-level
   `particula.EnvironmentData` until the API has broader public usage.

## Resolution Path

- Implement the smallest documented API that satisfies E2-F2 and leave builder
  and root-export decisions to downstream migration tracks.
