# Open Questions

1. What exact field name did E2-F1 choose for humidity/saturation state?
   Options include `relative_humidity`, `saturation_ratio`, or a more specific
   thermodynamic field.
2. Should pressure validation allow zero, following the existing pressure mixin,
   or require strictly positive physical pressure for environment state?
3. If relative humidity is used, should values above `1.0` be rejected, or does
   the roadmap require supersaturation support in the initial CPU container?
4. Should a builder be introduced now, or deferred until scalar-to-box
   broadcasting is needed by process migration tracks?
5. Should root-level `particula` exports include `EnvironmentData`, or is
   `particula.gas.EnvironmentData` sufficient for this feature?

## Resolution Path

- Treat E2-F1 as authoritative for schema naming and validation bounds.
- If E2-F1 is ambiguous, choose the smallest documented API that satisfies T2
  and record follow-up work for downstream tracks.
