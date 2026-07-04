# Open Questions

- What exact name and fields does E2-F2 provide for the environment container
  (`EnvironmentData`, `WarpEnvironmentData`, or another name)?
- Should explicit environment plus scalar temperature/pressure be rejected as
  ambiguous, or should explicit environment take precedence?
- Should condensation precompute per-box dynamic viscosity and mean free path on
  the host, or compute them inside Warp kernels from per-box temperature and
  pressure?
- Are humidity or saturation fields required in the first migration path, or is
  temperature/pressure sufficient for T5?
- Should wrapper APIs be public exports immediately, or should scalar functions
  remain the only public entry points with environment support added internally?
