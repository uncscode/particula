# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Buck branch or units diverge from CPU reference | Medium | High | Port exact Kelvin-to-Celsius and piecewise constants; test around 273.15 K | P2 implementer |
| Species modes/parameters are misordered | Medium | High | Validate species count and document positional identity; mixed-species tests | P1 implementer |
| Invalid configuration mutates output before failure | Medium | High | Complete host-side metadata validation before any kernel launch; snapshot tests | P1/P4 implementers |
| Omitted configuration breaks existing callers | Medium | Medium | Resolve explicit fail-early versus named legacy mode; preserve positional API | Feature owner |
| Configuration/output reside on a different Warp device | Low | High | Validate every Warp array against the active gas/environment device | P1 implementer |
| Refresh occurs once while E4-F3 later adds four substeps | Medium | High | Expose an internal callable refresh boundary and require E4-F3 to invoke it per substep | E4-F1/E4-F3 owners |
| GPU transcendental precision differs by backend | Medium | Medium | Use `float64`, explicit tolerances, Warp CPU parity, and optional CUDA coverage | P2/P4 implementers |
| Scope absorbs activity, latent heat, or gas coupling | Medium | Medium | Enforce E4 sibling boundaries and dependency gates in review | Feature owner |
