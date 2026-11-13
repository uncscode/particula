# E2-F6 Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Study helpers accidentally change production dtype defaults. | High: downstream schema behavior changes before evidence is accepted. | Keep candidates isolated; add assertions that `ParticleData`, builder coercion, Warp structs, and conversions remain `fp64`. | Implementer |
| Current GPU condensation path is mistaken for a conservation reference. | High: recommendation may be based on clamping rather than conserved gas/particle mass. | Use CPU conservation-limited functions as reference and report GPU clamping separately. | Numerical reviewer |
| NPF-to-droplet cases underrepresent dynamic range. | Medium: recommendation may miss small-particle fidelity loss. | Include explicit NPF/small-particle and droplet coexistence cases with documented mass/radius ranges. | Implementer |
| CUDA is unavailable during implementation. | Medium: throughput evidence may be incomplete. | Make GPU benchmarks skip-safe and still report CPU/memory/fidelity evidence; document missing runtime environment. | Implementer |
| Report recommendation is ambiguous. | High: future schema work cannot use the feature as a gate. | Require a final recommendation section with accepted, rejected, and follow-up options. | Maintainer |
| Representation alternatives are too large for this feature. | Medium: scope creep delays report. | Evaluate alternatives at study/projection level only and defer production implementation to later plans. | Planner |
