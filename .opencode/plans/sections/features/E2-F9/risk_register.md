# E2-F9 Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Docs describe planned `EnvironmentData` as implemented. | Users and planners assume unavailable APIs. | Medium | Verify source before writing; label missing environment container as planned. | Implementer |
| GPU examples require Warp or CUDA in default validation. | CI/docs validation becomes brittle. | Medium | Guard with `WARP_AVAILABLE` and keep CPU/default paths runnable. | Implementer |
| Schema drift details become stale. | Users lose data such as gas names or vapor pressure during transfers. | Medium | Source examples from `particula.gpu.conversion` and document exact caveats. | Implementer |
| Docs duplicate roadmap content excessively. | Maintenance burden increases. | Low | Summarize current support boundaries and link canonical roadmap pages. | Docs reviewer |
| Examples imply full GPU-resident simulation support. | Mis-set expectations for downstream work. | Medium | State explicit transfer boundaries and mark end-to-end GPU resident flows as future work. | Implementer |
| Notebook pairing is missed. | Docs examples drift between `.py` and `.ipynb`. | Low | Follow repository Jupytext workflow when notebooks are added. | Implementer |
