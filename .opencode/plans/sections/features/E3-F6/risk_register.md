## Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Example only configures latent heat and does not run condensation | High | Medium | Require `MassCondensation.execute()` and read `last_latent_heat_energy` after each step | Implementer |
| Example conditions produce zero or confusing energy diagnostics | Medium | Medium | Choose supersaturation/particle setup that produces visible mass transfer; print mass and gas diagnostics | Implementer |
| Docs imply GPU latent-heat parity | High | Low | Add explicit CPU-only wording and grep touched files for GPU/Warp/CUDA claims | Implementer/Reviewer |
| Notebook falls out of sync with source | Medium | Medium | Edit `.py` first, run validate_notebook sync, and commit both files | Implementer |
| Factory/builder API assumptions differ from production code | Medium | Low | Use existing tests and public exports as authority; prefer factory examples from tests | Implementer |
| Example is too computationally heavy for docs execution | Medium | Low | Keep particle/bin counts and timesteps small; use deterministic CPU-only setup | Implementer |

### Current Overall Risk

Low to medium. The feature primarily composes existing documented APIs, but it
must be careful to demonstrate the actual runnable workflow and avoid expanding
scope into GPU or numerical model changes.
