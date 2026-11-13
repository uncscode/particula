# Retired documentation for v0.3.0

Original paths are preserved below this directory. These sources are excluded
from publication, example discovery, and runtime imports.

| Original source | Reason | Replacement |
| --- | --- | --- |
| `docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py` | Redundant forwarding module; no scientific implementation | `docs/examples/fundamentals/data_containers.py` |

Source revision before the example move: `a16861c91`. Additional historical
retirements and provenance are recorded in phase 4.

## Narrative retirement

Source revision: `7b064295037c3c9a137f5dd13c03f4a40da3f11b`. Original files are preserved without edits.

| Original source | Reason | Current owner |
| --- | --- | --- |
| `docs/Theory/Accelerating_Python/index.md` | Outdated proposal/provider catalog | `docs/guide/gpu/execution.md` |
| `docs/Theory/Accelerating_Python/Details/Taichi_Exploration.ipynb` | Outdated proposal/provider catalog | `docs/guide/gpu/execution.md` |
| `docs/Theory/Accelerating_Python/Details/One-Line_Vision.md` | Outdated proposal/provider catalog | `docs/guide/gpu/execution.md` |
| `docs/Theory/OpenAI_Models/index.md` | Outdated proposal/provider catalog | `docs/contribute/ai-assistants/index.md` |
| `docs/Theory/OpenAI_Models/Models/Agents.md` | Outdated proposal/provider catalog | `docs/contribute/ai-assistants/index.md` |
| `docs/Theory/OpenAI_Models/Models/Chat.md` | Outdated proposal/provider catalog | `docs/contribute/ai-assistants/index.md` |
| `docs/Theory/OpenAI_Models/Models/Reasoning.md` | Outdated proposal/provider catalog | `docs/contribute/ai-assistants/index.md` |
| `docs/Theory/Technical/index.md` | Redundant hierarchy/catalog merged into owning index | `docs/theory/index.md` |
| `docs/Features/Roadmap/data-oriented-gpu.md` | Completed/mixed roadmap; current contracts and evidence extracted | `docs/contribute/roadmap.md` |
| `docs/Features/Roadmap/graph-capture-closeout.md` | Completed/mixed roadmap; current contracts and evidence extracted | `docs/contribute/validation/gpu/graph-capture-qualification.md` |
| `docs/Features/Roadmap/index.md` | Completed/mixed roadmap; current contracts and evidence extracted | `docs/contribute/roadmap.md` |
| `docs/guide/processes/catalog.md` | Redundant hierarchy/catalog merged into owning index | `docs/guide/processes/index.md` |

The GPU roadmap ownership policy is in `docs/guide/data-model/field-ownership.md`;
historical timings are in `docs/contribute/validation/gpu/coagulation-benchmarks.md`.
Graph-capture qualification remains blocked and v0.3.0 release gates remain open.
Provider catalogs contain no unique current Particula setup steps; current assistant
instructions already own that workflow. Taichi/vision material describes exploratory
proposals, not the supported Warp execution path. No supporting local assets were referenced.
