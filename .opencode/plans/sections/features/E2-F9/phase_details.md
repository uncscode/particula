# E2-F9 Phase Details

- [ ] **E2-F9-P1:** Foundation guide for data containers, shapes, transfers, and limitations
  - Issue: #1222 | Size: S | Status: implementation landed on branch; metadata still pending update
  - Goal: Publish the canonical guide tying container schemas, shape
    conventions, GPU transfer caveats, and current limitations together.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/index.md`, the minimum supporting updates to
    `docs/Features/particle-data-migration.md`, and possible supporting
    discoverability updates such as `docs/index.md`.
  - Branch outcome: the canonical guide was added, the Features index now links
    to it, and the migration guide now points readers back to the canonical
    contract page.
  - Tests: Documentation validation should focus on `mkdocs build --strict` and
    doc-snippet import accuracy for the shipped public APIs.

- [ ] **E2-F9-P2:** Minimal data-container and GPU transfer examples with guards
  - Issue: #1223 | Size: S | Status: implementation landed on branch; metadata still pending update
  - Goal: Add a small runnable example that shows documented container usage and
    optional Warp transfers without requiring CUDA for the default path.
  - Files: `docs/Examples/data_containers_and_gpu_foundations.py`,
    `docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py`,
    `docs/Examples/Data_Containers/index.md`, `docs/Examples/index.md`, and
    `particula/gpu/tests/data_containers_example_test.py`.
  - Branch outcome: the top-level wrapper forwards to the topic-directory
    implementation, the rendered examples page documents the canonical run
    command, Warp transfers stay behind `WARP_AVAILABLE` on `device="cpu"`, and
    smoke tests cover both the CPU-only success path and the optional Warp path.
  - Tests: `python docs/Examples/data_containers_and_gpu_foundations.py` and
    `pytest particula/gpu/tests/data_containers_example_test.py -q`.

- [ ] **E2-F9-P3:** Roadmap handoff links, docs index wiring, and validation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Link roadmap dependencies, update development docs/index wiring, and
    validate that docs/examples are discoverable by users and planner agents.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/warp-autodiff-limitations.md`, feature/example
    indexes, and any generated docs validation notes kept with the final docs
    update PR description rather than a new repo artifact.
  - Tests: Documentation link check, mkdocs build if available, and examples
    smoke validation.

## Phase Ordering Notes

- P1 should wait until the shipped container, transfer, and support-boundary
  contracts from E2-F2 through E2-F5 and E2-F8 are stable enough to document.
- P2 should follow P1 so examples teach the same API and limitation language as
  the foundation guide.
- P3 is the final handoff step and should validate links only after the guide and
  examples from P1 and P2 are in place.
