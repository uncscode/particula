# Documentation refactor plan

Status: **all five phases implemented and locally validated; single-PR delivery follows**.
Prepared 2026-11-12 from local `main` at `5448f2333`.

## Objective and scope

Target the **v0.3.0 release documentation**. Keep published material focused on
current use, useful migration guidance, and evidence supporting current claims.
Outdated material belongs in the repository-root `.trash/` for posterity, not
in a published archive. This organization does not itself approve a release.

A new scientist should be able to install the matching version, understand the
native data model, and run a small CPU example without first reading GPU
ownership contracts or development milestones. An experienced user should be
able to find a process guide, its equations, its examples, and its API reference
from one topic entry point.

This plan catalogs `docs/Features`, `docs/Examples`, `docs/Theory`, and
`docs/migration`, with the home page, navigation, contributor guidance, and
example infrastructure included where they affect organization. The catalog below records the initial proposal; implementation and validation
are recorded at the end. Production APIs and scientific behavior are unchanged.

The inventory is a dated planning snapshot, not a second permanent catalog to
maintain alongside the site. The eventual topic indexes should be the living
catalog. Keep this plan outside the public docs navigation.

## Findings from the current checkout

| Area | Tracked content | Organization problem |
| --- | --- | --- |
| Features | 29 Markdown pages | User guidance, low-level contracts, migration, studies, and release records share one section. |
| Examples | 17 Markdown pages, 59 Python files, 46 notebooks | Installation is inside Examples; ten scripts sit at its root; GPU examples are buried in the data-container card. |
| Theory | 24 Markdown pages, 10 notebooks, 1 Python file | Aerosol science sits beside software patterns, provider/model descriptions, Warp tutorials, and historical acceleration experiments. |
| Migration | 1 Markdown redirect | The visible folder is a redirect to five substantive pages under Features. |
| Total in scope | 187 tracked files | Paired formats and indexes are included; this is not 187 independent tutorials. |

All 46 Examples notebooks have matching Python files. The Theory benchmark
notebook also has a matching Python file. The catalog below groups these pairs,
giving **140 catalog entries** covering all 187 files.

Additional observations:

- Neither `mkdocs.yml` nor `mkdocs-full.yml` declares an explicit `nav`.
  The full build adds generated API pages and notebook rendering; navigation
  follows discovered content rather than a designed reader journey.
- The home page is 332 lines and includes detailed GPU masks, array ownership,
  validation, and lifecycle boundaries. Its first-use path competes with
  reference material.
- `Features/data-containers-and-gpu-foundations.md` is 1,827 lines;
  `Features/condensation_strategy_system.md` is 1,094;
  `Features/coagulation_strategy_system.md` is 789;
  `Features/Roadmap/data-oriented-gpu.md` is 2,237. These are candidates for
  separation by purpose, not arbitrary page-length limits.
- The Features landing page does not link directly to
  `native_scientific_processes.md`, `gpu_graph_capture.md`,
  `gpu_graph_capture_performance.md`, or `resident_benchmark_memory_budget.md`.
  They are not necessarily orphaned elsewhere.
- Names such as `Gas_Species.py`, `AtmosphereTutorial.py`, and
  `Particle_Representation_Tutorial.py` retain historical vocabulary even
  though the existing migration inventory records native rewrites. Their names
  are not evidence that their content can be deleted.
- `Examples/Data_Containers/data_containers_and_gpu_foundations.py` is a
  forwarding module to the root example, not a second scientific implementation.
- The working directory contains backup notebooks, bytecode caches, and
  `.assets/temp` generated copies. No `.bak` files or `.assets/temp` files are
  tracked. Treat local clutter separately from repository source removal.
- The migration index says final readiness validation is pending, while the
  native closeout says implementation is validated with review/release approval
  pending. Consolidation needs revision-specific status reconciliation; this
  plan does not decide release readiness.

Evidence: current files, notebook metadata/headings, the docs build configs and
workflow, the API generator and notebook routing hook, and the existing
E9-M5 example inventory. Earlier execution statements in that inventory are
historical evidence only. This audit has not rerun examples, validated all
scientific claims, checked external URLs, or built the site.

## Proposed navigation and folder structure

Use an explicit reader-oriented navigation order. Introduce lowercase,
hyphenated paths in staged moves; preserve the generated `API/` convention.
The agreed top-level sections are **Guide, Examples, Theory, API, and
Contribute**. Use Guide rather than Features to emphasize how to use Particula.
Installation and migration belong under Guide; current assistant setup belongs
under Contribute. Outdated material moves outside the site into `.trash/`.
Preserve useful old routes for retained content; retired content does not need
a permanent public compatibility page.

```text
docs/
  index.md                         # concise project overview and entry paths
  guide/
    index.md                       # start here and choose a learning path
    getting-started/
      index.md                     # installation -> first run -> next step
      installation/                # pip, uv, conda, new-to-Python help
      first-simulation.md           # one small supported native CPU workflow
    migration/
      index.md                     # version chooser, currently v0.3.0
      v0.3.0/                      # old -> new, troubleshooting, compatibility
    ai-assistants/                 # supported assistant setup and workflows
    data-model/                    # construction, shapes, units, ownership
    processes/                     # activity, condensation, coagulation, etc.
    gpu/                           # selection, transfers, slots, resident use
      resident/                    # lifecycle/checkpoints, capture runbook
  examples/
    index.md                       # small beginner path before full studies
    fundamentals/                  # native aerosol, gas, particles, state
    thermodynamics/                # activity, equilibria, surface properties
    processes/                     # condensation, coagulation, wall loss, etc.
    gpu/
      direct/
      resident/
    simulations/                   # full scientific workflows, runtime labels
  theory/
    index.md                       # scientific assumptions and equations
    thermodynamics/
    dynamics/
      cloud-droplet-coagulation/
    properties/
  API/                             # generated; public navigation via index
  contribute/
    index.md                       # retain existing contribution workflows
    architecture/                  # code concepts and execution internals
    validation/                    # numeric studies and GPU evidence records
    roadmap/                       # active and pending work only
    documentation.md               # placement, ownership, and review rules
```

Top-level tabs, in order: **Guide, Examples, Theory, API, Contribute**. The
project overview remains the home link, with Start here pointing into Guide.
Getting started, Migration, and AI assistants are Guide subsections. There is no public Archive section. Prefer two
or three meaningful navigation levels, with topic indexes for deeper material;
use descriptive group names instead of `Details`, `Notebooks`, or `Technical`.

A topic owns one canonical contract. Its guide links to scientific theory,
examples, API symbols, and validation evidence instead of restating their
contents. Shared data units and shape semantics belong to Data model. Execution
ownership belongs to GPU guides. Current release gates and supporting evidence
belong to validation records; completed roadmap history belongs in `.trash`.
Keep the CPU/direct-Warp/resident support distinctions visible where needed.

### Starting paths

1. **New user:** installation/version note -> first CPU simulation -> native
   aerosol/data model -> chosen process -> related theory.
2. **Migrating user:** v0.3.0 migration overview -> old/new API table -> changed
   construction and access -> process changes -> troubleshooting.
3. **GPU user:** CPU/native prerequisites -> explicit transfers/direct example
   -> capability selection -> resident lifecycle -> CUDA capture when applicable.
4. **Contributor:** repository/code map -> architecture -> contributing workflow
   -> relevant validation record and current roadmap.

The first-simulation page should reuse the existing tested quick-start material
and its scientific caveats. Do not invent another independent tutorial or imply
that an older published wheel contains the development API.

## Consolidation and removal decisions

### Agreed v0.3.0 retirement policy

- Move outdated documents to `.trash/docs-v0.3.0/<original-repository-path>`;
  for example, `docs/Theory/OpenAI_Models/index.md` becomes
  `.trash/docs-v0.3.0/docs/Theory/OpenAI_Models/index.md`. Preserve original
  names and relative structure; keep notebook/source pairs and necessary local
  assets together. Do not overwrite existing trash entries.
- Track these moves in Git. Use one `.trash/docs-v0.3.0/README.md` with the
  source revision, retirement reason, and replacement where one exists. This
  is a provenance record, not another maintained public documentation section.
- Retired material is outside the MkDocs source tree, navigation, and search.
  Update active links/tests/tools to current replacements; do not add `.trash`
  to active runtime imports, example discovery, or documentation generation.
- Retire obsolete API instructions, abandoned proposals, generic provider-model
  catalogs, superseded milestone narrative, and verified redundant pages or
  wrappers. First extract any unique guidance still needed for v0.3.0.
- Remove **all completed roadmaps and completed milestone/closeout narratives**
  from published docs and move their originals to `.trash/docs-v0.3.0/`.
  Completion includes explicit Shipped/Completed status; do not keep a public
  completed-roadmap archive or completed-epic status tables.
- For mixed roadmaps, extract active/pending work into a short current roadmap,
  extract still-authoritative contracts/evidence into guides or validation
  references, and retire the original. The current GPU roadmap lists Epics A–H
  as Shipped and Epic I as Pending: retire A–H planning narrative and preserve
  I as pending, without advancing its status.
- A completed plan can still have missing measurements. Preserve such gaps as
  current validation limitations, not as a reason to retain the completed
  roadmap. Graph-capture measurement evidence remains unavailable despite its
  Shipped plan status; retain that fact and its provenance in validation docs.
- The native v0.3.0 closeout currently records pending maintainer/release approval.
  Keep those open gates and revision-specific evidence under validation until
  resolved; do not infer completion or release approval from this cleanup.
- Keep useful old/new migration comparisons, distinct scientific examples, and
  evidence supporting current numerical or release claims. Age alone does not
  make a scientific result or unresolved evidence boundary outdated.
- Redirect old routes where retained content has a meaningful replacement.
  Routes for retired-only material may disappear; do not create blanket home
  redirects or maintain a public archive solely to keep every old URL alive.


| Candidate | Proposed treatment | Condition before implementation |
| --- | --- | --- |
| Container/GPU foundations mega-page | Split data model, transfers, slot semantics, and process-specific GPU contracts; retain a short old-path landing/redirect | Map every existing heading to an owner; preserve incoming fragment links and support boundaries. |
| Condensation/coagulation guides | Separate practical usage from detailed direct-GPU and validation material | Keep a single process entry point; preserve scientific limitations and tested snippets. |
| Native scientific process calls | Make it the process overview; consolidate repeated construction and ownership prose by linking | Keep unique native species-map and normalization contracts. |
| Roadmap index and data-oriented roadmap | Publish active/pending work only; move originals and all completed roadmap narrative to `.trash` | Extract current contracts, decisions, and evidence into guides/validation; remove completed-epic tables from public navigation and pages. |
| Migration redirect plus five feature pages | One versioned migration section under Guide; redirects at both old routes | Extract reusable current-API instruction to Data model; leave old/new comparisons in Migration. |
| Topic-local data-container forwarding script | Move redundant forwarding module to `.trash` after consolidating to one canonical example | Update callers, tests, and published commands; preserve any still-required compatibility until callers are migrated. |
| Activity/Equilibria tutorials spread across three topics | Co-locate, then review overlap | Similar subject matter is not duplicate content; compare learning objectives and scientific outputs before merging. |
| One-Line Vision and Taichi exploration | Move to `.trash` as outdated proposals/experiments | Preserve original source and supporting assets; replace active links with current GPU guidance. |
| OpenAI model catalog | Move to `.trash`; link current assistant instructions | Extract any unique, still-current Particula workflow guidance into Guide → AI assistants first. |
| Standalone Warp teaching examples | Move to contributor architecture/learning material | Keep explicit teaching-only status; do not mix them with validated Particula process examples. |
| Autodiff limitations | Keep as contributor design/reference material | It informs future work and is not proof of shipped differentiability. |
| Precision, stiffness, and coagulation validation studies | Keep as scientific validation/reference, not roadmap pages | Preserve evidence supporting current numerical claims and decisions. |
| Completed capture closeout and other completed roadmap records | Move originals to `.trash`; extract current evidence and limitations into validation pages | Preserve provenance and missing-measurement status; no public completed-roadmap archive. |
| Native migration release gates | Keep unresolved gates and evidence under validation | Current source still records pending maintainer/release approval; do not mark it complete. |
| `.py` / `.ipynb` pairs | Keep together | Pairing and retained outputs are required by the current example workflow. |
| Local backups/caches/generated copies | Separate optional local cleanup | Verify exact ignored paths and reproducibility; no deletion is part of this plan. |

The retirement policy is agreed. Exact overlap and obsolescence determinations
remain implementation checks: consolidate unique current content before moving
superseded originals to `.trash`. No files were retired during planning. Completed retirements are recorded below.

## Implementation sequence and completion gates

### Phase 1 — Navigation and orientation without physical moves

- Define exactly five top-level sections in the base/full configurations:
  Guide, Examples, Theory, API, and Contribute. Nest getting started and migration
  under Guide, and current assistant setup under Contribute. Share the
  common structure without exposing absent generated pages in the base build.
  Ensure generated API pages remain reachable from the API index in the full
  build; do not accidentally omit notebook pages or silently strand content.
- Rework the home page and existing section indexes around the four starting
  paths. Move detailed home-page reference text only after confirming its
  canonical destination contains the same contract.
- Group existing feature pages under Data model, Processes, GPU, and contributor
  records in navigation. Group GPU examples separately from beginner containers.
- Add concise purpose, prerequisites, and development/release context to entry
  pages. Label examples by CPU, optional Warp CPU/CUDA, or native CUDA only;
  distinguish tutorials, full simulations, and validation demonstrations.
- Gate: a newcomer can reach installation, a first native CPU workflow, a
  process example, and migration help without traversing a roadmap. Review both
  desktop and narrow navigation; all existing content stays reachable.

### Phase 2 — Migrate narrative documents in topic-sized changes

- Finalize an old-path/fragment -> new-path/fragment mapping before moving a
  group. Use the catalog below as the proposed destination inventory.
- Move migration, scientific theory, developer material, and guides separately.
  Relocate the existing assistant instructions under Contribute, retaining
  their content and compatibility routes; this does not expand the four-section
  catalog into a new assistant-content audit.
  Rewrite relative links, image/data paths, navigation entries, source URLs,
  contribution instructions, and current documentation references together.
- Split the large foundations/process pages by audience and authority. Replace
  mixed roadmaps with active/pending work only, extracting current contracts and
  evidence before retiring the originals. Completed roadmap content is not
  relocated to another public roadmap folder.
- Gate: retained content has tested old-route redirects and useful fragment
  destinations. Retired-only routes may disappear under the agreed policy;
  remove their active links or point to a substantive current replacement.
  Update active callers and tests; preserve historical planning records and
  execution logs with a relocation crosswalk rather than rewriting evidence.

### Phase 3 — Move examples and their dependencies

- Move notebook/script pairs together; group by learning objective. Use native
  names for rewritten tutorials with legacy filenames. Preserve code, plots,
  input datasets, units, outputs, and long-workload scope during pure moves.
- Update file-relative loaders, data paths, test fixtures, standalone commands,
  notebook metadata, CI selectors, plugin references, and release allowlists.
  The current release runner explicitly names dilution, CPU nucleation, and
  latent-heat source paths; these are part of the move, not just site links.
- Resolve the data-container forwarding module after checking all consumers.
- Gate: paired code cells remain equivalent after moves; affected sources run
  from documented working directories, and example/infrastructure suites pass.
  Re-execute complete notebooks for calculation changes. Preserve outputs for
  verified code-identical prose/path-only changes when compatible with the
  existing example policy; validate any changed runtime-relative paths.

### Phase 4 — Consolidate repetition and retire outdated material

- Compare overlapping tutorial contents and outputs before combining anything.
  Give each retained tutorial a distinct learning objective.
- Move outdated proposals, provider catalogs, superseded roadmap narrative, and
  all completed roadmaps/closeout narratives, and verified redundant files into
  `.trash/docs-v0.3.0/`, preserving original paths
  and recording reasons/replacements. Keep current scientific evidence in
  Contribute. Verify retired content is absent from the published site and
  active example discovery; no public archive is created. Check that no completed
  roadmap or completed-epic table remains published, while active/pending work
  and unresolved validation/release gates remain accurately documented.
- Replace repetitive support caveats with canonical links while retaining short
  local warnings needed to prevent misuse of a specific example.
- Gate: removal has an explicit replacement/provenance record and no required
  unique content is lost. Recheck links and retained published routes.

### Phase 5 — Make the organization sustainable

- Update the existing Add Example and Add Theory guidance plus feature-writing
  conventions to require an owning topic and canonical links before a new file.
- Prefer extending a topic page to creating one page per implementation phase.
  Put implementation plans in `.opencode/plans`; put durable public scientific
  validation in Contribute; do not automatically publish task closeouts as
  features.
- Add a small docs structural check for missing navigation destinations,
  unclassified authored pages, stale moved paths, and broken local links. Keep
  generated API pages and paired source files explicitly accounted for.
- Gate: every authored page has an audience, an owning topic, and a route from
  navigation or an owning index. The catalog is represented by those indexes,
  not by a new permanent spreadsheet or parallel documentation tree.

## Compatibility and validation checklist for implementation

- Capture base and full-build warning baselines before the first change.
  Run `mkdocs build --strict` and
  `mkdocs build --config-file mkdocs-full.yml --strict` after relevant changes.
  Existing failures must be identified separately; do not weaken warnings to
  conceal new regressions. CI currently builds the full configuration without
  `--strict`, so a green CI build alone is not complete link evidence.
- Check rendered navigation, notebook output rendering, images, download/source
  links, generated API links, and representative old URLs and fragments.
- Scan Markdown, notebook Markdown cells/metadata, Python file paths, tests,
  release scripts, configs, and contributor instructions for moved paths.
  Include repository-source URLs to both `Gorkowski` and `uncscode`; establish
  the intended canonical upstream before normalizing them.
- For example changes run `pytest examples_tests -q -Werror`; for affected
  infrastructure run `pytest scripts/tests -q -Werror`. Run affected native
  quick-start and migration snippet checks when moving/restructuring snippets.
  Run isolated release example checks if their path allowlist changes.
- Pure navigation/prose changes do not require the entire scientific package
  suite. Changed calculations require the owning scientific regressions and
  complete published workloads. Apply repository lint checks to changed source.
- Preserve explicit native distribution/volume semantics, independent material
  and gas widths, inert gases, caller-owned transfers, and device requirements.
  Never convert missing CUDA measurements or historical plan closure into a
  performance qualification or release claim.
- Redirects must account for MkDocs directory-style URLs and old anchors;
  a path-only redirect is insufficient when a large page is split. Redirect
  stubs should not become duplicate sidebar entries or primary search results.
  Select and test the redirect mechanism before the first physical move.

## Review decisions before implementation

Agreed navigation: Guide, Examples, Theory, API, and Contribute, with getting
started and migration under Guide and current assistant setup under Contribute.
Agreed retirement policy: focus on v0.3.0; move outdated material to root
`.trash/` for posterity instead of maintaining a published archive. All completed roadmaps are also to be removed from published docs; only
active/pending roadmap work remains public. No further
preference decision is needed for routine retirement under this policy; verify
content, dependencies, and current evidence needs before each move.

Remaining recommended defaults: stage navigation before moves; use lowercase/hyphenated new paths with compatibility
routes for retained content; preserve evidence needed by current claims; and
retire obsolete material to `.trash` after content and caller review.

The user authorized implementation of all five phases, a commit per phase,
and one pull request for the completed refactor.

## Complete catalog and proposed destinations

All current paths below are relative to `docs/`. Proposed destinations are also
relative to `docs/`, except `.trash/...` destinations, which are repository-root
relative. These are proposals, not files created by this task. A paired row represents both `.py` and `.ipynb`.
“Split” and “review” explicitly identify work that needs content-level decisions.
Existing E9-M5 execution evidence remains in its historical inventory; this
catalog does not claim fresh execution or scientific validation.

### Features — 29 entries

| Current source | Proposed destination | Treatment |
| --- | --- | --- |
| [Features/Roadmap/coagulation-validation.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/coagulation-validation.md) | `contribute/validation/gpu/coagulation.md` | Keep supported matrix and evidence boundaries. |
| [Features/Roadmap/condensation-parity-walkthrough.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/condensation-parity-walkthrough.md) | `contribute/validation/gpu/condensation-parity.md` | Keep ownership and bounded validation record. |
| [Features/Roadmap/condensation-stiffness-study.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/condensation-stiffness-study.md) | `contribute/validation/numerics/condensation-stiffness.md` | Keep case catalog, metrics, and recommendation evidence. |
| [Features/Roadmap/data-native-v030-closeout.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/data-native-v030-closeout.md) | `contribute/validation/releases/v0.3.0.md` | Retain pending review/release gates and revision-specific evidence; no completed-roadmap listing. |
| [Features/Roadmap/data-oriented-gpu.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/data-oriented-gpu.md) | `contribute/roadmap/index.md + guide/data-model/ + contribute/validation/ + .trash/docs-v0.3.0/docs/Features/Roadmap/data-oriented-gpu.md` | Extract pending Epic I and current contracts/evidence; retire original and all completed A–H planning narrative. |
| [Features/Roadmap/graph-capture-closeout.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/graph-capture-closeout.md) | `.trash/docs-v0.3.0/docs/Features/Roadmap/graph-capture-closeout.md + contribute/validation/gpu/graph-capture-profiling.md` | Retire completed closeout; extract current evidence requirements and missing-measurement status with original authority/provenance preserved. |
| [Features/Roadmap/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/index.md) | `contribute/roadmap/index.md + .trash/docs-v0.3.0/docs/Features/Roadmap/index.md` | Replace with active/pending work only; retire original and completed-epic inventories. |
| [Features/Roadmap/mass-precision-study.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/mass-precision-study.md) | `contribute/validation/numerics/mass-precision.md` | Keep precision recommendation and reproduction record. |
| [Features/Roadmap/warp-autodiff-limitations.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/Roadmap/warp-autodiff-limitations.md) | `contribute/architecture/warp/autodiff-limitations.md` | Keep design/reference notes; not a shipped capability claim. |
| [Features/activity_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/activity_system.md) | `guide/processes/activity.md` | Keep user guide; link thermodynamics theory. |
| [Features/backend_selection.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/backend_selection.md) | `guide/gpu/backend-selection.md` | Keep explicit selection and fallback contract. |
| [Features/coagulation_strategy_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/coagulation_strategy_system.md) | `guide/processes/coagulation.md + guide/gpu/direct/coagulation.md` | Split user workflow from detailed direct-GPU contract. |
| [Features/condensation_strategy_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/condensation_strategy_system.md) | `guide/processes/condensation.md + guide/gpu/direct/condensation.md` | Split user workflow from detailed direct-GPU contract. |
| [Features/data-containers-and-gpu-foundations.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/data-containers-and-gpu-foundations.md) | `guide/data-model/ + guide/gpu/transfers.md + guide/gpu/slots.md` | Split by authority; map old headings before extracting. |
| [Features/dilution_strategy_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/dilution_strategy_system.md) | `guide/processes/dilution.md` | Keep concise process contract. |
| [Features/gpu_graph_capture.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/gpu_graph_capture.md) | `guide/gpu/resident/graph-capture.md` | Keep operator runbook and CUDA-only boundary. |
| [Features/gpu_graph_capture_performance.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/gpu_graph_capture_performance.md) | `contribute/validation/gpu/graph-capture-profiling.md` | Keep evidence status and provenance requirements. |
| [Features/gpu_resident_checkpoints.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/gpu_resident_checkpoints.md) | `guide/gpu/resident/checkpoints.md` | Keep exact-device lifecycle contract. |
| [Features/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/index.md) | `guide/index.md` | Replace flat feature cards with topic entry points. |
| [Features/native_scientific_processes.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/native_scientific_processes.md) | `guide/processes/index.md` | Consolidate process orientation; preserve native semantics. |
| [Features/nucleation_strategy_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/nucleation_strategy_system.md) | `guide/processes/nucleation.md` | Keep CPU and direct-Warp scope explicit. |
| [Features/particle-data-migration/construction.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/particle-data-migration/construction.md) | `guide/migration/v0.3.0/construction.md + guide/data-model/construction.md` | Keep migration delta; extract reusable current construction guide. |
| [Features/particle-data-migration/containers.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/particle-data-migration/containers.md) | `guide/migration/v0.3.0/containers.md + guide/data-model/ownership.md` | Keep historical comparison; extract current ownership reference. |
| [Features/particle-data-migration/dynamics.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/particle-data-migration/dynamics.md) | `guide/migration/v0.3.0/dynamics.md` | Keep process migration and link canonical contracts. |
| [Features/particle-data-migration/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/particle-data-migration/index.md) | `guide/migration/v0.3.0/index.md` | Canonical migration overview with release-context reconciliation. |
| [Features/particle-data-migration/troubleshooting.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/particle-data-migration/troubleshooting.md) | `guide/migration/v0.3.0/troubleshooting.md` | Keep migration troubleshooting; avoid duplicating API reference. |
| [Features/resident_benchmark_memory_budget.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/resident_benchmark_memory_budget.md) | `contribute/validation/gpu/resident-benchmarks.md` | Keep measurement and memory-budget record. |
| [Features/slot_exhaustion_policies.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/slot_exhaustion_policies.md) | `guide/gpu/slot-exhaustion.md` | Keep developer-level primitive boundaries; link CPU planning. |
| [Features/wall_loss_strategy_system.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Features/wall_loss_strategy_system.md) | `guide/processes/wall-loss.md` | Keep process entry; separate detailed GPU reference if needed. |

### Examples — 76 entries covering 122 files

| Current source | Proposed destination | Treatment |
| --- | --- | --- |
| [Examples/Activity/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/index.md) | `examples/thermodynamics/activity/index.md` | Rework owning index and ordered learning path. |
| [Examples/Aerosol/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Aerosol/index.md) | `examples/fundamentals/index.md` | Rework owning index and ordered learning path. |
| [Examples/Chamber_Wall_Loss/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/index.md) | `examples/processes/wall-loss/index.md` | Rework owning index and ordered learning path. |
| [Examples/Data_Containers/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Data_Containers/index.md) | `guide/data-model/index.md` | Consolidate into data-model landing; link examples separately. |
| [Examples/Dynamics/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/index.md) | `examples/processes/index.md` | Rework owning index and ordered learning path. |
| [Examples/Equilibria/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Equilibria/index.md) | `examples/thermodynamics/equilibria/index.md` | Rework owning index and ordered learning path. |
| [Examples/Gas_Phase/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/index.md) | `examples/fundamentals/gas/index.md` | Rework owning index and ordered learning path. |
| [Examples/Nucleation/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Nucleation/index.md) | `examples/processes/nucleation/index.md` | Rework owning index and ordered learning path. |
| [Examples/Particle_Phase/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/index.md) | `examples/fundamentals/particles/index.md` | Rework owning index and ordered learning path. |
| [Examples/Setup_Particula/Details/Contributor_Setup.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/Details/Contributor_Setup.md) | `contribute/setup.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Setup_Particula/Details/New_to_Python.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/Details/New_to_Python.md) | `guide/getting-started/installation/new-to-python.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Setup_Particula/Details/Setup_Conda.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/Details/Setup_Conda.md) | `guide/getting-started/installation/conda.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Setup_Particula/Details/Setup_PIP.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/Details/Setup_PIP.md) | `guide/getting-started/installation/pip.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Setup_Particula/Details/Setup_UV.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/Details/Setup_UV.md) | `guide/getting-started/installation/uv.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Setup_Particula/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Setup_Particula/index.md) | `guide/getting-started/installation/index.md` | Move installation out of Examples; reconcile version context. |
| [Examples/Simulations/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/index.md) | `examples/simulations/index.md` | Rework owning index and ordered learning path. |
| [Examples/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/index.md) | `examples/index.md` | Rework owning index and ordered learning path. |
| [Examples/Activity/activity_tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/activity_tutorial.py) + [Examples/Activity/activity_tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/activity_tutorial.ipynb) | `examples/thermodynamics/activity/activity-tutorial.py + .ipynb` | Co-locate activity tutorials; review overlap before consolidation. |
| [Examples/Activity/bat_activity_example.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/bat_activity_example.py) + [Examples/Activity/bat_activity_example.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/bat_activity_example.ipynb) | `examples/thermodynamics/activity/bat-activity-example.py + .ipynb` | Co-locate activity tutorials; review overlap before consolidation. |
| [Examples/Activity/equilibria_example.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/equilibria_example.py) + [Examples/Activity/equilibria_example.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/equilibria_example.ipynb) | `examples/thermodynamics/activity/equilibria-example.py + .ipynb` | Co-locate activity tutorials; review overlap before consolidation. |
| [Examples/Activity/ideal_activity_example.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/ideal_activity_example.py) + [Examples/Activity/ideal_activity_example.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/ideal_activity_example.ipynb) | `examples/thermodynamics/activity/ideal-activity-example.py + .ipynb` | Co-locate activity tutorials; review overlap before consolidation. |
| [Examples/Activity/kappa_activity_example.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/kappa_activity_example.py) + [Examples/Activity/kappa_activity_example.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Activity/kappa_activity_example.ipynb) | `examples/thermodynamics/activity/kappa-activity-example.py + .ipynb` | Co-locate activity tutorials; review overlap before consolidation. |
| [Examples/Aerosol/Aerosol_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Aerosol/Aerosol_Tutorial.py) + [Examples/Aerosol/Aerosol_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Aerosol/Aerosol_Tutorial.ipynb) | `examples/fundamentals/native-aerosol.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.py) + [Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.ipynb) | `examples/processes/wall-loss/chamber-forward-simulation.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.py) + [Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.ipynb) | `examples/processes/wall-loss/rectangular-wall-loss-strategy.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.py) + [Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.ipynb) | `examples/processes/wall-loss/spherical-wall-loss-strategy.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.py) + [Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb) | `examples/processes/wall-loss/wall-loss-tutorial.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.py) + [Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.ipynb) | `examples/processes/wall-loss/wall-loss-builders-factory.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Data_Containers/data_containers_and_gpu_foundations.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py) | `.trash/docs-v0.3.0/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py` | Retire forwarding module after callers move to the canonical example. |
| [Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.py) + [Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.ipynb) | `examples/processes/coagulation/charge/coagulation-with-charge-functional.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.py) + [Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.ipynb) | `examples/processes/coagulation/charge/coagulation-with-charge-objects.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.py) + [Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb) | `examples/processes/coagulation/coagulation-1-pmf-pattern.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.py) + [Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.ipynb) | `examples/processes/coagulation/coagulation-2-pdf-pattern.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.py) + [Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb) | `examples/processes/coagulation/coagulation-3-particle-resolved-pattern.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Coagulation_4_Compared.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.py) + [Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb) | `examples/processes/coagulation/coagulation-4-compared.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.py) + [Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.ipynb) | `examples/processes/coagulation/coagulation-5-time-evolution.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.py) + [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb) | `examples/processes/coagulation/functional/coagulation-basic-1-pmf.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.py) + [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb) | `examples/processes/coagulation/functional/coagulation-basic-2-pdf.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.py) + [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb) | `examples/processes/coagulation/functional/coagulation-basic-3-compared.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.py) + [Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb) | `examples/processes/coagulation/functional/coagulation-basic-4-particleresolved.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Condensation/Condensation_1_Bin.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_1_Bin.py) + [Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb) | `examples/processes/condensation/condensation-1-bin.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Condensation/Condensation_2_MassBin.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.py) + [Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb) | `examples/processes/condensation/condensation-2-massbin.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Condensation/Condensation_3_MassResolved.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.py) + [Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb) | `examples/processes/condensation/condensation-3-massresolved.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Condensation/Condensation_Latent_Heat.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py) + [Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb) | `examples/processes/condensation/condensation-latent-heat.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Condensation/Staggered_Condensation_Example.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.py) + [Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb) | `examples/processes/condensation/staggered-condensation-example.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Dynamics/Customization/Adding_Particles_During_Simulation.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.py) + [Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb) | `examples/processes/customization/adding-particles-during-simulation.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Equilibria/Notebooks/activity_part1.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Equilibria/Notebooks/activity_part1.py) + [Examples/Equilibria/Notebooks/activity_part1.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Equilibria/Notebooks/activity_part1.ipynb) | `examples/thermodynamics/equilibria/activity-part1.py + .ipynb` | Keep functional workflow; compare overlap with activity examples. |
| [Examples/Equilibria/Notebooks/equilibria_part1.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Equilibria/Notebooks/equilibria_part1.py) + [Examples/Equilibria/Notebooks/equilibria_part1.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Equilibria/Notebooks/equilibria_part1.ipynb) | `examples/thermodynamics/equilibria/equilibria-part1.py + .ipynb` | Keep functional workflow; compare overlap with activity examples. |
| [Examples/Gas_Phase/Notebooks/AtmosphereTutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/AtmosphereTutorial.py) + [Examples/Gas_Phase/Notebooks/AtmosphereTutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/AtmosphereTutorial.ipynb) | `examples/fundamentals/gas/environment-data.py + .ipynb` | Keep native/functional content; replace historical filename vocabulary. |
| [Examples/Gas_Phase/Notebooks/Gas_Species.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/Gas_Species.py) + [Examples/Gas_Phase/Notebooks/Gas_Species.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/Gas_Species.ipynb) | `examples/fundamentals/gas/gas-data.py + .ipynb` | Keep native/functional content; replace historical filename vocabulary. |
| [Examples/Gas_Phase/Notebooks/Vapor_Pressure.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/Vapor_Pressure.py) + [Examples/Gas_Phase/Notebooks/Vapor_Pressure.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Gas_Phase/Notebooks/Vapor_Pressure.ipynb) | `examples/fundamentals/gas/vapor-pressure.py + .ipynb` | Keep native/functional content; replace historical filename vocabulary. |
| [Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py) + [Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb) | `examples/processes/nucleation/custom-nucleation-single-species.py + .ipynb` | Keep scientific example; move pair together when present. |
| [Examples/Nucleation/cpu_nucleation.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Nucleation/cpu_nucleation.py) | `examples/processes/nucleation/cpu-nucleation.py` | Keep scientific example; move pair together when present. |
| [Examples/Nucleation/gpu_direct_nucleation.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Nucleation/gpu_direct_nucleation.py) | `examples/gpu/direct/gpu-direct-nucleation.py` | Keep scientific example; move pair together when present. |
| [Examples/Particle_Phase/Notebooks/Activity_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Activity_Tutorial.py) + [Examples/Particle_Phase/Notebooks/Activity_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Activity_Tutorial.ipynb) | `examples/thermodynamics/particle-properties/activity-tutorial.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Particle_Phase/Notebooks/Aerosol_Distributions.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Aerosol_Distributions.py) + [Examples/Particle_Phase/Notebooks/Aerosol_Distributions.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Aerosol_Distributions.ipynb) | `examples/fundamentals/particles/aerosol-distributions.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Particle_Phase/Notebooks/Distribution_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Distribution_Tutorial.py) + [Examples/Particle_Phase/Notebooks/Distribution_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Distribution_Tutorial.ipynb) | `examples/fundamentals/particles/distribution-tutorial.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.py) + [Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.ipynb) | `examples/thermodynamics/particle-properties/activity-functions.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.py) + [Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.ipynb) | `examples/fundamentals/particles/particle-data.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.py) + [Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.ipynb) | `examples/thermodynamics/particle-properties/particle-surface-tutorial.py + .ipynb` | Keep native/functional content; review activity overlap without dropping distinct science. |
| [Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.py) + [Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb) | `examples/simulations/biomass-burning-cloud-interactions.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.py) + [Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb) | `examples/simulations/cloud-chamber-multi-cycle.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.py) + [Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb) | `examples/simulations/cloud-chamber-single-cycle.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.py) + [Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb) | `examples/simulations/cough-droplets-partitioning.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.py) + [Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb) | `examples/simulations/organic-partitioning-and-coagulation.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/Simulations/Notebooks/Soot_Formation_in_Flames.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.py) + [Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb) | `examples/simulations/soot-formation-in-flames.py + .ipynb` | Keep full workload, outputs, limitations, and runtime context. |
| [Examples/cpu_dilution.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/cpu_dilution.py) | `examples/processes/dilution/cpu-dilution.py` | Keep scientific example; move pair together when present. |
| [Examples/data_containers_and_gpu_foundations.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/data_containers_and_gpu_foundations.py) | `examples/fundamentals/data-containers.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_coagulation_direct.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_coagulation_direct.py) | `examples/gpu/direct/gpu-coagulation-direct.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_complete_process_sequence.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_complete_process_sequence.py) | `examples/gpu/direct/gpu-complete-process-sequence.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_condensation_parity_walkthrough.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_condensation_parity_walkthrough.py) | `examples/gpu/direct/gpu-condensation-parity-walkthrough.py` | Keep validation walkthrough; distinguish from first-use tutorial. |
| [Examples/gpu_direct_kernels_quick_start.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_direct_kernels_quick_start.py) | `examples/gpu/direct/gpu-direct-kernels-quick-start.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_resident_graph_capture.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_resident_graph_capture.py) | `examples/gpu/resident/gpu-resident-graph-capture.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_resident_multi_timestep.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_resident_multi_timestep.py) | `examples/gpu/resident/gpu-resident-multi-timestep.py` | Keep scientific example; move pair together when present. |
| [Examples/gpu_resident_session.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/gpu_resident_session.py) | `examples/gpu/resident/gpu-resident-session.py` | Keep scientific example; move pair together when present. |
| [Examples/native_state_access.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Examples/native_state_access.py) | `examples/fundamentals/native-state-access.py` | Keep scientific example; move pair together when present. |

### Theory — 34 entries covering 35 files

| Current source | Proposed destination | Treatment |
| --- | --- | --- |
| [Theory/Accelerating_Python/Details/One-Line_Vision.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Accelerating_Python/Details/One-Line_Vision.md) | `.trash/docs-v0.3.0/docs/Theory/Accelerating_Python/Details/One-Line_Vision.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/Accelerating_Python/Details/Taichi_Exploration.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Accelerating_Python/Details/Taichi_Exploration.ipynb) | `.trash/docs-v0.3.0/docs/Theory/Accelerating_Python/Details/Taichi_Exploration.ipynb` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/Accelerating_Python/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Accelerating_Python/index.md) | `.trash/docs-v0.3.0/docs/Theory/Accelerating_Python/index.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/Activity_Calculations/activity_theory.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Activity_Calculations/activity_theory.md) | `theory/thermodynamics/activity-theory.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Activity_Calculations/equilibria_theory.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Activity_Calculations/equilibria_theory.md) | `theory/thermodynamics/equilibria-theory.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Activity_Calculations/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Activity_Calculations/index.md) | `theory/thermodynamics/index.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Code_Concepts/Details/Design_Patterns.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Code_Concepts/Details/Design_Patterns.md) | `contribute/architecture/design-patterns.md` | Move software-design guidance out of scientific Theory. |
| [Theory/Code_Concepts/Details/Object_Oriented_Patterns.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Code_Concepts/Details/Object_Oriented_Patterns.md) | `contribute/architecture/object-oriented-patterns.md` | Move software-design guidance out of scientific Theory. |
| [Theory/Code_Concepts/Details/WARMED_principle.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Code_Concepts/Details/WARMED_principle.md) | `contribute/architecture/warmed-principle.md` | Move software-design guidance out of scientific Theory. |
| [Theory/Code_Concepts/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Code_Concepts/index.md) | `contribute/architecture/index.md` | Move software-design guidance out of scientific Theory. |
| [Theory/OpenAI_Models/Models/Agents.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/OpenAI_Models/Models/Agents.md) | `.trash/docs-v0.3.0/docs/Theory/OpenAI_Models/Models/Agents.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/OpenAI_Models/Models/Chat.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/OpenAI_Models/Models/Chat.md) | `.trash/docs-v0.3.0/docs/Theory/OpenAI_Models/Models/Chat.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/OpenAI_Models/Models/Reasoning.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/OpenAI_Models/Models/Reasoning.md) | `.trash/docs-v0.3.0/docs/Theory/OpenAI_Models/Models/Reasoning.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/OpenAI_Models/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/OpenAI_Models/index.md) | `.trash/docs-v0.3.0/docs/Theory/OpenAI_Models/index.md` | Move outdated material to `.trash`; extract any unique current guidance first. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Fluid_and_Particle_Properties_Comparison.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Fluid_and_Particle_Properties_Comparison.ipynb) | `theory/dynamics/cloud-droplet-coagulation/dns-fluid-and-particle-properties-comparison.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Horizontal_Velocity_Comparison.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Horizontal_Velocity_Comparison.ipynb) | `theory/dynamics/cloud-droplet-coagulation/dns-horizontal-velocity-comparison.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Kernel_Comparison.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Kernel_Comparison.ipynb) | `theory/dynamics/cloud-droplet-coagulation/dns-kernel-comparison.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Distribution_Comparison.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Distribution_Comparison.ipynb) | `theory/dynamics/cloud-droplet-coagulation/dns-radial-distribution-comparison.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Relative_Velocity_Comparison.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Relative_Velocity_Comparison.ipynb) | `theory/dynamics/cloud-droplet-coagulation/dns-radial-relative-velocity-comparison.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/Droplet_Coagulation_Kernel_Ayala2008.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/Droplet_Coagulation_Kernel_Ayala2008.md) | `theory/dynamics/cloud-droplet-coagulation/droplet-coagulation-kernel-ayala2008.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/index.md) | `theory/dynamics/cloud-droplet-coagulation/index.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Condensation_Equations.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Condensation_Equations.md) | `theory/dynamics/condensation-equations.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/Nucleation_Equations.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/Nucleation_Equations.md) | `theory/dynamics/nucleation-equations.md` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Dynamics/ionparticle_coagulation.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Dynamics/ionparticle_coagulation.ipynb) | `theory/dynamics/ionparticle-coagulation.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Properties/dynamic_viscosity.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Properties/dynamic_viscosity.ipynb) | `theory/properties/dynamic-viscosity.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/Properties/mean_free_path.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/Properties/mean_free_path.ipynb) | `theory/properties/mean-free-path.ipynb` | Keep scientific theory/validation with equations and provenance. |
| [Theory/Technical/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/Technical/index.md) | `theory/index.md` | Consolidate empty hierarchy level into scientific landing. |
| [Theory/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/index.md) | `theory/index.md` | Replace mixed directory description with scientific topic map. |
| [Theory/nvidia-warp/datastructures.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/datastructures.md) | `contribute/architecture/warp/datastructures.md` | Move Warp teaching material; preserve illustrative-only boundaries. |
| [Theory/nvidia-warp/examples/fluids.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/examples/fluids.md) | `contribute/architecture/warp/teaching/fluids.md` | Move Warp teaching material; preserve illustrative-only boundaries. |
| [Theory/nvidia-warp/examples/geometry.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/examples/geometry.md) | `contribute/architecture/warp/teaching/geometry.md` | Move Warp teaching material; preserve illustrative-only boundaries. |
| [Theory/nvidia-warp/examples/gpu_benchmarks.py](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/examples/gpu_benchmarks.py) + [Theory/nvidia-warp/examples/gpu_benchmarks.ipynb](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/examples/gpu_benchmarks.ipynb) | `contribute/validation/gpu/legacy-benchmarks/gpu-benchmarks.py + .ipynb` | Keep historical benchmark viewer; preserve data paths and evidence limitations. |
| [Theory/nvidia-warp/index.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/index.md) | `contribute/architecture/warp/index.md` | Move Warp teaching material; preserve illustrative-only boundaries. |
| [Theory/nvidia-warp/kernels.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/Theory/nvidia-warp/kernels.md) | `contribute/architecture/warp/kernels.md` | Move Warp teaching material; preserve illustrative-only boundaries. |

### Existing migration route — 1 entry

| Current source | Proposed destination | Treatment |
| --- | --- | --- |
| [migration/particle-data.md](https://github.com/Gorkowski/particula/blob/9e024234f2f6401313981e8e3c85a772d675f795/docs/migration/particle-data.md) | `guide/migration/v0.3.0/index.md` | Retain old-route redirect; point it at the consolidated migration section. |

## Initial planning deliverable verification (historical snapshot)

- All 187 tracked files in the four scoped directories are covered exactly once across 140 catalog entries.
- All 46 Examples notebook/source pairs and the Theory benchmark pair are accounted for.
- Current-source links resolve locally; proposed targets are intentionally not yet created.
- Only this plan was added. No documentation, configuration, examples, or production files were changed.
- No implementation tests or site builds were run for this planning-only change.

## Implementation record

### Phase 1 — completed

- Added explicit Guide / Examples / Theory / API / Contribute navigation, with
  section landing pages first and plugin-aware API/notebook navigation.
- Replaced the home-page reference dump and legacy marketing template with a
  short overview. Extracted its unchanged CPU calculation into First simulation
  and updated its independent conservation regression.
- Reworked Guide, Features, Examples, Theory, and Contribute entry paths; added
  development-checkout installation context and separated GPU examples.
- Baseline strict builds passed (base 1.56 s; full 50.85 s). Updated strict
  builds passed (base 1.53 s; full 49.72 s), with no warning/error or broken-link
  messages. The full build renders notebooks without executing calculations.
- Quick-start regression: 2 passed. Changed Python passed Ruff checks/format.
- Browser inspection verified correct five-section landing links, desktop
  Guide layout, and the readable 390 px mobile layout.
- Remaining folder moves, content splits, retirement, durable checks, and final
  example validation belong to phases 2–5.

### Phase 2 — narrative relocation and extraction completed

- Moved 54 narrative paths into Guide, Theory, and Contribute; split container,
  transfer, slot, direct-process, and execution references into 12 additional
  topic pages. Current construction/access snippets now live in Data model;
  migration retains historical comparisons and links to those canonical guides.
- Added a relocation manifest and post-build redirects outside navigation and
  search. Verified all 54 redirect destinations and all 137 split-page fragment
  destinations against rendered HTML. Old heading fragments retain useful
  destinations even when their sections moved to different pages.
- Updated current docs, notebook Markdown links, assistant workflow selection,
  extraction-tool paths, and snippet-test references. Three affected notebooks
  retain exactly unchanged code cells and outputs.
- Strict base/full builds passed (1.67 s / 50.38 s); no warning/error or broken-link
  messages. Focused native/theory/process/quick-start tests: 46 passed using a
  writable temporary Warp cache. No scientific assertions or tolerances changed.
- Roadmap/history extraction is intentionally completed with retirement in
  phase 4, so original authorities remain available until their successor
  evidence pages are ready. Example and theory notebook moves are phase 3.

### Phase 3 — example and theory source relocation completed

- Relocated 125 example/theory sources, including all 46 Examples notebook/script
  pairs, into topic folders. Python stems use snake_case for ordinary imports;
  prose-only theory notebook stems use hyphens. Updated release allowlists,
  source tests, CI selectors, helper instructions, and repository-source links.
- Retired the redundant data-container forwarding module after migrating its
  callers; retained all canonical scientific assertions and tolerances.
- All 46 example pairs have equivalent executable ASTs. All 55 moved notebooks
  initially retained identical code ASTs, execution counts, and outputs. The
  historical benchmark pair had a pre-existing mismatch: its notebook lacked
  the Python source's controlled-artifact validation. Synchronized it to that
  source and executed the complete viewer against the existing local recorded
  artifact. This is viewer validation, not new measured GPU qualification.
- Source example suite: 221 passed, 2 optional skips. Infrastructure: 132 passed.
  Fresh-wheel release example suite: 20 passed. Ruff passed for example sources,
  example tests, release runner, and infrastructure tests. Production package
  files have no diff.
- Strict base/full builds passed (1.47 s / 57.78 s). Final link/reachability
  checks and browser review will run across the complete phase-5 tree.

### Phase 4 — consolidation and retirement completed

- Retired 12 narrative/proposal files after extraction (13 archived originals
  including the phase-3 forwarding script). Verified every archived file is
  byte-identical to its recorded source revision and absent from active docs.
- Replaced the mixed/completed GPU roadmaps with pending Epic I only. Extracted
  field ownership, dated machine-bounded coagulation timings, and capture
  qualification gates into their owning guide/validation pages. Retained
  UNSHIPPED/BLOCKED measurement status and pending v0.3.0 release approval.
- Removed obsolete provider catalogs, the Taichi/vision proposals, the empty
  Technical hierarchy, and the redundant process catalog. Current assistant
  setup already contains the useful workflow instructions; no local supporting
  assets were required by the retired proposals.
- Compared activity/equilibrium tutorial objectives and preserved their distinct
  strategy, functional, phase-separation, and partitioning calculations. Added a
  topic learning-objective map. No numerical snippets, outputs, or tolerances
  changed in this phase.
- Focused process, foundations, parity, and capture-example regressions:
  70 passed, 2 optional skips. Strict base/full builds passed (1.37 s / 48.08 s).
  Compatibility routes point to retained current content; obsolete-only model
  catalogs and proposals have no published archive or blanket home redirect.

### Phase 5 — sustainable ownership and final verification completed

- Curated the five sections into topic subfolders, keeping each section landing
  first. Removed the generic Detailed reference bucket. Added GPU/example and
  validation entry points; all 156 authored pages are reachable from navigation
  or an owning index. Generated API remains available in the full build.
- Updated Add Example, Add Theory, feature/theory authoring agents, and the
  documentation guide to prefer canonical topic pages. Public implementation
  closeouts are no longer a feature-writing pattern. The initial catalog above
  now links to its immutable pre-move revision; the relocation manifest and
  topic indexes describe the implemented organization.
- Added `scripts/check_docs_structure.py` and failure-oriented regressions for
  missing destinations/anchors, orphan pages, stale links, source pairs, source
  discovery, rendered notebook links, API output, redirects, and retired search
  entries. CI runs source and rendered checks plus a strict full build.
- Fixed notebook publication routing: notebook prose links now resolve to built
  destinations. Excluded all paired Python sources from page generation, fixing
  the historical viewer's Python page overwriting its notebook output.
  Disambiguated repeated/inline-code notebook headings and preserved meaningful
  original anchors in the relocation manifest.
- Final browser review verified desktop and 390 px navigation, first-use links,
  the corrected Theory tab, generated API entry, retained notebook outputs and
  benchmark chart, and an old foundations fragment redirect to Transfers. The
  temporary viewport override was reset.
- The viewer still contained a blanket condensation performance recommendation.
  Replaced it with artifact-specific limits and preserved its original pair at
  revision `a16861c91` in `.trash`. There are now 15 byte-identical archived
  originals. The viewer's code/output sync was fully executed against existing
  recorded data; this is not current performance qualification. CI reports the
  known artifact-only viewer unavailable when neither controlled data nor an
  export exists, without skipping ordinary notebooks or syntax failures.
- All 187 original scoped sources have a living or archived destination. All
  46 Examples pairs and the retained benchmark pair have equivalent executable
  code. All 54 other moved notebooks retain exactly the original code ASTs,
  execution counts, and saved outputs. No production-package diff exists.
- Validation: 221 source-example tests passed, 2 optional-device skips (phase 3);
  70 affected process/foundations/capture checks passed, 2 skips (phase 4); final
  infrastructure suite 150 passed; isolated fresh-wheel release examples 20
  passed. All 12 standalone sources exited successfully from the repository
  root. Native CUDA capture reported unavailable without fallback.
- Final strict base/full builds passed (1.38 s / 48.38 s). Source and rendered
  structural checks report zero errors, including every retained redirect and
  mapped fragment. Changed source passed Ruff; the diff has no whitespace errors.

## Completion audit

| Requirement | Evidence |
| --- | --- |
| Five top-level sections and clear starting paths | Shared MkDocs navigation, section indexes, desktop/mobile review |
| Topic folders, split long guides, versioned migration | Guide/Data model/Processes/GPU, nested examples/theory, v0.3.0 migration |
| Preserve current contracts and scientific limits | Field-ownership policy, validation records, unchanged package and scientific ASTs/assertions |
| Remove completed roadmaps and outdated material | Pending-only roadmap; 15 provenance-recorded originals outside docs; no retired search entries |
| Preserve routes, assets, and pairs | Relocation manifest, complete rendered link/anchor checks, equivalent pairs, retained outputs |
| Update release/test/discovery consumers | 221 example tests, 150 infrastructure tests, 20 installed-wheel checks, 12 standalone runs |
| Sustainable contribution rules | Owning-topic rules, updated agents, CI structural checks and strict publication |
| Commit phases and deliver one PR | Phases 1–4 committed separately; this phase-5 commit completes implementation before the single PR |

This refactor does not approve the v0.3.0 release or qualify CUDA performance.
Those explicit scientific/maintainer gates remain open in their validation records.

## Follow-up — retire vector-store publication

- The vector-store build and upload are no longer part of the documentation
  workflow. Archive `docs/.assets/vector_file_generator.py` and
  `docs/.assets/vector_store_upload.py` unchanged under `.trash/docs-v0.3.0/`.
- Remove the OpenAI and GitPython development dependencies; the retired
  uploader was their only Python consumer in this repository.
- The associated `AIdocs.yml` action was already removed in `8dc8bc51f`.
  Keep the current MkDocs site/API publication workflow.
- Validation: archived scripts match their source bytes; active build/tooling
  references and dedicated dependencies are absent; documentation structure
  reports zero errors, the strict base build passes, and the diff passes
  whitespace checks.
