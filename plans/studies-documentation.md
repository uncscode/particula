# Organize published calculations and evidence under Studies

**Final placement decision.** The full former Contribute validation collection
belongs in Studies. Split records by the question they answer, keep method and
qualification details with the owning scientific study, and put timing protocols
beside their benchmarks. The [workflow and release qualification collection](../docs/studies/validation/workflows-and-releases/index.md)
covers cross-cutting checks. The original inventory and provisional dispositions
below document the planning baseline.

Planning baseline: upstream main at `ebabce042e`, November 19, 2026. The counts below describe the source at that revision. The proposed destinations are editorial choices for the migration.

## What readers should see

Keep **Explore topics** first in the sidebar, with its four scientific branches. Replace the existing Examples section with **Studies**. Guide, Theory, API and Contribute keep their own entries. Opening Studies should immediately answer “Where is the Python code?” and “Where can I read comparisons with the literature?”

```text
Explore topics
Guide
Studies
  All runnable Python scripts
  Worked examples
  Tutorials
  Literature analyses
  Validation studies
  Benchmarks
Theory
API
Contribute
```

The five study categories correspond to folders under `docs/studies/`: `examples/`, `tutorials/`, `literature/`, `validation/` and `benchmarks/`. **All runnable Python scripts** is an index across all five. Put it first in the expanded menu and link it from the Studies landing page and the first-simulation guide. Use ordinary links that work with JavaScript disabled. Topic, study category and required device filters can narrow the list, with shareable URLs and a clear reset.

Each study gets one main sidebar entry. On its page, put **View code** and **Download Python script** near the title when code is available. Place the complete ZIP beside them when helpers or data are required. A study with several scripts lists what each runs. A prose-only analysis appears in Literature analyses and on its scientific topic pages. The code index contains one row per registered runnable script.

### Where Theory ends and Studies begins

Keep **Theory** top level. Put derivations, assumptions, units and applicability there, including short snippets that help explain an equation. Put a complete runnable calculation or a report investigating a particular result under **Studies**. A paper can supply the equation on one page and the comparison on another. Move the eight registered Theory scripts to Studies, keeping their execution keys. In the cloud-droplet coagulation derivation, for example, link **Run the kernel comparison** beside the equation. Beside the comparison plot, link **Read the kernel derivation**. Both pages remain discoverable through the same scientific topic.

### Keep each browsing decision small

Use **six immediate choices as a working maximum** for a Studies sidebar group or landing-page directory. Count each visible study and subfolder once, including secondary-category listings on landing pages. Make the folder title open its landing page so an extra “Overview” row does not consume a choice. Where the sidebar renderer uses a disclosure control for expansion, keep the landing-page link and the expansion control independently accessible by keyboard. Small collections can list their studies directly.

At the seventh choice, review what the reader is trying to distinguish. First check whether any study belongs in a different primary category. If the collection is still crowded, introduce two to four meaningful subject or task groups and give each a landing page. A landing can mix a group with individual studies. Keep a single study directly accessible until there is enough material to justify a group. If the titles form one useful sequence with no sensible split, retain the sequence and record the editorial exception during review. Never split automatically by title, date or list position.

Mirror meaningful groups on disk. Count primary studies and child collections when reviewing a folder. A report with a script and frozen specification can occupy a named study folder with one entry in its parent. Its internal links expose each file by purpose. Keep a standalone script directly in its collection. For collection landings, use a short introduction that explains the choice, with immediate children generated in a consistent order from the same records as the sidebar. Preserve a curated order for tutorials. The full script catalog remains searchable and filterable across every group. Its results are a list of matches and can exceed six.

**A first Literature analyses landing.** The seven provisional primary literature records in the inventory below can become four choices. These labels and assignments are a pilot outline.

| Choice in the menu | Destination |
| --- | --- |
| Chamber haze and cloud formation | A landing with the two Π Chamber studies, one on haze and cloud spectra and one on seed chemistry and organic vapor. |
| Water uptake and cloud activation | A landing with the citric acid hygroscopicity, size-resolved CCN retrieval and BAT cloud activation studies. |
| Dry particle mobility sizing | Direct link to the PSL slip-factor comparison. |
| Nanoparticle scavenging in a chamber | Direct link to the NaCl capture study. |

The spherical wall-loss study could appear as a fifth link here through its secondary literature category. Its primary sidebar home remains Validation studies. This example lets a new reader choose among scientific questions without adding a folder for each paper.

**Where to pilot next.** The current Prepared workflows menu has 15 entries, Full simulations has 12, and Activity has nine. Their source folders contain 15, 11 and eight registered scripts respectively. Start with Prepared workflows. The first CPU-to-CUDA walkthrough is a tutorial, measured RTX 5060 scaling belongs in Benchmarks, and the charged-collision and atmospheric-ion studies are candidates for a shared subject landing among the remaining worked examples. Recount the proposed destinations before naming further subfolders. These Studies guidelines apply alongside the existing two-to-four-child convention for the separate Explore topics tree.

## Quick inventory

Counts come from the tracked, non-hidden Markdown and Python sources, the execution registry, the topic registry and the sidebar. The Validation row is a subset of Contribute. The home page, shared data, images and generated API pages sit outside these section counts.

| Current directory | Markdown | Registered Python scripts | Migration decision |
| --- | ---: | ---: | --- |
| Guide | 50 | 0 | Keep task instructions, first steps and short snippets. Link full runs from the instruction that uses them. |
| Examples | 13 | 93 | Classify each script by the reader's question. Salvage introductions, prerequisites and curated sequences from the 13 indexes. Generate repeated lists. |
| Theory | 9 | 8 | Keep equations, assumptions and references in Theory. Move full runnable calculations into Studies and link them from the relevant theory pages. |
| Contribute, including Validation | 59 | 1 | Keep contributor procedures, release decisions and qualification gates here. Review the Validation subset separately. |
| ↳ Contribute/Validation | 32 | 1 | Triage scientific reports, supporting specifications and operational material. Move the recorded benchmark viewer with its historical evidence. |

All **102** scripts have percent cells, a distinct registry source, a topic resource and a saved manifest. Manifest presence establishes coverage only. The execution registry specifies CPU for 71 scripts and CUDA for 31. Seven use the long-running policy, including two CUDA scripts. Derive the device filter from the registered device. For example, the historical GPU benchmark viewer runs on CPU and displays previously measured GPU timings.

The 93 Examples scripts currently break down into processes (31), GPU (24), thermodynamics (13), fundamentals (12), simulations (11) and optics (2). These are useful subjects and execution contexts. They do not determine whether the page teaches a workflow or tests a scientific claim.

The [topic graph](../website/topics.json) contains **58 concepts, 199 resource records and 36 supporting-page dispositions**. All scripts have a primary topic and 27 also have secondary topics. The [sidebar](../website/navigation.json) already points across folder boundaries, including two example scripts in Guide → Getting started. Topic pages currently list directly associated resources grouped by section. The API contributes one graph resource for its overview. Links to individual functions need more precision than that overview provides.

### Triage the 32 validation pages

The following is a first-pass disposition by current filename under [Validation](../docs/contribute/validation/). Confirm the main question and unique evidence before moving a page. Several records combine scientific findings with engineering decisions.

| Proposed treatment | Current records |
| --- | --- |
| Literature analyses, with explicit links to the paper, observable and associated scripts | `transport/psl-mobility`, `thermodynamics/{pi-chamber,pi-chamber-species,citric-acid-comparison,ccn-retrieval,bat-ccn}`, `coagulation/nacl-scavenging` |
| Validation studies or supporting evidence for a study | `gpu/{spherical-wall-loss,charged-coagulation}`, `coagulation/van-der-waals`, `thermodynamics/bat-activity`, `optics/{reference-corpus,cuda}`, `numerics/{initial-size-inversion,condensation-stiffness,adaptive-methods}` |
| Benchmarks or worked studies, according to the primary question | `gpu/coagulation-benchmarks` with the CPU viewer, `gpu/bat-humidity-pulse` for measured workflow cost, `gpu/environment-sweep` for the scientific workload |
| Companion material | `gpu/spherical-wall-loss-specification` lives beside its study and retains its addressable frozen criteria. |
| Contributor procedures and decisions | `releases/v0.3.0`, `gpu/{graph-capture-qualification,graph-capture-profiling,resident-benchmarks,process-field-extensions}`, `numerics/simulation-workflow`. Keep the unmeasured resident benchmark matrix visibly unmeasured. |
| Review section by section | `gpu/{prepared-rtx5060,condensation-parity,coagulation}`, `numerics/{scientific-correctness,mass-precision}`. Link the scientific result from Studies and keep ownership, release gates or implementation decisions in Contribute. Preserve headings that have inbound links. |
| Replace the old index | `index` routes readers to the scientific collections and the remaining contributor records through a transition page or targeted redirects. |

The counts inventory the full collection. The proposed groupings are a triage pass, not an editorial review of every article. In particular, a mixed page can already have a better canonical study in the execution registry.

## One home for a study, several ways to find it

Choose a primary folder by the opening question. **Worked examples** apply a calculation, **Tutorials** guide a reader through a sequence, **Literature analyses** investigate a named paper or dataset, **Validation studies** test a bounded claim against a stated reference, and **Benchmarks** report measured computational cost for a specified workload. A literature comparison can also be a validation study. Give it one home and one optional secondary category when both browsing paths are useful. A numerical check within an introductory tutorial does not by itself justify a validation label.

Use the existing topic resource as the authoring point. For a study with one script, the Python page can be the main resource. For a longer literature analysis, the Markdown report can be the main resource and point to one or more scripts. Retain separate report, specification and code pages when they answer distinct questions. Only the main resource gets the study's sidebar entry.

| Information | Authoritative input | Generated behavior |
| --- | --- | --- |
| Primary category | First folder under `docs/studies/` for the main resource | Main sidebar home, breadcrumb and primary category filter. An optional controlled secondary category adds an index listing pointing to the same study. |
| Scientific subjects | Existing `topic` and `also` IDs in `website/topics.json` on the main resource | Topic links on the study, a Studies group on matching topic pages, and topic filters in the catalog. Companion pages can inherit the study's subjects unless explicitly classified for a separate purpose. |
| Code, device and downloads | Existing key in `website/examples.json`, with its source, device, dependencies and invocation | Exactly one code-index row per key. Resolve the current URL and source/ZIP downloads from the key. The saved manifest supplies run provenance. |
| Supporting material | Named links from the main resource to the companion report, frozen specification, or additional run keys | Links with clear labels in both directions. Pages that support several studies can list all referring studies. |
| Guide, Theory and Contribute connections | Explicit resource targets on the main study, with a relation such as “How to use this”, “Underlying theory” or “Qualification procedure” | Show the link on the study and a “Related studies” link on the target. Validate all paths and fragments. |
| API connections | Curated public symbol used or explained by the study | Resolve the symbol against the generated API and verify its anchor. Surface the reverse link at the matching symbol when supported, otherwise on its owning module page with the symbol named. |

Extend the existing [resource validator](../scripts/docs_topics.py) and [site preparation](../scripts/docs_site.py) to resolve these relationships once. Reserve an explicit stable study ID for each main entry, reusing the existing execution key for a single-script study where appropriate. Use that ID to group companions, and the existing execution key for each script. Keep each page's existing coverage disposition. Generated headings, filters, sidebar and backlinks should consume the same resolved records. No parallel tag list in Python headers or hand-maintained reverse-link arrays are needed. A new author should be able to register a script, choose its home and subjects, and specify any precise relationships once.

Shared topics enable browsing. They are too broad to establish a prerequisite, an API dependency or scientific support for a claim. A “Browse studies on condensation” link may include descendants of the selected topic, deduplicated by study ID. The existing topic cards and curated **Start here** entries retain their present behavior. Specific guide, theory, procedure and API backlinks use the explicit relationships above. Display evidence descriptions such as “compared with published theory” or “compared with measured slip factors” in the study text or summary. A category alone makes no qualification claim.

For example, the wall-loss report would own study ID `spherical-wall-loss-theory` in `studies/validation/`. Its existing topic IDs are `losses-deposition` and `verification`. The author adds secondary category `literature`, run key `spherical-wall-loss-theory`, the frozen-specification resource, a how-to link to `docs/guide/processes/wall-loss.md`, and API symbol `particula.dynamics.wall_loss.SphericalWallLossStrategy`. The site can then build both category listings, the topic links, the code entry and the guide/API backlinks. The report owns the evidence narrative. The run owns the displayed code, figures and execution metadata.

### Pilot the links with real pages

| Starting point | Expected path and return |
| --- | --- |
| A new user looking for code | Studies → All runnable Python scripts → **Run, save and resume a closed aerosol growth experiment** → View code or download. The Guide's first CPU simulation links directly to that same tutorial. |
| A scientist browsing wall loss | Explore topics → the `losses-deposition` topic → **CUDA wall loss agrees with spherical deposition theory**. The study also appears under Validation studies and Literature analyses. Link the wall-loss guide, the `SphericalWallLossStrategy` API entry, the executable key `spherical-wall-loss-theory`, and the frozen specification. The guide and API point back. Preserve the distinction between theoretical survival and measured chamber decay. |
| A reader checking experimental evidence | Studies → Literature analyses → **Dry PSL mobility sizing** → the comparison with 49 measured slip factors and the `psl-mobility` script. The size and transport topics reach the same study. The PSL guide retains the practical sizing workflow. |
| A reader following an equation | Theory → the cloud-droplet coagulation kernel derivation → a registered DNS comparison script → back to the derivation. Link a specific comparison that exercises the equation and identify which result to inspect. |
| A maintainer checking performance | Contribute → the resident benchmark procedure → relevant benchmark evidence when available. The historical coagulation viewer identifies both the recorded GPU workload and its CPU viewing requirement. Preserve the distinction between those workloads. |

## Migration order and acceptance

1. **Make a source-to-destination map before moving files.** Start with the 102 execution keys, 13 Examples indexes and 32 validation Markdown pages. Record the proposed study ID, main page, companion sources, old/new paths and fragments, topic IDs, and affected consumers. Review the eight Theory scripts against their surrounding explanations. Check scientific narratives elsewhere in Guide and Theory for links to existing executions. Count immediate children in the proposed folders and in the rendered navigation, including secondary listings. Record where a landing or a justified exception is needed. Reconcile new upstream additions at each batch. Finish when every scoped source has a destination or a documented reason to remain.
2. **Build the catalog and link resolver against current sources.** Extend the graph's section roles for Studies, site preparation, [topic components](../website/src/components/TopicExplorer.astro), navigation generation and structure checks. During migration, allow explicit proposed categories for unmoved records, then derive these from the final paths and retire the temporary overrides. Test a multi-script study, a prose-only analysis, a companion shared between studies, a secondary topic, a missing target and a nonexistent API anchor. Have the structure check report any Studies group above six with its path, count and titles for editorial review. Test the six-to-seven boundary, grouped companions, and a full catalog with more than six results. Confirm one row per registered execution and one primary sidebar entry per study. A reader should reach the code index by opening Studies and selecting its first item.
3. **Move a representative pilot end to end.** Start with the CPU tutorial, CUDA wall-loss comparison and its frozen specification, PSL measurement comparison, one Theory calculation, and the recorded benchmark viewer. Hold the original execution keys and results directories stable. Check source and ZIP downloads, local relative imports, helper and data destinations, links in percent-cell prose, figure paths and `__file__` assumptions. The [redirect generator](../scripts/docs_site.py) currently accepts old Markdown and notebook sources. Extend it for old Python page routes, check collisions where retired notebook and Python routes coincide, and point earlier relocations straight to final destinations. Verify old fragments against the built site. Preview a populated Literature analyses landing and the reorganized Prepared workflows collection, including paths back to their parent landings. Verify a reader can open both the derivation and its runnable comparison from either direction.
4. **Move the remaining studies in batches by scientific topic.** Update the execution registry, topic resources, section indexes, test loaders, CI paths, release-example allowlist and Ruff paths together. Add Studies to the [installed docs locator](../particula_docs.py) and [source distribution inputs](../pyproject.toml) with the first move. Preserve useful plain Markdown entry points in the offline bundle. Update the authoring guide, Particula Helper skills, literature publication instructions and curated LLM index at cutover. Keep historical plans identifiable as historical. Replace the old Examples and Validation landings with working paths to their successors.
5. **Verify navigation, scientific provenance and distribution.** Run the topic and structure checks, website Python/JS tests, website build and rendered structure check. Exercise links with the deployed base path, desktop and mobile menus, keyboard navigation, disabled JavaScript and search. Check for one searchable main study per canonical URL and for redirects outside the search index. Run affected example tests, release example checks and wheel/sdist docs-bundle verification. Confirm each runnable key and companion is still reachable from the installed sources and each claimed API backlink resolves in the rendered site.

The website build renders saved results. For relocation-only edits, compare executable signatures, input paths and digests, output integrity, displayed freshness and original execution metadata before and after. Input names contribute to fingerprints, so retaining data and helper paths where practical reduces unnecessary reruns. Where the runner requires a refresh, use its registered workload and device and record the new evidence. Historical runs retain their recorded revision and supported reproduction path.

**Ready for cutover** means a new reader can find code from the sidebar, a scientist can traverse topic → study → evidence or API and back, and a contributor can reach the procedure that produced a record. Check all three journeys on the rendered pilot before applying the folder convention to the rest of the collection.
