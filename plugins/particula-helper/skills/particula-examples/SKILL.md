---
name: particula-examples
description: "Create or edit published Particula examples in docs with an adaptable scientific narrative, explanations for every code block, plots or process visuals, closing reflections, saved results and topic/navigation integration. Not for private study scripts, general guides or API docstrings."
---

# Author a worked example

Work in the selected Particula checkout. Read its `website/AUTHORING.md` and
`docs/contribute/documentation.md`, plus `../../references/example-authoring.md`
for the adaptable format and visual requirements. Use
`../../references/scientific-narratives.md` for evidence and interpretation,
`../../references/documentation-topics.md` for categorization, and
`../../references/repository-validation.md` for checks.

## Scope and write

1. Establish the scientific question, reader prerequisites, intended takeaway,
   model assumptions, units and supported backend. For an edit, inspect the
   current source, registered invocation, saved results and topic associations.
   Preserve useful content, stable routes and evidence outside the request.
2. Adapt the reference outline to this problem. Keep a recognizable progression
   from question and setup through explained code and visual evidence to
   interpretation and reflections. Introduce every code block with nearby prose
   explaining its purpose, important inputs and what to inspect in its output.
   Group small related fragments rather than filling the page with trivial
   narration. Keep helper definitions and their later invocations understandable.
3. Plan the visual before the calculation: ordinarily a results plot from the
   example's data, with a problem/process schematic when it clarifies geometry,
   mechanisms or information flow. A qualitative example can use a meaningful
   schematic instead. Follow the reference's visual and accessibility guidance;
   never present an illustrative diagram as computed or measured evidence.
4. Author maintained percent-cell Python under `docs/examples/` (or an existing
   theory calculation in `docs/theory/`). Do not create paired notebooks or edit
   generated Starlight pages. Separate simulation and plotting where feasible
   so results can be replotted from compact saved data. Use supported APIs and
   independent scientific checks; read `../../references/scientific-boundaries.md`
   when changing model behavior or making CPU/GPU claims.
5. Interpret the displayed evidence beside its output. End with a section that
   answers the opening question, explains applicability and limitations, and
   proposes a concrete next investigation. Label unrun extensions as proposals.

## Run, connect and verify

- Register/update `website/examples.json`: source, title, invocation, execution
  policy/device, timeout, input dependencies and saved-data/replot configuration
  as applicable. Use a nearby comparable example and the current registry schema.
  Scientific changes require `python scripts/docs_examples.py refresh EXAMPLE_KEY`
  for the full workload. Plot-only changes use `replot EXAMPLE_KEY` on verified
  saved data; prose-only edits retain outputs when executable fingerprints match.
  Keep provenance and CUDA requirements; do not silently shrink workloads or
  substitute CPU. Report unavailable execution instead of claiming fresh results.
- Store current results in `website/results/EXAMPLE_KEY/`. Save compact plot
  summaries within the 10 MiB per-example limit, not full particle states.
  Reuse existing saved-data mechanisms; do not reconstruct numerical data from
  pictures. Register helper inputs and keep downloads runnable after extraction.
- Update the example's resource entry in `website/topics.json` with one primary
  scientific topic, useful optional associations and the correct page role.
  Maintain `website/navigation.json` and relevant index/prerequisite/follow-on
  links. Review `start_here` placement when it helps the topic's reading path.
  Prefer existing concepts; follow the topic reference if a new concept is needed.
  Renames/moves need the existing relocation mechanism and updated links.
- Run topic and structure checks, applicable example tests and the website build.
  Inspect the actual figures and rendered example for readable labels, captions,
  code/prose/output order and working topic/navigation links. A successful build
  alone does not establish scientific correctness or visual quality. Report any
  inspection or execution that remains unavailable.

Deliver the example source, results/provenance, topic and navigation changes,
checks performed, and remaining evidence limits. Apply this workflow to the
requested examples; do not retrofit unrelated examples automatically.
