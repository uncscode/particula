---
name: particula-docs
description: "Organize and maintain Particula guides, theory, topic navigation and docstrings in Astro/Starlight. Route creating or editing published worked examples to particula-examples. Not for unrelated document drafting or private study scripts."
---

# Maintain documentation

Read `../../references/repository-conventions.md` for the selected checkout.
Then read its `docs/contribute/documentation.md` and `website/AUTHORING.md`
for current placement and build rules. Use `../../references/docstrings.md`
for function/class/module prose, and `../../references/scientific-boundaries.md`
when documenting physical or execution contracts.

For collection reviews and topic navigation, read
`../../references/documentation-topics.md`, including the independent Explore
topics menu folder and stable topic routes. Inventory and classify the full
requested scope, distinguish mapped pages from editorially reviewed pages, and
record a prioritized backlog. Apply this workflow without rewriting articles
when the request is only to audit or plan.

For creating or editing published worked examples, use particula-examples and
`../../references/example-authoring.md`: adaptable page structure, prose before
every code block, results plots or process visuals, closing reflections and
complete topic/navigation integration. For other scientific explanations, use
`../../references/scientific-narratives.md`. Keep short reference edits
proportional.

Choose the reader's task and extend its owning topic before creating a page:
Guide for supported usage, Examples for calculations, Theory for scientific
basis, API generated from docstrings, Contribute for development/evidence.
Implementation phases are not public features. Put requested plans in `plans/`
and durable scientific evidence in `docs/contribute/validation/` with revision,
device, limitations and outstanding gates.

Example execution and saved-result handling follow particula-examples and the
checkout's `website/AUTHORING.md`. Scientific changes require a full refresh;
plot-only changes use verified saved-data replots; prose-only changes can retain
results when executable fingerprints match. Preserve provenance and required
CUDA execution. Do not create paired notebooks or edit generated pages.

Update navigation, links, registry inputs, loader/tests and release allowlists
when moving sources. Use the existing relocation mechanism for meaningful old
routes. Build with `npm --prefix website run build`; it renders saved results
and does not run simulations. Run structure/tests from
`../../references/repository-validation.md` as appropriate.

Document public direct steps and required concrete setup accurately without
promoting every helper to a public API. Keep numerical, execution and performance
claims bounded by actual evidence. Report checks and any output inspection or
required execution that could not be completed.
