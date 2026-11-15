---
name: particula-docs
description: "Organize and maintain Particula documentation, its topic graph, scientific narratives, docstrings, and published Python examples in Astro/Starlight. Not for unrelated document drafting or private simulation notebooks."
---

# Maintain documentation

Read `../../references/repository-conventions.md` for the selected checkout.
Then read its `docs/contribute/documentation.md` and `website/AUTHORING.md`
for current placement and build rules. Use `../../references/docstrings.md`
for function/class/module prose, and `../../references/scientific-boundaries.md`
when documenting physical or execution contracts.

For collection reviews and topic navigation, read
`../../references/documentation-topics.md`. Inventory and classify the full
requested scope, distinguish mapped pages from editorially reviewed pages, and
record a prioritized backlog. Apply this workflow without rewriting articles
when the request is only to audit or plan.

For examples and scientific explanations, read
`../../references/scientific-narratives.md`. Establish the question and intended
reader capability, connect code to evidence, interpret results, and close with
a bounded answer and concrete research extension. Use the topic graph to connect
prerequisites and follow-on concepts. Keep short reference edits proportional.

Choose the reader's task and extend its owning topic before creating a page:
Guide for supported usage, Examples for calculations, Theory for scientific
basis, API generated from docstrings, Contribute for development/evidence.
Implementation phases are not public features. Put requested plans in `plans/`
and durable scientific evidence in `docs/contribute/validation/` with revision,
device, limitations and outstanding gates.

Maintained examples are percent-cell Python sources under `docs/examples/`
and `docs/theory/`, registered in `website/examples.json`. They are not paired
notebooks. Jupytext still parses the sources; do not remove the dependency on
the assumption that it only supports notebook pairing.

For changes to simulation physics, parameters, numerical methods or scientific
inputs, run `python scripts/docs_examples.py refresh EXAMPLE_KEY` for each
affected complete workload. Preserve workload sizes, assertions and CUDA
requirements. For plotting-only changes, use
`python scripts/docs_examples.py replot EXAMPLE_KEY` from saved numerical plot
data; do not rerun the simulation. Inspect the resulting figures/text. Replot
must verify data integrity and unchanged simulation inputs, preserve original
execution provenance, and record rendering provenance separately. Missing or
incomplete data requires a one-time data-saving refresh; never reconstruct
numerical observations from an image or label a replot as a new simulation.

Store each example's current outputs directly in `website/results/EXAMPLE_KEY/`,
without timestamp/run subfolders or current-generation pointers. Git retains
history. Publish only after success; failures preserve the previous bundle.
Save compact plotted summaries (times, counts, distributions and quantiles),
not full particle states. Numerical plot data has a hard 10 MiB per-example
serialized JSON limit; exceeding it fails without truncating values. Prose-only
edits can retain results when executable fingerprints still match.

Update navigation, links, registry inputs, loader/tests and release allowlists
when moving sources. Use the existing relocation mechanism for meaningful old
routes. Build with `npm --prefix website run build`; it renders saved results
and does not run simulations. Run structure/tests from
`../../references/repository-validation.md` as appropriate.

Document public direct steps and required concrete setup accurately without
promoting every helper to a public API. Keep numerical, execution and performance
claims bounded by actual evidence. Report checks and any output inspection or
required execution that could not be completed.
