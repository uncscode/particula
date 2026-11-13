---
name: particula-docs
description: "Maintain Particula repository docstrings, scientific guides, theory, and published Python examples in Astro/Starlight, including saved outputs and links. Not for unrelated document drafting or private simulation notebooks."
---

# Maintain documentation

Read `../../references/repository-conventions.md` for the selected checkout.
Then read its `docs/contribute/documentation.md` and `website/AUTHORING.md`
for current placement and build rules. Use `../../references/docstrings.md`
for function/class/module prose, and `../../references/scientific-boundaries.md`
when documenting physical or execution contracts.

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

For executable changes, run `python scripts/docs_examples.py refresh EXAMPLE_KEY`
for every affected complete workload, inspect figures/tables/text, and retain
sources plus `website/results/` bundles. Preserve workload sizes, assertions,
scientific parameters and CUDA requirements. Failed refreshes retain prior
outputs and do not establish acceptance. Prose-only edits may keep results;
check fingerprints without inventing a rerun or changing recorded provenance.

Update navigation, links, registry inputs, loader/tests and release allowlists
when moving sources. Use the existing relocation mechanism for meaningful old
routes. Build with `npm --prefix website run build`; it renders saved results
and does not run simulations. Run structure/tests from
`../../references/repository-validation.md` as appropriate.

Document public direct steps and required concrete setup accurately without
promoting every helper to a public API. Keep numerical, execution and performance
claims bounded by actual evidence. Report checks and any output inspection or
required execution that could not be completed.
