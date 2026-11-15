# Organize scientific topics

Use this reference when reviewing a documentation collection, adding a page, or
changing topic navigation. Work in the selected checkout, not installed payloads.
The current `website/topics.json` owns concepts, page associations, and explicit
supporting-page dispositions. Existing Guide, Examples, Theory, API, and
Contribute sections describe document roles; the topic graph connects them.

## Review a collection

Inventory all authored pages and registered Python examples before proposing
moves or deletion. Assign each a primary topic or an explicit supporting-page
reason. Note its scientific question, reader prerequisites, distinct contribution,
and editorial gaps. Similar titles do not establish duplicate content. Preserve
unique calculations, evidence, and device boundaries when merging material.
Use topic associations and better indexes before changing established routes.

Keep a prioritized backlog in the requested plan location. Separate scientific
contradictions, missing interpretation, and navigation problems. A page being
mapped does not mean its science or editorial quality has been reviewed.

## Graph rules

- Root: aerosol science and modeling. Branching nodes have two to four children;
  leaves have none. Collapse single-child branches. Resource lists are not
  branches and have no four-item limit. Never invent empty categories to meet
  a quota. Group concepts by meaningful scientific distinctions.
- Aim for at most four selections from root to subject. Record a `depth_reason`
  for a necessary deeper topic. Use stable IDs independent of display labels.
- Each concept has one primary parent. Siblings should have comparable scope.
  Name the science using recognizable terms, not implementation filenames.
- Keep hierarchy, prerequisites, related concepts, and page associations distinct.
  A parent is not automatically a prerequisite. Parent and prerequisite graphs
  must be acyclic; related links are reciprocal. Explain relationships through
  topic descriptions and nearby prose when the connection is not evident.
- Each page has one primary topic and optional additional associations. Link to
  one canonical article rather than copying explanations across branches.
- Record true alternate names and abbreviations in `aliases`. Broader remembered
  phrases belong in `search_phrases`; do not declare them scientific synonyms.
- Every topic has a label, description, local SVG icon, and useful resources or
  children. Icons accompany text and illustrate concepts rather than measured
  results. Preserve accessible labels, theme contrast, and reduced motion.
- Keep the card description short. Fill `context` with the scientific distinction
  and evidence limits that make the topic useful; use `investigation` for one
  concrete research question with a meaningful comparison or diagnostic. These
  fields should connect the resources, not repeat their titles or promise results
  that the linked examples have not established. Curate `start_here` resources
  for a practical first reading, and recheck them when examples change.
- Inventory operational guidance explicitly as supporting material where it does
  not belong in the scientific tree. Keep it reachable through existing sections.
- Check graph coverage, links, icons, and rendered navigation after changes.
  An added page needs a disposition; an added concept needs a useful destination.

Read the checkout's contributor documentation and website authoring guide for
current build commands and generated-file boundaries. Do not edit generated
Starlight content or install caches. The offline topic entry should still provide
useful ordinary links; the website adds progressive interactive navigation.

## Basis

The distinction between preferred/alternate labels, broader/narrower concepts,
and associated concepts follows [W3C SKOS](https://www.w3.org/TR/skos-primer/).
The branching and depth limits are Particula editorial conventions, not a claim
that SKOS or learning research mandates a particular number of choices.
