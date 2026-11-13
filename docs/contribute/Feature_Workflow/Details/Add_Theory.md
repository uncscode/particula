# Add Theory

## Goal

Create a short, citable document that explains the science behind the new feature.

## What to include

Include enough information that a new reader would be able to understand and implement the feature. This should include:

- An overview of the physical/chemical process.
- Governing equations (Unicode symbols preferred) with variable definitions.
- Assumptions or limitations.
- At least one primary literature citation.

## Steps

1. Create a new issue on GitHub and assign it to yourself.
2. Create a branch on your forked repo for this issue.
3. Identify the owning scientific topic under `docs/theory/`. Extend its existing page when the purpose matches; add a descriptive lowercase filename only for a distinct explanation. Put software architecture in `docs/contribute/architecture/` and durable numerical evidence in `docs/contribute/validation/`.
4. Link the file in the index.md file in the same folder.
5. Write using Markdown; no specific imports needed.
6. Commit this file in a branch.
7. Create your pull request.

Follow the [documentation placement rules](../../documentation.md). Implementation
plans and completed task narratives belong in `.opencode/plans/`.
