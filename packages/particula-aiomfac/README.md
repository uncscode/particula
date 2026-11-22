# particula-aiomfac

This development distribution identifies the AIOMFAC activity reference used by Particula. Import
`UPSTREAM_VERSION` and `UPSTREAM_REVISION` from `particula_aiomfac` to identify the scientific model.
The 0.1.0 milestone wheel contains provenance only. The source distribution includes the
[reference workflow](reference/README.md), input cases, fixture data, and developer tests.
The numerical Python/CUDA calculator and Particula provider are later milestones.

For local installations build both distributions, then install the Particula wheel with its
`aiomfac` extra and point pip's `--find-links` at the backend wheel directory.
Publish the compatible backend before a public Particula release advertises this extra.
