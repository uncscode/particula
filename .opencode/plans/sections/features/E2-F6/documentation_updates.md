# E2-F6 Documentation Updates

## Required Documentation

- Create a precision/mass representation report, preferably under
  `docs/Features/` or a clearly linked roadmap subpage.
- Include the final recommendation, validation evidence, tested cases,
  candidate alternatives, and reproduction instructions.
- Cross-link the report from `docs/Features/Roadmap/data-oriented-gpu.md` where
  the T6 requirement is described.

## Recommended Content

- Current baseline: absolute per-species `fp64` mass storage on CPU and GPU.
- Candidate alternatives: absolute `fp32`, mixed precision, log mass, and
  reference/binned mass scaling.
- Metrics: conservation, small-particle fidelity, clamping behavior, memory
  footprint, and throughput.
- Decision: recommended policy and follow-up constraints for schema/dtype work.

## Notebook/Example Notes

If the evidence is presented in a notebook under `docs/Examples/` or
`docs/Theory/`, follow the repository Jupytext workflow: edit the paired `.py`
file, sync the notebook, execute it, and commit both files.
