# Nucleation: connect formation rates to material and particle counts

A formation law predicts a source; committing that source must also respect gas
availability, particle composition, physical volume, and storage capacity. Choose
an example according to the question you need to answer.

| Question | Start here | Evidence to inspect |
| --- | --- | --- |
| Can an initially inactive population receive particles without creating mass? | [Bounded CPU nucleation](cpu_nucleation.py) | Mapped particle-plus-gas inventory, active counts, fixed capacity |
| How does an imposed source compete with growth of existing particles? | [Custom single-species workflow](custom_nucleation_single_species.py) | External vapor budget, source threshold, growth and collision ordering |
| What must a caller supply for a low-level Warp source step? | [Direct nucleation](../../gpu/direct/gpu_direct_nucleation.py) | Transfers, source configuration, sidecar ownership, synchronization |

Begin with the CPU example and the [process contract](../../../guide/processes/nucleation.md).
The CPU runnable is a deterministic one-box path using fixed slots and
partitioning gas. Its exhaustion controls are part of the model setup, not a
reason to silently resize or discard particles.

The custom workflow is an illustrative saturation-based mass source with explicit
slot activation. It is useful for understanding accounting and process coupling;
its coefficient and threshold are not a validated atmospheric formation law.

The direct Warp example uses `nucleation_step_gpu` and defaults to the Warp CPU
device. It demonstrates a distinct low-level execution path, with caller-owned
transfers and work arrays. Running it on Warp CPU does not establish CUDA
performance or replace the supported CPU runnable.

After construction works, ask whether a predicted source survives changes in
formation diameter, precursor abundance, and timestep. Track both particle number
and material transferred: two cases can conserve the same mass while producing
very different numbers of nuclei. Use a sourced rate law and its applicability
bounds before interpreting that sensitivity as a research result.
