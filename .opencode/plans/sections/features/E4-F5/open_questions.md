# Open Questions

1. What exact particle-concentration and box-volume convention maps per-particle
   mass to gas kg/m3 in the GPU containers? Resolve against the CPU reference
   before P2 implementation.
2. Which deterministic reduction implementation meets Warp CPU/CUDA parity
   while retaining fixed-shape caller-owned scratch?
3. Should new scratch be exposed as individual keyword-only buffers or one
   typed sidecar established by E4-F3? Preserve public positional compatibility.
4. What explicit fp64 tolerances will separate strict bookkeeping conservation
   from broader physics parity?
5. Should invalid nonbinary int32 partitioning values fail on the host preflight
   or through a device-side validation status without hidden synchronization?

Diagnostics requested: none. These questions must not weaken the issue #1272
production-hook and conservation gates.
