# Full CUDA forcing publication


The complete `gpu-prepared-forcing` publication subsequently passed on the
RTX 5060 from pinned revision `480ecf2c1aa95da42985098c53e4ecdc66133d7b`.
Its package source hash is
`bbe6cb05017e86b19f7dc166c18cac96c0d762b5300c03990d91cd1a4e08ce8b`.
The guarded run took **3,139.79 s**, retaining all sixteen boxes, 4,096 slots
per box, 49,152 initially active particles, four materials and five gas lanes.
The coarse run used 6,000 one-microsecond cycles; the fine run used 12,000
half-microsecond cycles and `rtol=1e-7` rather than `1e-5`. Both retained the
same physical duration and 120 forcing control times.

- Maximum final per-particle total-mass relative difference: **2.32234e-7**
  against the predeclared 1% threshold.
- Maximum final water-vapor relative difference: **8.85905e-9** against 0.1%.
- Maximum per-box/species inventory residual: **1.15463e-14** coarse and
  **2.93099e-14** fine, with external vapor changes and dilution accounted for.
- Coarse adaptive work: 9,682 accepted and 1,711 rejected trials; fine work:
  17,281 accepted and 2,797 rejected trials. Both had zero domain rejections.
- Coarse coupling used 12,000 full downloads and 12,000 full uploads; the fine
  run used 24,000 of each. These measured costs preclude claiming fully resident
  execution or inferring a GPU speedup from this correctness comparison.

All saved-file hashes and scientific inputs were verified. The retained compact
report contains thirteen frames and numerical refinement/work summaries, without
full particle states. Figures were inspected before and after CPU-only replot;
the replot preserves the original CUDA execution revision, device and timestamp.
Its invocation now explicitly shows the figure, avoiding duplicate notebook
display of the helper's returned figure. The extracted download's source and
helpers matched byte-for-byte, imported from its own directory, and rendered the
saved data with GPU calls disabled and no data mutation. That download check
tested import/rendering, not a second execution of the full scientific workload.

The sparse displayed history records the supplied initial state before the first
control update, then complete cycles before the next control update. It does not
resolve all control changes or pulse peaks. The final-state refinement test is
numerical consistency evidence for the declared case, not an independent
experimental reference or an error bound at every intermediate time. The full
cyclic chamber and the remaining CPU scientific studies are still required.
The imported bundle passed the 563-page website build and rendered verification
with zero errors; see `.artifacts/coupled-forcing-website.log`. The final ZIP's
executable source and all helper bytes match the independently tested download.
