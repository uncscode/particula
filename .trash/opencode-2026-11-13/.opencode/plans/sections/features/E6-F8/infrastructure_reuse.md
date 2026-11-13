# Infrastructure Reuse

- E6-F7's planned `particula/dynamics/nucleation/nucleation_strategies.py` and
  `particle_source.py` are the authoritative equations, units, validity gates,
  source semantics, and independent CPU oracle. Port; do not reinterpret them.
- E6-F5's planned `particula/gpu/kernels/slot_management.py` supplies strict
  active/free predicates, ascending fixed-shape free indices, exact `wp.int32`
  counts, and activation. Do not duplicate slot classification.
- E6-F6's planned `particula/gpu/kernels/exhaustion.py` supplies complete-demand
  planning, resampling-first precedence, optional scaling fallback, and
  conservation diagnostics. Do not silently reduce demand.
- `particula/gpu/warp_types.py:25-105` defines `WarpParticleData` and
  `WarpGasData`; preserve their caller-owned arrays and container identities.
- `particula/gpu/conversion.py:113` and adjacent conversion helpers define the
  only explicit CPU-to-Warp boundary. The kernel step must not call them.
- `particula/gpu/kernels/condensation.py:131` demonstrates a typed optional
  scratch sidecar; its ownership validation around lines 993-1500 and
  gas-coupled `condensation_step_gpu` at line 1814 provide the preflight,
  inventory-finalization, and caller-buffer-identity pattern.
- `particula/gpu/kernels/environment.py:81-230` provides positive-finite scalar
  and per-box same-device normalization patterns where temperature/pressure
  gates are required by the frozen E6-F7 model.
- `particula/gpu/kernels/__init__.py` provides lazy intended entry-point export
  conventions; configuration and implementation kernels remain concrete-module
  APIs unless an explicit public contract requires otherwise.
- `docs/Theory/Technical/Dynamics/Nucleation_Equations.md:117-180` records the
  bounded equations, composition, inventory coupling, and fixed-slot behavior.
