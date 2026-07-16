# Documentation Updates

- Create `docs/Features/Roadmap/condensation-parity-walkthrough.md` as the
  canonical T8 report: fixture, independent CPU/Warp construction, equations,
  separate criteria, observed result format, focused commands, bounded claims,
  and deferred-capability ownership table.
- Create `docs/Examples/gpu_condensation_parity_walkthrough.py` as a runnable,
  deterministic Warp CPU walkthrough. Keep it distinct from the broader direct
  kernels quick start and avoid a notebook unless interactive presentation is
  demonstrably needed.
- Update `docs/Features/condensation_strategy_system.md` to link the report and
  distinguish its walkthrough evidence from strategy/`Runnable` parity.
- Update `docs/Features/data-containers-and-gpu-foundations.md` focused
  reproduction table with the walkthrough command and its three evidence
  categories.
- Update `docs/Examples/index.md` with the new runnable example, required Warp
  CPU policy, and optional CUDA note.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Epic E carry-forward row
  with the final artifact and owner table while preserving later-epic scope.
- Update `docs/Features/Roadmap/index.md` to mark the T8 artifact available for
  E5-F9 closeout without prematurely marking E5 shipped.
- Update `.opencode/plans/sections/features/E5-F8/` phase/status content after
  implementation and hand the artifact paths to E5-F9.

Documentation must use the exact signed-energy units (`kg * J/kg = J`), state
that `energy_transfer` is caller-owned write-only diagnostic output, and avoid
claims of temperature feedback, general CPU strategy parity, graph replay,
broad autodiff, adaptive stepping, performance, or required CUDA support.
