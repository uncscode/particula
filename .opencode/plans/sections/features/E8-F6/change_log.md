# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-08-30 | Selected symbolic full-retention and checkpoint-interval tape projections with explicit variables and no empirical or measured-tape claim | user decision |
| 2026-08-30 | Selected a documented, version-qualified Warp/CUDA allocator high-water API as the only optional observed peak source; unsupported or incomplete coverage remains explicitly unavailable without NVML substitution | user decision |
| 2026-08-30 | Initial E8-F6 plan drafted for opt-in captured-versus-uncaptured multi-box scaling and reproducible memory-budget evidence; classifier diagnostics preserved as none | plan-feature-drafter |
| 2026-09-05 | Issue #1581 shipped P1's concrete host-only benchmark schema/artifact support and default-collection tests: validated frozen records and provenance, deterministic schema JSON round trips, and verified-`.artifacts` atomic generic JSON writes. No Warp/CUDA interaction, production execution/export, public documentation, or benchmark publication changed. | implementation reconciliation |
| 2026-09-05 | Issue #1582 shipped P2's test-only captured-versus-uncaptured resident comparison: schema-v2 paired device-synchronized samples, setup/capture provenance, an isolated fixed artifact, lazy CUDA/native-capture fixture support, hardware-free contract coverage, and one opt-in CUDA row. CUDA/native-capture absence cleanly skips; no generic artifact, public API, CPU fallback, speed threshold, or performance claim was added. | implementation reconciliation |
