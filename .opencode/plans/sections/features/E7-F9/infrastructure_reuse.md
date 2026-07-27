# Infrastructure Reuse

- `particula/execution/diagnostics.py` and scheduler diagnostic barriers planned
  by E7-F5: extend typed resident views and registered output buffers; do not add
  host callbacks to normal GPU steps.
- `particula/execution/gpu_session.py`, `gpu_resources.py`, and `checkpoint.py`
  planned by E7-F4: reuse lifecycle, fixed-shape resource registration, explicit
  sync/restore, and versioned checkpoint boundaries.
- `particula/execution/process_graph.py`, `scheduler.py`,
  `thermodynamic_updates.py`, and `state_updates.py` planned by E7-F5: use the
  canonical order and stale-derived-state protections rather than duplicating a
  test-only scheduler.
- E7-F7 communication declarations and ledgers: reuse synchronous extensive-
  amount staging and conservation diagnostics for transport/expansion cases.
- E7-F8 `StreamRegistry` and stream checkpoint records: reuse stable logical-box
  identities and process namespaces for restart/reordering evidence.
- `particula/gpu/warp_types.py:24-184`: preserve fixed-shape multi-box schemas.
- `particula/gpu/conversion.py:120-317,422-666`: spy on the established explicit
  upload, synchronization, and restore boundaries.
- `particula/gpu/kernels/thermodynamics.py:318-377`: preserve on-device
  vapor-pressure refresh ordering after temperature changes.
- `docs/Examples/gpu_complete_process_sequence.py:380-494`: reuse fixtures and
  explicit-transfer style, but replace illustrative direct orchestration with the
  E7 user-facing session API.
- `particula/gpu/tests/process_sequence_test.py:808-872,1651-1677`: reuse
  no-intermediate-restore and five-process composition fixtures.
- `particula/gpu/tests/gpu_complete_process_sequence_example_test.py:494-801`:
  reuse call-order, transfer-spy, and failure-propagation patterns.
- `particula/tests/runnable_test.py`: retain CPU sequence behavior as an
  independent reference.
- `particula/gpu/tests/kernel_exports_test.py`: guard deliberate public exports.
- `docs/Features/data-containers-and-gpu-foundations.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md:1472-1606`: preserve existing
  ownership wording, support boundaries, Track T9 scope, and the Epic G exit bar.
