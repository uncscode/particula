# Open Questions

- [ ] Which neutral or charged wall-loss scenario gives the smallest stable
  statistical integrated fixture while still exercising persistent RNG?
  - Resolve in P1 from E6-F3/F4's recorded sample-size and sigma bounds; do not
    invent a weaker tolerance.
- [ ] Should the canonical example use one box for readability while the
  integration test uses multiple boxes/species?
  - Proposed: yes. Keep the published example minimal and put broad shape and
    conservation coverage in `process_sequence_test.py`.
- [ ] Which exact diagnostics should the example print after slot exhaustion
  handling?
  - Resolve after E6-F5/F6/F8 freeze sidecar names; print counts and policy
    outcomes, not device arrays or unstable object representations.
- [x] Does E6-F9 implement backend selection or a process scheduler?
  - Resolved 2026-07-21: No. Both are explicitly owned by Epic G; E6-F9 only
    calls direct entry points in a fixed validation/example sequence.
- [x] May the example transfer state to the host between processes?
  - Resolved 2026-07-21: No. CPU/Warp conversion occurs at setup and final
    checkpoint boundaries only.
- [ ] What measured tolerances and focused commands should be published?
  - Resolve in P2 from passing E6-F1-F8 contracts and record exact values in the
    roadmap and feature guides before closeout.
