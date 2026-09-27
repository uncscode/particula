# Example Tasks

1. P1: Replace condensation `_make_data_inputs` facade conversions with M2
   native initialization; retain missing-strategy and single-box rejection
   behavior and add mixed-gas mapping snapshots.
2. P2: Extend staggered conservation at V=0.25/4 with two species and unequal
   weights. Assert gas and particle changes separately, distinguishing closed
   inventory from `update_gases=False` reservoir behavior.
3. P3: Migrate Brownian, charged and combined fixtures; audit all sedimentation
   and turbulence rows. Assert existing independent kernel/rate references.
4. P4: Audit resolved binning/merge in `coagulation_strategy_abc.py`; add a
   case where unweighted mass sums conceal weighted-inventory errors.
5. P5: Migrate wall-loss access using M1 normalization. Preserve geometry and
   helper-parity tests in BOTH CPU suites and run each explicitly.
6. P6: Build an M2 flat aggregate and call scientific strategies directly in
   an integration fixture. Publish scientific contracts/evidence for M4,
   without rewriting runnables or prematurely deleting legacy APIs.

These are examples within the six serial phases, not parallel workstreams.
