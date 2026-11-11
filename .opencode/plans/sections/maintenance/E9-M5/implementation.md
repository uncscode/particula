# M5 single-PR implementation record

## Accepted delivery — 2026-11-11

The maintainer confirmed M4 completion and requested implementation of M5,
followed by one PR. Base: `3393e47fa` (merged upstream PR #1633).
Branch: `codex/e9-m5-native-examples-docs`; target: `uncscode/particula:main`;
issue: #1609. This supersedes per-phase issue generation and separate PRs.

## Serial checkpoints

1. Scope and exhaustive inventory (in progress).
2. P1 foundational tutorials with runtime fixtures and executed pairs.
3. P2 condensation/coagulation and repository helper example.
4. P3 wall loss, dilution and nucleation.
5. P4 composed simulations, customization, Activity and Equilibria.
6. P5 direct-Warp and resident examples.
7. P6 migration, replacement guidance and public quick starts.
8. P7 contributor guidance and final consumer audit.
9. Final validation and M6 handoff.

Each example-changing checkpoint includes its tests and executed notebook
outputs. Preserve existing paths and scientific content. No production physics,
new API, facade workaround, legacy deletion, release or performance claim.
Native capability gaps block the owning migration rather than justify reducing
its scientific workload. M6 acceptance and PR merge remain separate decisions.

## Inventory and evidence

`example_inventory.json` enumerates all 58 Python sources and 46 paired
notebooks at the base. Pending rows are not migration or execution evidence.
Expand the ledger with published snippets, links, quantities, tolerances,
commands and final-revision results during the owning checkpoint. Audit the
whole tree again at closeout; absence of legacy spellings is insufficient.

Use the repository `.venv`, existing notebook sync/execution tools, and current
`examples_tests/` runtime suites. Old plan references to removed documentation
tests do not authorize recreating prose-only assertions. All published workloads
must execute; timeout skips remain incomplete evidence. Required final gates:
untargeted repository coverage, examples and infrastructure suites, both explicit
CPU wall-loss suites, GPU/resident boundary checks, lint/type checks, payload
consistency and strict MkDocs. Preserve thresholds and scientific tolerances.

## P1 partial validation — 2026-11-11

- Aerosol, gas species and gas/environment tutorials migrated; each standalone
  source and synchronized notebook executed successfully. Three numerical
  fixtures pass in `examples_tests/native_foundations_example_test.py`.
- Canonical data-container example now stores explicit resolved counts at
  physical volume, with the guide forwarding to it. Both entrypoints execute;
  `WARP_CACHE_PATH=/tmp/particula-m5-warp .venv/bin/pytest
  examples_tests/data_containers_example_test.py -q --no-cov`: 14 passed.
  Enabled transfer failures propagate instead of claiming CPU-only success.
- Notebook tools require `PYTHONPATH=/home/kyle/Code/Agent` for the existing
  ADW utility dependency. Execution uses repository `.venv` on PATH and local
  Jupyter kernel access. Initial restricted-kernel failures were rerun with
  access; no failed attempt is counted as a successful run.
- Gas-species coefficients and their convention are retained as illustrative
  inputs, not requalified scientific data. The altitude tutorial evaluates
  only its published water diagnostic over the cold range, retaining all gas
  lanes without extrapolating unqualified organic coefficients.

P1 remains incomplete. Particle tutorials, remaining functional foundations,
landing-page reconciliation, plot inspection and strict rendering are pending.
These working-tree results are provisional until the P1 commit and closeout.

### Foundation checkpoint validation

All ten P1 notebooks passed source-first sync followed by full execution;
all ten retain output cells. Three unchanged Python sources were restored after
bidirectional sync exposed stale notebook text; execution was then repeated.
Do not sync output-bearing notebooks back through stale sources after execution.
The six native-foundation assertions and fourteen transfer assertions pass
(20 total). Strict MkDocs passed. Plot montages were inspected; radius-PDF
coordinate conversion now includes the explicit 2e6 Jacobian and an independent
population/material-integral regression. Final P1 ledger and snippet audit
remain pending; this checkpoint does not complete M5.

### P1 execution and inventory audit

All twelve P1 Python entrypoints executed directly with repository `.venv`,
`MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/particula-m5-mpl`, and
`WARP_CACHE_PATH=/tmp/particula-m5-warp`; every exit code was zero. All ten
paired notebooks also executed and retain outputs. Functional activity,
surface and vapor-pressure examples retain their independent strategies;
no executable facade imports or conversion bridges remain in P1 sources.
Topic-index paths remain stable; discovery-based MkDocs navigation includes
these pages. The inventory assigns later published snippet migration to P6/P7.
P1 implementation gate is passed at the foundation checkpoint plus this
record. Final M5-wide validation and acceptance remain pending.

### P2 condensation checkpoint

All five condensation source/notebook pairs now use native data and explicit
process strategies/species maps. Full original workloads execute successfully;
saved outputs are retained. Closed resolved cases assert species inventory
conservation at rtol=1e-12 and atol=1e-30. The fixed-gas mass-bin experiment
remains an open reservoir, explicitly distinguished from conservation tests.
Native latent energy (J) is divided by represented volume for J/m³ reporting.
Eleven focused tests pass. Plot inspection supports finite trajectories and
roundoff-level conservation, not a universal stability ranking. P2 coagulation
and helper starter migration remain pending; this is not the P2 completion gate.

### P2 coagulation and helper checkpoint

All eleven coagulation pairs execute at their original published workload sizes,
including four smaller examples under Functional and both charged-kernel routes.
All 16 plots inspected. Resolved runs conserve independently weighted inventory
at rtol=1e-12, atol=1e-30; native binned diagnostics match independent sums or
radius quadrature. The old preset builder regenerated default bimodal inputs,
overwriting the declared concentration array. Native initialization now honors
the declared modes/numbers; the correction is explicit in each affected example.
For the two-hour 80 nm pattern cases, material mass drifts are -0.00225778 (PMF)
and -0.0000324505 (PDF); no conservation tolerance was relaxed or exact
conservation claimed. The 50,000-particle time comparison retains 12 calls of
600 s with 50 substeps. Charged kernels have 1,281 exact-zero entries from the
broad input range; final arrays are finite, nonnegative and symmetric. Plots mask
zeros and retain/document intermediate underflow warnings.

The helper starter and doctor now require the native checkout, with no facade
fallback. Native default condensation and coagulation pass their existing 1e-6
mass tolerance (errors 7.9e-16 total and 2.5e-9 particle, respectively). Historical
failed high-loading evidence is retained. Installation instructions distinguish
the native checkout from old released packages. OpenCode payload regenerated.

P2 gate: 64 focused native/example scientific tests and eight plugin distribution
tests passed; generated payload check and diff whitespace check passed.

### P3 implementation and validation

CPU dilution now uses unified gas with three ordered species and native resolved
counts [4,8] in 2 m³, retaining reported densities [2,4]. Identity and protected
mass/charge/volume assertions supplement exact exponential decay. The supported
CPU nucleation script was already native after M4 and remains unchanged.

Wall-loss strategy tutorials retain the original 250-node grid and full 6000 s
(spherical) / 1800 s (rectangular) horizons. Native Euler rejects negative updates,
so internal substeps use ceil(max(k)*output_dt/0.9), respectively 2 and 9.
Independent repeated-Euler expressions check final densities at rtol=1e-12,
atol=1e-30; fixed masses remain identical. Rate labels are m^-3 s^-1, not 1/s.
The combined tutorial uses 150 nodes over 10 nm–10 µm: the original 1 nm–100 µm
coagulation tails produced negative interpolated gains after depletion. The
native admission failure was not bypassed. The formerly skipped condensation
snippet now uses an explicit low-volatility organic fixed reservoir (1e-8 kg/m³,
1e-5 Pa, molar mass 0.12), with full ordered gas configuration. This is an
illustrative input correction, not preservation of the old placeholder physics.
The independent chamber functional example keeps all 1000 times/250 nodes and
asserts finite nonnegative distributions and monotonic total-number loss.

Custom nucleation retains the saturation-dependent law, initial 100-particle
400 nm-radius sample, 0.1 cm³ volume, five stored times and two process substeps.
It removes facade monkeypatches, uses 100,000 fixed slots with unit weights,
and debits gas by newly represented mass divided by volume. Capacity rejects
before gas debit. The original debit omitted the volume factor. Full runs
activate 2,281 then 10,481 nuclei for the seeded initial sample; stochastic
coagulation is assessed through inventory, not exact particle identities.
Independent extensive inventory includes the external vapor source at every
stored time (rtol=1e-12, atol=1e-30). Plot bounds now include the initial 400 nm
mode and 2 nm-radius nuclei; prior 100 nm upper bounds hid the initial population.
A focused fixture checks two nuclei at non-unit volume and capacity failure.

P3 gate: 507 focused tests passed with warnings as errors, including both CPU
wall-loss strategy suites. All six notebook pairs executed without errors; 12
plots inspected. Dilution and supported CPU nucleation standalone runs passed.
