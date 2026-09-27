# Scope

## Modules and directories

- `particula/particles/particle_data.py` and adjacent `tests/`: audit existing
  derived properties and implement only necessary missing access/mutation
  helpers, with explicit concentration interpretation.
- `particula/gas/gas_data.py`, `environment_data.py` and adjacent `tests/`:
  required data-native helpers, ordered metadata and validation behavior.
- A small concrete shared validation helper may be introduced only if the
  approved alignment contract requires it; decide location in P1 without
  creating an aggregate or a second state authority.
- `particula/particles/representation.py`, legacy gas species, native
  condensation/coagulation and GPU conversions are read-only audit references
  except for adjacent characterization/regression tests. Their consumer
  migration and deletion belong to later tracks.
- `docs/Features/data-containers-and-gpu-foundations.md`: bounded developer
  contract update in P5, clearly separating delivered helpers from future M2
  aggregate APIs. Broad tutorials and migration guidance remain M5 work.

## Interfaces and deliverables

1. A reviewed field/units/shape/ownership table and raw-versus-normalized
   concentration ledger for every supported CPU distribution convention.
2. Explicit ordered alignment among unified gas names/mask, particle mass and
   density lanes, environment species lanes and process-owned configuration.
3. Whole-container getter, individual setter and coordinated replacement
   specification with a concrete acceptance matrix for M2.
4. Necessary data-native helpers, co-located scientific and rejection tests,
   and a completed-and-validated handoff packet.

## Excluded and downstream ownership

| Plan | Exclusive downstream ownership |
|---|---|
| E9-M2 | Flat Aerosol, actual aggregate getters/setters and atomic replacement, construction/builders/presets |
| E9-M3 | Condensation, coagulation and wall-loss physics consumer migration |
| E9-M4 | Runnable composition, dilution/nucleation orchestration and CPU adapters |
| E9-M5 | Supported examples, notebooks and comprehensive user documentation |
| E9-M6 | Legacy API/bridge/export deletion last and v0.3.0 readiness |

No new physics, general multi-box CPU execution, container renaming,
precision/storage-schema redesign, permanent facade, GPU public API,
automatic GPU rebinding, hidden transfer/fallback, graph-capture redesign,
performance claim, diagnostics track or Epic I implementation is included.
Do not delete facade-based scientific tests or retained CPU↔Warp helpers.
