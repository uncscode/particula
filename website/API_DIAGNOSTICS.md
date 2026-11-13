# API extraction diagnostics

The migration retains all 237 package modules from the previous API generator's
coverage rules. Test modules and build helpers are excluded from publication.
The complete extraction log is retained in `.cache/griffe.log`; warnings are
not globally disabled.

The tested Griffe 1.14.0 extraction has three unresolved external aliases:

| Alias in `particula.util.chemical.thermo_import` | External target |
| --- | --- |
| `CAS_from_any` | `chemicals.identifiers.CAS_from_any` |
| `Chemical` | `thermo.chemical.Chemical` |
| `get_pubchem_db` | `chemicals.identifiers.get_pubchem_db` |

These are optional third-party imports, not missing Particula implementations.
External-package resolution is disabled deliberately: the site documents
Particula, and does not generate copies of third-party APIs. No internal alias
was unresolved in the migration extraction.

Five docstring warnings originate in excluded test helpers:

- `particula/execution/tests/resident_benchmark_support.py`: `Availability` is
  parsed as an undocumented parameter and is absent from the function signature
  (two diagnostics).
- `particula/gpu/tests/benchmark_test.py`: two `step_fn` parameters have no type
  annotation; one yielded value has no type annotation (three diagnostics).

These diagnostics do not affect published signatures or module coverage. A new
warning in a published module must be investigated; this record is not a blanket
exception for future warnings. The rendered site verifier checks module routes,
local links and referenced symbol anchors after generation.
