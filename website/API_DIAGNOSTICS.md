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

## Legacy symbol coverage

`legacy-api-anchors.json` records all 1,860 Python symbol and docstring anchors
from 237 API pages in publication revision
`6597906ba8dd09bdccb9c3e502d8155c6668a1d4`. The build checks every recorded
anchor, in addition to module coverage and links used within the new site.
Special methods remain documented. For concrete-module members previously
published outside `__all__`, the documentation-only Griffe model extends its
visibility list with those existing members. This does not change Python
package exports or promote concrete-only interfaces to public APIs.

Module anchors target the page heading. MkDocs-specific nested docstring
anchors target the owning symbol's section. Missing actual symbols fail the
build rather than receiving a misleading alias to the top of the page.
