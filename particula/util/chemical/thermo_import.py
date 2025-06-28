"""Imports for thermo and chemicals libraries, if available."""

try:  # optional ‑– thermo, chemicals
    from chemicals.identifiers import (  # type: ignore
        CAS_from_any,
        get_pubchem_db,
    )
    from thermo.chemical import Chemical  # type: ignore

    CHEMICALS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CAS_from_any = None  # type: ignore
    get_pubchem_db = None  # type: ignore
    CHEMICALS_AVAILABLE = False
    Chemical = None  # type: ignore

__all__ = [
    "Chemical",
    "CAS_from_any",
    "get_pubchem_db",
    "CHEMICALS_AVAILABLE",
]
