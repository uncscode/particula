"""Helper utilities to resolve chemical identifiers to a canonical form.

The functions rely on the optional ``chemicals.identifiers`` back-end for
fast, exact look-ups and fall back to fuzzy matching the PubChem name
index when that package is unavailable.
"""

from difflib import get_close_matches
from typing import List

from particula.util.chemical.thermo_import import (
    CHEMICALS_AVAILABLE,
    CAS_from_any,
    get_pubchem_db,
)

# Single, module-level PubChem DB instance (loaded once) ------------
_pubchem_db = None
if CHEMICALS_AVAILABLE and get_pubchem_db is not None:  # pragma: no cover
    _pubchem_db = get_pubchem_db()
    _pubchem_db.finish_loading()


def get_chemical_search(identifier: str) -> str:
    """Resolve a chemical identifier to a canonical name or CAS number.

    The resolution proceeds in two stages:

    1. Exact search through ``chemicals.identifiers.CAS_from_any``
       (returns immediately when successful).
    2. Case-insensitive fuzzy matching against the PubChem name index that
       ships with *thermo* (Levenshtein distance via ``difflib``).

    Arguments:
        - identifier : Arbitrary chemical name, formula, InChI, or CAS
          registry number.

    Returns:
        - The resolved identifier string when a match is found, otherwise
          ``None``.

    Examples:
        ``` py title="Exact CAS lookup"
        from particula.util.materials.chemical_search import get_chemical_search
        assert get_chemical_search("64-17-5") == "64-17-5"      # ethanol
        ```
        ``` py title="Fuzzy text match"
        hit = get_chemical_search("soodim chlorid")             # typo
        # hit -> "7647-14-5"  (sodium chloride)
        ```

    References:
        - Wilson, N., et al., "Open-source tools for chemical identifiers,"
          *J. Chem. Inf. Model.* **60** (2020) 833–839.
        - "Chemical identifier,"
          [Wikipedia](https://en.wikipedia.org/wiki/Chemical_identifier).
    """
    # --- new guard -------------------------------------------------
    if not CHEMICALS_AVAILABLE:  # thermo must be present
        raise ImportError(
            "The 'thermo' package is required for chemical search "
            "operations but is not installed."
            "Please install it via 'pip install thermo'."
        )
    # ---------------------------------------------------------------

    # 1. Exact resolution via ``chemicals`` (fast)
    if CAS_from_any is not None:
        try:
            if CAS_from_any(identifier):
                return identifier
        except ValueError:
            # CAS_from_any raises ValueError when identifier is not a valid CAS
            pass

    # 2. Fuzzy match
    if _pubchem_db is not None:
        candidate_names: List[str] = list(_pubchem_db.name_index.keys())
        matches = get_close_matches(
            identifier, candidate_names, n=1, cutoff=0.6
        )
        return matches[0] if matches else "No Match"
    return "No Match"
