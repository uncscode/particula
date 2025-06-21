"""Utility functions for querying the thermo chemical database."""
from __future__ import annotations

from difflib import get_close_matches

try:  # pragma: no cover - dependency optional
    from thermo.chemical import Chemical
    from thermo.identifiers import name_to_CAS
except ImportError:  # pragma: no cover - dependency missing during import
    Chemical = None
    name_to_CAS = None


def get_chemical_search(identifier: str) -> str | None:
    """Return the resolved chemical name or a close match.

    Parameters
    ----------
    identifier:
        Any string accepted by ``thermo.chemical.Chemical`` (name, formula,
        or CAS number).

    Returns
    -------
    str | None
        ``identifier`` if it directly resolves to a ``Chemical``. If not,
        the closest matching chemical name from the thermo database, or
        ``None`` if no close match is found.
    """
    if Chemical is None or name_to_CAS is None:
        raise ImportError(
            "The 'thermo' package is required for chemical search operations. "
            "Please install it using 'pip install thermo'."
        )

    try:
        Chemical(identifier)
        return identifier
    except Exception:
        names = list(name_to_CAS.keys())
        matches = get_close_matches(identifier, names, n=1, cutoff=0.6)
        return matches[0] if matches else None
