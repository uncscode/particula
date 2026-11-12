"""Host-side interpretation of explicit Warp particle population metadata.

Codes are stable transfer metadata, not algorithm selectors. A known kind
does not establish that a particular numerical method supports it. Zero is
an invalid admission sentinel; it never implies resolved counts.
This module does not import Warp or inspect device payloads.
"""

from numbers import Integral
from typing import Any

_UNTAGGED = 0
DISCRETE = 1
CONTINUOUS_PDF = 2
PARTICLE_RESOLVED = 3

_KINDS = {
    _UNTAGGED: None,
    DISCRETE: "discrete",
    CONTINUOUS_PDF: "continuous_pdf",
    PARTICLE_RESOLVED: "particle_resolved",
}


def particle_distribution_code(particles: Any) -> int:
    """Read a known integer metadata code without touching device arrays.

    Missing metadata yields the invalid zero sentinel for consumer rejection.
    Warp scalar wrappers expose their integer through ``value``. Unknown codes,
    booleans and noninteger values reject without coercion.
    """
    value = getattr(particles, "distribution_kind", _UNTAGGED)
    value = getattr(value, "value", value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("distribution_kind must be a known integer code")
    code = int(value)
    if code not in _KINDS:
        raise ValueError("distribution_kind must be a known integer code")
    return code


def particle_distribution_type(particles: Any) -> str:
    """Decode transferable population metadata, rejecting unsupported PDF."""
    code = particle_distribution_code(particles)
    if code == _UNTAGGED:
        raise ValueError("GPU population requires explicit population metadata")
    if code == CONTINUOUS_PDF:
        raise ValueError("GPU continuous_pdf populations are unsupported")
    return str(_KINDS[code])
