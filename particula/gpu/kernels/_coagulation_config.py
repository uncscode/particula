"""Host-only configuration resolution for GPU coagulation mechanisms."""

from dataclasses import dataclass
from types import MappingProxyType

BROWNIAN_MECHANISM = "brownian"
CHARGED_HARD_SPHERE_MECHANISM = "charged_hard_sphere"
SEDIMENTATION_SP2016_MECHANISM = "sedimentation_sp2016"
TURBULENT_SHEAR_ST1956_MECHANISM = "turbulent_shear_st1956"

CANONICAL_COAGULATION_MECHANISMS = (
    BROWNIAN_MECHANISM,
    CHARGED_HARD_SPHERE_MECHANISM,
    SEDIMENTATION_SP2016_MECHANISM,
    TURBULENT_SHEAR_ST1956_MECHANISM,
)

BROWNIAN_MECHANISM_FLAG = 1
CHARGED_HARD_SPHERE_MECHANISM_FLAG = 2
SEDIMENTATION_SP2016_MECHANISM_FLAG = 4
TURBULENT_SHEAR_ST1956_MECHANISM_FLAG = 8

_COAGULATION_MECHANISM_FLAGS = MappingProxyType(
    {
        BROWNIAN_MECHANISM: BROWNIAN_MECHANISM_FLAG,
        CHARGED_HARD_SPHERE_MECHANISM: CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        SEDIMENTATION_SP2016_MECHANISM: SEDIMENTATION_SP2016_MECHANISM_FLAG,
        TURBULENT_SHEAR_ST1956_MECHANISM: TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
    }
)


@dataclass(frozen=True)
class CoagulationMechanismConfig:
    """Configure host-side coagulation mechanism selection.

    Attributes:
        mechanisms: Requested identifiers, or ``None`` for Brownian.
        distribution_type: Required representation; only ``"particle_resolved"``
            is structurally supported.
    """

    mechanisms: tuple[str, ...] | None = None
    distribution_type: str = "particle_resolved"


@dataclass(frozen=True)
class _ResolvedCoagulationMechanismConfig:
    """Store a normalized, structurally valid mechanism configuration."""

    mechanisms: tuple[str, ...]
    distribution_type: str
    mask: int


def resolve_coagulation_mechanism_config(
    config: CoagulationMechanismConfig,
) -> _ResolvedCoagulationMechanismConfig:
    """Resolve a configuration through host-side structural validation.

    Args:
        config: Configuration whose distribution type and mechanism identifiers
            are validated and normalized into canonical mechanism order.

    Returns:
        The normalized configuration, including its additive mechanism mask.

    Raises:
        ValueError: If the distribution type is unsupported, mechanisms are not
            a non-empty tuple of strings, a mechanism is duplicated or unknown.
    """
    if config.distribution_type != "particle_resolved":
        raise ValueError(
            "distribution_type must be exactly 'particle_resolved'."
        )
    mechanisms = config.mechanisms
    if mechanisms is None:
        mechanisms = (BROWNIAN_MECHANISM,)
    if not isinstance(mechanisms, tuple) or not mechanisms:
        raise ValueError("mechanisms must be a non-empty tuple of strings.")
    if not all(isinstance(mechanism, str) for mechanism in mechanisms):
        raise ValueError("mechanisms must contain only string identifiers.")
    seen: set[str] = set()
    for mechanism in mechanisms:
        if mechanism in seen:
            raise ValueError(f"Duplicate coagulation mechanism '{mechanism}'.")
        seen.add(mechanism)
    unknown = next(
        (
            mechanism
            for mechanism in mechanisms
            if mechanism not in _COAGULATION_MECHANISM_FLAGS
        ),
        None,
    )
    if unknown is not None:
        raise ValueError(f"Unknown coagulation mechanism '{unknown}'.")
    normalized_mechanisms = tuple(
        mechanism
        for mechanism in CANONICAL_COAGULATION_MECHANISMS
        if mechanism in mechanisms
    )
    return _ResolvedCoagulationMechanismConfig(
        mechanisms=normalized_mechanisms,
        distribution_type=config.distribution_type,
        mask=sum(
            _COAGULATION_MECHANISM_FLAGS[item] for item in normalized_mechanisms
        ),
    )


def validate_coagulation_mechanism_capabilities(
    resolved: _ResolvedCoagulationMechanismConfig,
) -> None:
    """Enforce the executable and deferred mechanism-mask boundary.

    Args:
        resolved: Structurally valid, normalized configuration to check.

    Returns:
        None.

    Raises:
        ValueError: If the mechanism mask is a deferred three-way combination.
    """
    if resolved.mask in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15):
        return
    raise ValueError("Additive coagulation execution is deferred.")
