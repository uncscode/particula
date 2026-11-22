"""Optional direct activity-provider contract for native condensation."""

from typing import Any, Protocol


class PreparedActivityProvider(Protocol):
    """Version 1 provides primitive checkpoint identity and device binding.

    A binding owns all workspace and a float64 (box, slot, material)
    activity buffer. Its evaluate(masses, counts, temperature, invalid)
    enqueues on the current Warp stream, setting invalid[0] on a failed row.
    Only mapped neutral targets have meaningful output. Status retains original
    box and slot order for diagnosis at an observation or failure boundary.
    """

    prepared_provider_version: int

    def prepared_identity(self) -> tuple:
        """Return immutable layout, physics and parameter identity."""
        ...

    def bind_device(self, shape, targets, device) -> Any:
        """Allocate owner-specific workspace and mapped outputs."""
        ...


def provider_identity(activity):
    """Identify an external strategy that opts into preparation."""
    version = getattr(activity, "prepared_provider_version", None)
    if version is None:
        return None
    if version != 1:
        raise ValueError(f"unsupported prepared activity version {version}")
    return activity.prepared_identity()


def restore_provider(identity):
    """Rebuild a registered backend and verify its scientific identity."""
    if identity[0] == "aiomfac":
        from particula_aiomfac.adapter import AIOMFACActivity

        return AIOMFACActivity.from_identity(identity)
    raise ValueError(f"unknown prepared activity backend {identity[0]}")
