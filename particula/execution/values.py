"""Dependency-neutral values shared by execution package boundaries."""

from enum import Enum


class FallbackPolicy(str, Enum):
    """Declare whether an eligible capability error is re-raised.

    Or dispatched to CPU fallback.
    """

    RAISE = "raise"
    CPU = "cpu"


class FallbackBoundary(str, Enum):
    """Declare the caller-visible boundary for an explicit fallback request."""

    PRE_UPLOAD = "pre_upload"
    RESTORED = "restored"


class CPUStateAuthority(str, Enum):
    """Declare the asserted authority of CPU state for a fallback request."""

    CPU_AUTHORITATIVE = "cpu_authoritative"
    RESIDENT = "resident"
    UPLOADED = "uploaded"
    MUTATED = "mutated"
