"""Controlled sign and magnitude patterns for matched one-box charge sweeps."""

import re


def charge_mode(label):
    """Return the sign pattern and integer elementary charges per particle."""
    if label == "neutral":
        return "neutral", 0
    token = re.fullmatch(r"(like|bipolar|segregated)([1-9][0-9]*)?", label)
    if token is None:
        raise ValueError(f"Unknown initial charge pattern {label}")
    kind = token.group(1)
    default = 8 if kind == "segregated" else 4
    return kind, int(token.group(2) or default)


def canonical_charge(label):
    """Use the same identity for equivalent initial charges."""
    return {
        "bipolar4": "bipolar",
        "like4": "like",
        "segregated8": "segregated",
    }.get(label, label)


def initial_charge(index, size_quantile, label, where):
    """Apply the same prescription to NumPy and CUDA particle identifiers."""
    kind, q = charge_mode(label)
    if kind in ("neutral", "like"):
        return index * 0 + q
    if kind == "bipolar":
        return where(index % 2 == 0, q, -q)
    return where(size_quantile > 0.5, q, -q)
