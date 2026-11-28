"""Portable measurement profiles and source-located diagnostics."""

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from os import PathLike
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator

ProfileSource = str | PathLike[str] | dict

_BUILTINS = ("cpc-3010", "smps-aim")
_RESERVED = {
    "scan_id",
    "source_sha256",
    "time_utc",
    "source_line_start",
    "source_line_end",
    "repaired",
    "grid_id",
    "vendor_status",
    "vendor_flags",
    "scan_duration_s",
    "retrace_duration_s",
}


class ProfileError(ValueError):
    """A machine-readable diagnostic for a profile or its source binding."""

    def __init__(
        self,
        code,
        path,
        message,
        *,
        observed=None,
        expected=None,
        source_line_start=None,
        source_line_end=None,
    ):
        """Attach a stable code and JSON Pointer to the diagnostic."""
        self.severity = "error"
        self.observed = observed
        self.expected = expected
        self.source_line_start = source_line_start
        self.source_line_end = source_line_end
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path or '/'}: {message}")


@dataclass(frozen=True)
class ValidationResult:
    """Result of checking one profile and optionally its source header."""

    errors: tuple[ProfileError, ...] = ()

    def raise_for_errors(self):
        """Raise the first diagnostic, when present."""
        if self.errors:
            raise self.errors[0]


def _pairs(pairs):
    mapping = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in mapping:
            raise ProfileError(
                "invalid_key", "", f"Duplicate or non-string key: {key!r}"
            )
        mapping[key] = value
    return mapping


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        return _pairs(self.construct_pairs(node, deep=True))


def canonical(value):
    """Serialize JSON values deterministically for provenance."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def digest(value):
    """Return the SHA-256 of a normalized definition."""
    return hashlib.sha256(canonical(value)).hexdigest()


def _resource(name):
    return (
        files("particula.data")
        .joinpath("profiles", name)
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def _validator():
    return Draft202012Validator(json.loads(_resource("profile.schema.json")))


def describe_profile(profile: ProfileSource | None = None) -> dict:
    """Return the installed schema or a validated profile."""
    if profile is None:
        return json.loads(_resource("profile.schema.json"))
    return load_profile(profile)


def _decode(profile):
    if isinstance(profile, dict):
        return json.loads(canonical(profile), object_pairs_hook=_pairs)
    if isinstance(profile, str) and profile in _BUILTINS:
        return json.loads(_resource(profile + ".json"))
    path = Path(profile)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw, object_pairs_hook=_pairs)
    if path.suffix.lower() in (".yaml", ".yml"):
        # Reject aliases before construction to keep the profile a finite tree.
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(raw)):
            raise ProfileError(
                "alias", "", "Use explicit JSON-compatible values"
            )
        return yaml.load(raw, Loader=_Loader)  # noqa: S506
    raise ProfileError("profile_format", "", "Expected a YAML or JSON profile")


def _validate_array(field, index):
    axes = field["axes"]
    names = [axis["name"] for axis in axes]
    columns = field["columns"]
    shape = [len(axis["values"]) for axis in axes]
    if (
        len(set(names)) != len(names)
        or math.prod(shape) != len(columns)
        or len(set(columns)) != len(columns)
    ):
        raise ProfileError(
            "array_shape",
            f"/arrays/{index}",
            "Expected distinct axes and columns matching the shape",
        )
    for j, axis in enumerate(axes):
        values = axis["values"]
        if not (
            values == sorted(values) or values == sorted(values, reverse=True)
        ):
            raise ProfileError(
                "axis_order",
                f"/arrays/{index}/axes/{j}/values",
                "Coordinates must increase or decrease along each axis",
            )


def load_profile(profile: ProfileSource) -> dict:
    """Load and validate a built-in ID, mapping, YAML or JSON file."""
    try:
        value = _decode(profile)
        canonical(value)
    except ProfileError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ProfileError("profile_decode", "", str(exc)) from exc
    if isinstance(value, dict) and value.get("unresolved_settings"):
        raise ProfileError(
            "legacy_unresolved",
            "/unresolved_settings",
            "Resolve the migration diagnostics before importing",
        )
    error = next(_validator().iter_errors(value), None)
    if error:
        pointer = "".join(
            "/" + str(x).replace("~", "~0").replace("/", "~1")
            for x in error.absolute_path
        )
        raise ProfileError(
            "profile_schema",
            pointer,
            error.message,
            observed=error.instance,
            expected=error.validator_value,
        )
    names = set(_RESERVED)
    for key, fields in [
        ("fields", value.get("fields", [])),
        ("spectrum", [value["spectrum"]] if "spectrum" in value else []),
        ("arrays", value.get("arrays", [])),
    ]:
        for i, field in enumerate(fields):
            name = field["name"]
            if name in names:
                raise ProfileError(
                    "name_conflict",
                    f"/{key}",
                    f"Duplicate or reserved name: {name}",
                )
            names.add(name)
            if key == "arrays":
                _validate_array(field, i)
    try:
        ZoneInfo(value["time"]["timezone"])
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ProfileError("timezone", "/time/timezone", str(exc)) from exc
    return value


def validate_profile(
    profile: ProfileSource, *, source=None
) -> ValidationResult:
    """Check profile semantics and optionally bind a source header."""
    try:
        value = load_profile(profile)
        if source is not None:
            from particula.data.reader import inspect_source

            inspect_source(source, value)
    except ProfileError as exc:
        return ValidationResult((exc,))
    return ValidationResult()
