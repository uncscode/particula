"""Translate beta stream settings while preserving the original filter audit."""

from copy import deepcopy
from dataclasses import dataclass

from particula.data.profile import ProfileError, describe_profile, load_profile

_SUPPORTED = {
    "relative_data_folder",
    "filename_regex",
    "MIN_SIZE_BYTES",
    "data_loading_function",
    "header_row",
    "data_checks",
    "data_column",
    "data_header",
    "time_column",
    "time_format",
    "delimiter",
    "time_shift_seconds",
    "timezone_identifier",
    "data_sizer_reader",
}
# Definitions for the actual scalar headers in the beta's AIM fixtures.
_AIM_FIELDS = {
    "Lower Size (nm)": ("lower_size", "length", "nm", "m"),
    "Upper Size (nm)": ("upper_size", "length", "nm", "m"),
    "Sample Temp (C)": ("sample_temperature", "temperature", "degC", "K"),
    "Sample Pressure (kPa)": ("sample_pressure", "pressure", "kPa", "Pa"),
    "Relative Humidity (%)": (
        "relative_humidity",
        "relative_humidity",
        "%",
        "%",
    ),
    "Median (nm)": ("median_diameter", "length", "nm", "m"),
    "Mean (nm)": ("mean_diameter", "length", "nm", "m"),
    "Geo. Mean (nm)": ("geometric_mean_diameter", "length", "nm", "m"),
    "Mode (nm)": ("mode_diameter", "length", "nm", "m"),
    "Geo. Std. Dev.": ("geometric_standard_deviation", "ratio", "1", "1"),
    "Total Conc. (#/cm³)": (
        "total_number",
        "number_concentration",
        "1/cm^3",
        "1/m^3",
    ),
}


@dataclass
class Migration:
    """Result of translating a beta stream configuration.

    Attributes:
        profile: Candidate import profile with the original settings under
            `legacy_settings`.
        errors: Diagnostics for settings requiring a migration decision.
    """

    profile: dict
    errors: list[ProfileError]

    def resolved_profile(self) -> dict:
        """Return the validated profile when every legacy option is resolved.

        Raises:
            ProfileError: Migration recorded an unresolved setting or the
                candidate profile fails validation.
        """
        if self.errors:
            raise self.errors[0]
        return load_profile(self.profile)


def _scalar_fields(settings, overrides, errors):
    fields = []
    columns = settings.get("data_column", [])
    labels = settings.get("data_header", [])
    if len(columns) != len(labels):
        errors.append(
            ProfileError(
                "legacy_columns",
                "/data_header",
                "Supply one label per selected column",
            )
        )
    counter = describe_profile("cpc-3010")["fields"]
    known = {"CPC_count[#/sec]": counter[0], "Temperature[degC]": counter[1]}
    for i, (column, label) in enumerate(zip(columns, labels, strict=False)):
        spec = overrides.get(label, known.get(label))
        if spec is None and column in _AIM_FIELDS:
            name, quantity, source, target = _AIM_FIELDS[column]
            spec = dict(
                name=name,
                quantity=quantity,
                source_unit=source,
                output_unit=target,
                dtype="float64",
                description=label,
            )
        if spec is None:
            errors.append(
                ProfileError(
                    "legacy_units",
                    f"/data_header/{i}",
                    f"Supply quantity and units for {label}",
                )
            )
            continue
        fields.append(dict(spec, column=column))
    return fields


def _legacy_header(settings, on_bad_record, issue):
    checks = settings.get("data_checks", {})
    skip = checks.get("skip_rows", 0)
    header = settings.get("header_row", 0)
    if (
        checks.get("skip_end", 0)
        or checks.get("replace_chars")
        or skip not in (0, header + 1)
    ):
        issue(
            "/data_checks",
            "Resolve trimming and character replacement before migration",
        )
    if set(checks).difference(
        {"skip_rows", "skip_end", "replace_chars", "characters", "char_counts"}
    ):
        issue("/data_checks", "Unknown legacy row filter")
    if (
        checks.get("characters") or checks.get("char_counts")
    ) and on_bad_record is None:
        issue(
            "/data_checks",
            "Choose a bad-record policy to replace legacy character filters",
        )
    return header if skip else None


def _sizer_options(profile, settings, issue):
    settings_2d = settings.get("data_sizer_reader", {})
    if (
        settings_2d.get("convert_scale_from", "dw/dlogdp").lower()
        != "dw/dlogdp"
    ):
        issue(
            "/data_sizer_reader/convert_scale_from",
            "Integrated bins require explicit conversion",
        )
    axis = profile["spectrum"]["axis"]
    for key, target in (
        ("Dp_start_keyword", "first"),
        ("Dp_end_keyword", "last"),
    ):
        axis[target] = settings_2d.get(key)
    if set(settings_2d).difference(
        {"Dp_start_keyword", "Dp_end_keyword", "convert_scale_from"}
    ):
        issue("/data_sizer_reader", "Unsupported sizer option")


def _legacy_clock(settings):
    clock = settings.get("time_format")
    cols = settings.get("time_column", [])
    zone = settings.get("timezone_identifier")
    if clock == "epoch" and len(cols) == 1:
        clock_spec = {
            "kind": "unix",
            "unit": "s",
            "timezone": zone,
            "column": cols[0],
        }
    else:
        clock_spec = {
            "kind": "calendar",
            "format": clock,
            "timezone": zone,
            "columns": cols,
        }
    clock_spec["shift_seconds"] = settings.get("time_shift_seconds", 0)
    return clock_spec


def migrate_settings(
    settings: dict,
    *,
    fields: dict | None = None,
    on_bad_record: str | None = None,
) -> Migration:
    """Translate a beta 1D or AIM 2D settings mapping.

    Args:
        settings: Beta scalar or sizer stream settings, including its clock,
            column selectors and source discovery rules.
        fields: Scalar field definitions keyed by legacy output label. Supply
            the quantity, source unit and output unit for unfamiliar channels.
        on_bad_record: Profile policy for malformed rows. An explicit choice
            is required to replace legacy character and length filters.

    Returns:
        Candidate profile, original settings and diagnostics for unresolved
        choices. Call resolved_profile() before using the result for import.
    """
    errors: list[ProfileError] = []

    def issue(path, message):
        errors.append(ProfileError("legacy_setting", path, message))

    for key in set(settings).difference(_SUPPORTED):
        issue("/" + key, "Unsupported setting")
    function = settings.get("data_loading_function")
    if function not in ("general_1d_load", "general_2d_load"):
        issue(
            "/data_loading_function", "Expected a beta scalar or sizer reader"
        )
    sizer = function == "general_2d_load"
    profile = describe_profile("smps-aim" if sizer else "cpc-3010")
    profile["profile_id"] = "legacy_delimited"
    profile["description"] = "Translated beta delimited export"
    profile["legacy_settings"] = deepcopy(settings)
    header = _legacy_header(settings, on_bad_record, issue)
    profile["on_bad_record"] = on_bad_record or "error"
    cfg = profile["input"]
    cfg["delimiter"] = settings.get("delimiter", ",")
    cfg["header_record"] = header
    profile["time"] = _legacy_clock(settings)
    if sizer:
        _sizer_options(profile, settings, issue)
    else:
        profile["fields"] = _scalar_fields(settings, fields or {}, errors)
        if profile["on_bad_record"] != "join_3010_logger_fragments":
            cfg.pop("record_width", None)
        if profile["fields"] != describe_profile("cpc-3010")["fields"]:
            profile.pop("instrument_family", None)
            profile.pop("measurement_principle", None)
    if not errors:
        try:
            load_profile(profile)
        except ProfileError as exc:
            errors.append(exc)
    if errors:
        profile["unresolved_settings"] = [
            {"code": e.code, "path": e.path, "message": e.message}
            for e in errors
        ]
    return Migration(profile, errors)
