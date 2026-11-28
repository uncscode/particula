"""Bind source selectors, measurement units, coordinates and clocks once."""

import math
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from particula.data.profile import ProfileError, digest
from particula.util.convert_units import get_unit_conversion

_TICKS = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _index(selector, header, width, path):
    if isinstance(selector, int):
        n = selector
    elif header is None or header.count(selector) != 1:
        raise ProfileError(
            "source_column", path, f"Expected one header {selector!r}"
        )
    else:
        n = header.index(selector)
    if width is not None and n >= width:
        raise ProfileError(
            "source_column", path, f"Column {n} exceeds width {width}"
        )
    return n


@lru_cache(maxsize=128)
def _affine(source, target):
    if source == target:
        return 1.0, 0.0
    zero, one = get_unit_conversion(source, target, value=np.array([0.0, 1.0]))
    return float(one - zero), float(zero)


class _Field:
    def __init__(self, spec, indices, missing, shape=()):
        self.name = spec["name"]
        self.indices = indices
        self.missing = missing
        self.scale, self.offset = _affine(
            spec["source_unit"], spec["output_unit"]
        )
        dtype = pl.Float64 if spec["dtype"] == "float64" else pl.Float32
        self.maximum = (
            np.finfo(np.float32).max.item() if dtype == pl.Float32 else math.inf
        )
        self.dtype = pl.Array(dtype, shape) if shape else dtype
        self.shape = shape

    def decode(self, row):
        values = []
        for idx in self.indices:
            token = row[idx].strip()
            if token in self.missing:
                values.append(None)
            else:
                value = float(token) * self.scale + self.offset
                if not math.isfinite(value) or abs(value) > self.maximum:
                    raise ValueError(f"Nonfinite {self.name} at column {idx}")
                values.append(value)
        if not self.shape:
            return values[0]
        if len(self.shape) == 1:
            return values
        return np.asarray(values, dtype=object).reshape(self.shape).tolist()


def _sizer_grid(profile, header, preamble, index):
    spec = profile["spectrum"]
    axis = spec["axis"]
    for k, expected in spec["require_preamble"].items():
        if preamble.get(k) != expected:
            raise ProfileError(
                "preamble",
                "/spectrum/require_preamble",
                f"Expected {k}={expected}",
            )
    for key in spec["preserve_preamble"]:
        if key not in preamble:
            raise ProfileError(
                "preamble",
                "/spectrum/preserve_preamble",
                f"Missing correction or version metadata: {key}",
            )
    start = index(axis["after"], "/spectrum/axis/after") + 1
    stop = index(axis["before"], "/spectrum/axis/before")
    if "first" in axis:
        first = index(axis["first"], "/spectrum/axis/first")
        last = index(axis["last"], "/spectrum/axis/last")
        if not start <= first <= last < stop:
            raise ProfileError(
                "grid", "/spectrum/axis", "Invalid inclusive bounds"
            )
        start, stop = first, last + 1
    try:
        scale, offset = _affine(axis["source_unit"], axis["output_unit"])
        mid = np.array(header[start:stop], dtype=float) * scale + offset
        cpd = float(preamble[axis["channels_per_decade_key"]])
        if not (
            math.isfinite(cpd)
            and cpd > 0
            and mid.size
            and np.all(np.isfinite(mid))
            and np.all(mid > 0)
            and np.all(np.diff(mid) > 0)
        ):
            raise ValueError("Invalid diameter grid or channels per decade")
        # Rounded vendor midpoint labels may differ by one last printed digit.
        nominal = mid[0] * 10.0 ** (np.arange(len(mid)) / cpd)
        tolerance = (
            np.array(
                [
                    float(Decimal(1).scaleb(Decimal(t).as_tuple().exponent))
                    for t in header[start:stop]
                ]
            )
            * scale
        )
        if np.any(
            np.abs(nominal - mid)
            > (tolerance + tolerance[0] * nominal / mid[0])
        ):
            raise ValueError(
                "Midpoints disagree with reported channels per decade"
            )
    except (KeyError, ValueError, InvalidOperation) as exc:
        raise ProfileError("grid", "/spectrum/axis", str(exc)) from exc
    grid = {
        "coordinates": {"diameter": mid.tolist()},
        "axes": [axis],
        "channels_per_decade": cpd,
        "log10_bin_width": 1.0 / cpd,
        "edge_convention": "midpoint times 10^(±0.5/channels_per_decade)",
        "source_midpoints": header[start:stop],
    }
    return grid, list(range(start, stop))


class Binding:
    """Unit-checked mapping from one input layout to a typed scan frame."""

    def __init__(self, profile, header, preamble):
        self.profile = profile
        self.width = profile["input"].get(
            "record_width", len(header) if header else None
        )

        def index(selector, path):
            return _index(selector, header, self.width, path)

        clock = profile["time"]
        self.clock = clock
        shift = Decimal(str(clock.get("shift_seconds", 0))) * 10**9
        if shift != shift.to_integral_value():
            raise ProfileError(
                "clock_shift",
                "/time/shift_seconds",
                "Clock shift must resolve to whole nanoseconds",
            )
        self.shift_ns = int(shift)
        self.zone = ZoneInfo(clock["timezone"])
        selectors = clock.get("columns", [clock.get("column")])
        self.clock_columns = [index(x, "/time") for x in selectors]
        self.schema = {
            "scan_id": pl.UInt64,
            "source_line_start": pl.UInt64,
            "source_line_end": pl.UInt64,
            "repaired": pl.Boolean,
            "time_utc": pl.Datetime("ns", "UTC"),
        }
        self.fields = []
        self.grids = {}
        self.grid_id = None
        self.preamble = preamble
        missing = frozenset(profile["missing_tokens"])
        for field in profile.get("fields", []):
            self.fields.append(
                _Field(field, [index(field["column"], "/fields")], missing)
            )
        grid = {}
        if "spectrum" in profile:
            grid, indices = _sizer_grid(profile, header or [], preamble, index)
            self.fields.append(
                _Field(profile["spectrum"], indices, missing, (len(indices),))
            )
        for field in profile.get("arrays", []):
            indices = [index(c, "/arrays") for c in field["columns"]]
            axes = field["axes"]
            self.fields.append(
                _Field(
                    field,
                    indices,
                    missing,
                    tuple(len(a["values"]) for a in axes),
                )
            )
            grid.setdefault("variables", {})[field["name"]] = axes
        if grid:
            self.grid_id = digest(grid)
            self.grids[self.grid_id] = grid
            self.schema["grid_id"] = pl.String
        self.schema.update({f.name: f.dtype for f in self.fields})
        self.flag = None
        if "vendor_flags" in profile:
            f = profile["vendor_flags"]
            self.flag = (index(f["column"], "/vendor_flags"), f["separator"])
            self.schema.update(
                vendor_status=pl.String, vendor_flags=pl.List(pl.String)
            )
        self.timing = {}
        if "scan_timing" in profile:
            t = profile["scan_timing"]
            for key in ("scan", "retrace"):
                name = key + "_duration_s"
                self.timing[name] = (
                    index(t[key + "_duration_column"], "/scan_timing"),
                    0.001 if t["unit"] == "ms" else 1.0,
                )
                self.schema[name] = pl.Float64

    def timestamp(self, row):
        """Parse epoch ticks or an unambiguous local wall time to UTC ns."""
        if self.clock["kind"] == "unix":
            try:
                ticks = (
                    Decimal(row[self.clock_columns[0]])
                    * _TICKS[self.clock["unit"]]
                )
                if not ticks.is_finite() or ticks != ticks.to_integral_value():
                    raise ValueError("Invalid epoch precision")
                result = int(ticks)
            except InvalidOperation as exc:
                raise ValueError("Invalid epoch time") from exc
        else:
            text = " ".join(row[i] for i in self.clock_columns)
            dt = datetime.strptime(text, self.clock["format"])
            if dt.tzinfo is None:
                options = set()
                for fold in (0, 1):
                    instant = dt.replace(
                        tzinfo=self.zone, fold=fold
                    ).astimezone(UTC)
                    if instant.astimezone(self.zone).replace(tzinfo=None) == dt:
                        options.add(instant)
                if len(options) != 1:
                    raise ValueError(
                        f"Ambiguous or nonexistent local time {text}"
                    )
                dt = options.pop()
            delta = dt.astimezone(UTC) - _EPOCH
            result = (
                delta.days * 86400 + delta.seconds
            ) * 10**9 + delta.microseconds * 1000
        result += self.shift_ns
        if not -(2**63) <= result < 2**63:
            raise ValueError("Timestamp outside nanosecond range")
        return result

    def decode(self, row, start, end, repaired=False):
        """Decode a record using its source line as its local scan ID."""
        if self.width is not None and len(row) != self.width:
            raise ValueError(f"Expected {self.width} fields; found {len(row)}")
        data = {
            "scan_id": start,
            "source_line_start": start,
            "source_line_end": end,
            "repaired": repaired,
            "time_utc": self.timestamp(row),
        }
        if self.grid_id:
            data["grid_id"] = self.grid_id
        for field in self.fields:
            data[field.name] = field.decode(row)
        if self.flag:
            col, sep = self.flag
            data["vendor_status"] = row[col]
            data["vendor_flags"] = [v for v in row[col].split(sep) if v]
        for name, (col, scale) in self.timing.items():
            seconds = float(row[col]) * scale
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid {name}")
            data[name] = seconds
        return data
