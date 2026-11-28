"""Stream one delimited measurement export, preserving physical source spans."""

import csv
import hashlib
import io
import os
import re
import sys
from collections import Counter
from itertools import chain, takewhile
from pathlib import Path
from threading import Lock

import polars as pl

from particula.data._binding import Binding
from particula.data.profile import ProfileError, digest, load_profile

_CSV_LIMIT_LOCK = Lock()

ISSUE_SCHEMA = {
    "code": pl.String,
    "source_line_start": pl.UInt64,
    "source_line_end": pl.UInt64,
    "message": pl.String,
}
# This narrow grammar covers plain, complete 3010 records. Other exports and
# every exceptional record go through the quote-aware profile decoder.
_PLAIN = re.compile(
    rb"[12][0-9]{9},[0-9]{1,9},[0-9]{1,3}\.[0-9]{1,4},[0-9]{1,3},[0-9]{1,3}\.[0-9]{1,6}\r?\n"
)
_FRAGMENT = re.compile(r"[0-9]{10},(?:[0-9]+(?:\.[0-9]+)?,){0,2}")
_COMPLETE = re.compile(r"[0-9]{10}(?:,[0-9]+(?:\.[0-9]+)?){4}")


class _Source(io.RawIOBase):
    """Hash bytes received from the file and cap each underlying read."""

    def __init__(self, stream, chunk, hasher):
        self.stream, self.chunk, self.hasher = stream, chunk, hasher

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.stream.readinto(memoryview(buffer)[: self.chunk])
        self.hasher.update(memoryview(buffer)[:count])
        return count


class _Lines:
    def __init__(self, reader, encoding, maximum, first):
        self.reader, self.encoding, self.maximum = reader, encoding, maximum
        self.line, self.used = first, 0

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.reader.readline(self.maximum - self.used + 1)
        if not raw:
            raise StopIteration
        self.line += 1
        self.used += len(raw)
        if self.used > self.maximum:
            raise ProfileError(
                "record_size",
                "",
                f"Record exceeds byte limit at line {self.line}",
                source_line_start=self.first,
                source_line_end=self.line,
            )
        encoding = self.encoding
        # A concatenated export can start with a new signature. Preserve a
        # literal U+FEFF on continuation lines of a quoted field.
        if encoding == "utf-8-sig" and self.line != self.first:
            encoding = "utf-8"
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ProfileError(
                "encoding",
                "/input/encoding",
                f"Invalid text at line {self.line}",
            ) from exc


def _next_record(rows, maximum):
    # The stdlib shares its character limit across readers. Only raise it
    # while decoding a record, restoring it before yielding to the caller.
    with _CSV_LIMIT_LOCK:
        original = csv.field_size_limit()
        if original >= maximum:
            return next(rows)
        csv.field_size_limit(min(maximum, sys.maxsize))
        try:
            return next(rows)
        finally:
            csv.field_size_limit(original)


def _records(lines, delimiter):
    rows = csv.reader(lines, delimiter=delimiter, strict=True)
    while True:
        start = lines.line + 1
        lines.used = 0
        lines.first = start
        try:
            row = _next_record(rows, lines.maximum)
        except StopIteration:
            return
        except csv.Error as exc:
            raise ProfileError(
                "csv_record", "/input", f"Lines {start}–{lines.line}: {exc}"
            ) from exc
        yield row, start, lines.line, lines.used


def _layout(records, profile):
    preamble = {}
    header = None
    index = profile["input"]["header_record"]
    if index is not None:
        try:
            for _ in range(index):
                row, *_ = next(records)
                preamble.update(zip(row[::2], row[1::2], strict=False))
            header = next(records)[0]
        except StopIteration as exc:
            raise ProfileError(
                "missing_header",
                "/input/header_record",
                f"Expected header at record {index}",
            ) from exc
    return Binding(profile, header, preamble), header


def _sections(records, profile, binding, header):
    """Split concatenated AIM exports when the preamble or header recurs."""
    if "spectrum" not in profile:
        yield binding, header, records
        return
    first_key = next(iter(binding.preamble), None)
    axis = profile["spectrum"]["axis"]
    while True:
        boundary = []

        def is_observation(record, boundary=boundary):
            row = record[0]
            repeated_preamble = first_key is not None and row[:1] == [first_key]
            repeated_header = axis["after"] in row and axis["before"] in row
            if repeated_preamble or repeated_header:
                boundary.append(record)
                return False
            return True

        yield binding, header, takewhile(is_observation, records)
        if not boundary:
            return
        row = boundary[0][0]
        if row[:1] == [first_key]:
            binding, header = _layout(chain(boundary, records), profile)
        else:
            header = row
            binding = Binding(profile, header, binding.preamble)


def inspect_source(source, profile) -> dict:
    """Bind selectors and return the source layout and clock example."""
    profile = load_profile(profile)
    with Path(source).open("rb") as stream:
        lines = _Lines(stream, profile["input"]["encoding"], 2**22, 0)
        rows = _records(lines, profile["input"]["delimiter"])
        binding, header = _layout(rows, profile)
        example = next(rows, None)
    return {
        "encoding": profile["input"]["encoding"],
        "clock_example": (
            [
                example[0][i]
                for i in binding.clock_columns
                if i < len(example[0])
            ]
            if example
            else None
        ),
        "source_line_start": example[1] if example else None,
        "header": header,
        "preamble": binding.preamble,
        "grids": binding.grids,
    }


def _fast_counter(block, start):
    names = ["seconds", "count_rate", "celsius", "aux1", "aux2"]
    data = pl.read_csv(
        io.BytesIO(block),
        has_header=False,
        schema=dict(
            zip(
                names,
                [pl.Int64, pl.Float64, pl.Float64, pl.String, pl.String],
                strict=True,
            )
        ),
        columns=[0, 1, 2],
    )
    line = pl.int_range(start, start + pl.len(), dtype=pl.UInt64)
    return data.select(
        line.alias("scan_id"),
        line.alias("source_line_start"),
        line.alias("source_line_end"),
        pl.lit(False).alias("repaired"),
        (pl.col("seconds") * 10**9)
        .cast(pl.Datetime("ns", "UTC"))
        .alias("time_utc"),
        "count_rate",
        (pl.col("celsius") + 273.15).alias("temperature"),
    )


def _plain_batches(stream, hasher, size, row_limit):
    line = 1
    while True:
        pos = stream.tell()
        block = stream.read(size)
        stop = block.rfind(b"\n") + 1
        if stop == 0:
            stream.seek(pos)
            return
        block = block[:stop]
        rest, count = _PLAIN.subn(b"", block)
        if rest:
            stream.seek(pos)
            return
        if count > row_limit:
            stop = sum(
                len(s) for s in block.splitlines(keepends=True)[:row_limit]
            )
            block = block[:stop]
            count = row_limit
        stream.seek(pos + stop)
        hasher.update(block)
        yield _fast_counter(block, line)
        line += count


def _events(records, profile):
    """Recognize only a comma-terminated prefix and its numeric continuation."""
    pending = None
    repair = profile["on_bad_record"] == "join_3010_logger_fragments"
    for row, start, end, size in records:
        raw = ",".join(row) if repair and all("," not in x for x in row) else ""
        if pending:
            prefix, first, total = pending
            pending = None
            if (
                start == end
                and not re.match(r"^[0-9]{10},", raw)
                and _COMPLETE.fullmatch(prefix + raw)
            ):
                yield (
                    (prefix + raw).split(","),
                    first,
                    end,
                    total + size,
                    "split_record_repaired",
                )
                continue
            yield None, first, first, total, "unresolved_fragment"
        if repair and start == end and _FRAGMENT.fullmatch(raw):
            pending = raw, start, size
        else:
            yield row, start, end, size, None
    if pending:
        _, first, total = pending
        yield None, first, first, total, "unresolved_fragment"


class MeasurementReader:
    """Stream batches containing scan and diagnostic frames."""

    def __init__(
        self,
        source,
        profile,
        *,
        input_chunk_bytes=2**20,
        batch_scans=32768,
        batch_bytes=2**23,
        max_record_bytes=2**22,
    ):
        """Set independent limits for raw chunks, batches and records."""
        self.source = Path(source)
        self.profile = load_profile(profile)
        limits = input_chunk_bytes, batch_scans, batch_bytes, max_record_bytes
        if any(type(limit) is not int or limit <= 0 for limit in limits):
            raise ValueError("Reader limits must be positive integers")
        self.chunk, self.rows, self.bytes, self.maximum = limits
        self.issue_counts = Counter()
        self.count = 0
        self.manifest = None
        self.segment = 0

    def batches(self):
        """Yield bounded batches and finalize provenance at EOF."""
        self.manifest = None
        self.issue_counts.clear()
        self.segment = 0
        hasher = hashlib.sha256()
        with self.source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            initial = 0
            if self.profile == load_profile("cpc-3010") and self.maximum >= 64:
                # Each normalized scalar row occupies ~50 bytes.
                size = min(self.chunk, self.bytes // 3, self.rows * 30)
                for data in _plain_batches(stream, hasher, size, self.rows):
                    initial += data.height
                    yield data, pl.DataFrame(schema=ISSUE_SCHEMA)
            self.count = initial
            buffered = io.BufferedReader(_Source(stream, self.chunk, hasher))
            cfg = self.profile["input"]
            physical = _Lines(buffered, cfg["encoding"], self.maximum, initial)
            records = _records(physical, cfg["delimiter"])
            binding, header = _layout(records, self.profile)
            segments, grids = [], {}
            for section, (layout, labels, rows) in enumerate(
                _sections(records, self.profile, binding, header)
            ):
                self.segment = section
                segments.append(
                    {
                        "header": labels,
                        "preamble": layout.preamble,
                        "grid_id": layout.grid_id,
                    }
                )
                grids.update(layout.grids)
                yield from self._decoded(rows, layout)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Source changed during import")
            self.manifest = {
                "format_version": 1,
                "parser_version": "1",
                "source": self.source.name,
                "source_sha256": hasher.hexdigest(),
                "source_bytes": stream.tell(),
                "profile": self.profile,
                "profile_sha256": digest(self.profile),
                "header": header,
                "preamble": binding.preamble,
                "grids": grids,
                "segments": segments,
                "scan_count": self.count,
                "issue_counts": dict(self.issue_counts),
                "scan_identity": ["source_sha256", "scan_id"],
            }

    def _decoded(self, records, binding):
        rows, issues = [], []
        budget = 0
        for row, start, end, size, code in _events(records, self.profile):
            if size > self.maximum:
                raise ProfileError(
                    "record_size",
                    "",
                    f"Record exceeds byte limit at lines {start}–{end}",
                    source_line_start=start,
                    source_line_end=end,
                )
            if rows or issues:
                if (
                    max(len(rows), len(issues)) >= self.rows
                    or budget + size * 8 > self.bytes
                ):
                    yield _frames(rows, issues, binding)
                    rows, issues, budget = [], [], 0
            budget += size * 8
            if row is None:
                issues.append(
                    self._issue(code, start, end, "Incomplete logger record")
                )
                continue
            try:
                data = binding.decode(
                    row, start, end, start != end and code is not None
                )
            except (IndexError, ValueError, OverflowError) as exc:
                if self.profile["on_bad_record"] == "error":
                    raise ProfileError(
                        "bad_record",
                        "",
                        f"Lines {start}–{end}: {exc}",
                        source_line_start=start,
                        source_line_end=end,
                    ) from exc
                issues.append(self._issue("bad_record", start, end, str(exc)))
                continue
            rows.append(data)
            self.count += 1
            if code:
                issues.append(
                    self._issue(
                        code, start, end, "Joined two physical logger lines"
                    )
                )
            issues.extend(self._missing(row, binding, start, end))
        yield _frames(rows, issues, binding)

    def _missing(self, row, binding, start, end):
        for field in binding.fields:
            for i in field.indices:
                if row[i].strip() in field.missing:
                    yield self._issue(
                        "missing_token",
                        start,
                        end,
                        f"{field.name} column {i}: {row[i]!r}",
                    )

    def _issue(self, code, start, end, message):
        self.issue_counts[code] += 1
        return {
            "code": code,
            "source_line_start": start,
            "source_line_end": end,
            "message": message,
        }


def _frames(rows, issues, binding):
    return (
        pl.from_dicts(rows, schema=binding.schema)
        if rows
        else pl.DataFrame(schema=binding.schema),
        pl.from_dicts(issues, schema=ISSUE_SCHEMA)
        if issues
        else pl.DataFrame(schema=ISSUE_SCHEMA),
    )
