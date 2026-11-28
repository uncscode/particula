# Read instrument measurements

Install the optional reader with `pip install "particula[data]"`.
Use `particula.data` for a new delimited format and `particula.instruments`
for a named instrument. A measurement carries a Polars scan table, its units,
the source profile, any coordinate grids, and diagnostics.

## Choose an import profile

`cpc-3010` reads a five-field condensation particle counter log. It returns
a `ParticleCountRate` with detector counts/s and temperature in kelvin.
`smps-aim` reads an AIM number-weighted export. It returns a
`MobilityNumberSpectrum` with an electrical-mobility diameter axis in metres
and number per log₁₀ diameter decade in m⁻³. The profile supplies the clock
zone for the AIM export. Set it for the deployment before interpreting times.

For a custom instrument, start from the closest profile and save the edited
mapping as JSON or plain YAML. This example adapts a counter to a headered
three-column file containing Unix seconds, counts/s and degrees Celsius.

```python
import json
from pathlib import Path

from particula.data import describe_profile, inspect_source, validate_profile
from particula.instruments import Instrument

profile = describe_profile("cpc-3010")
profile.update(profile_id="bench_counter", on_bad_record="report")
profile["input"].update(header_record=0, record_width=3)
profile["time"]["column"] = "seconds"
profile["fields"][0]["column"] = "pulses_per_second"
profile["fields"][1]["column"] = "temp_C"

path = Path("bench-counter.json")
path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
validate_profile(path, source="bench.csv").raise_for_errors()
layout = inspect_source("bench.csv", path)
product = Instrument(path).read("bench.csv")
```

`describe_profile()` returns the installed versioned JSON Schema. It supplies
allowed keys, physical quantities, compatible units and field descriptions to
editors and automated profile authors. Validation errors expose a `code`,
`path` (JSON Pointer), and `message`. A source inspection also returns the
bound header, a sample clock token and the preamble. A calendar clock needs an
explicit timezone. An ambiguous daylight-saving transition produces a
diagnostic. Field-level missing tokens become nulls and retain their original
spelling in the report.

For a beta settings dictionary, call
`migrate_settings(settings, on_bad_record="report")`. Inspect `.errors` and
supply any unknown quantity definitions by legacy output label through
`fields=`. Call `.resolved_profile()` to obtain a validated configuration.
Select the 3010 fragment-repair policy only for its documented logger layout.
The profile retains the legacy file-selection rules and filters for audit.
Supply the source path explicitly.

## Work with a dataset larger than memory

```python
import polars as pl

from particula.data import import_to_parquet, open_measurement

import_to_parquet("bench.csv", "bench-dataset", profile=path)
dataset = open_measurement("bench-dataset")
mean_rate = dataset.lazy_frame.select(pl.col("count_rate").mean()).collect()
repairs = dataset.lazy_issues.filter(pl.col("code") == "split_record_repaired")
```

`.frame` and `.issues` collect all rows in memory. Use `.lazy_frame` and
`.lazy_issues` to project or filter on disk. The reader stages parts in the
destination's parent and publishes a versioned manifest after the last part.
A matching retry reopens the completed product. A different source digest or
profile requires a separate destination. Move the whole directory to relocate
a dataset.

`input_chunk_bytes` controls raw reads. `batch_scans` limits the number of
scans buffered for a part and `batch_bytes` provides a second budget for wide
or irregular input. A single logical record may occupy a batch by itself.
`max_record_bytes` caps the size of an individual CSV record. Physical line
spans are one-based and inclusive. `(source_sha256, scan_id)` identifies a
reading across source files, filters and batch sizes.

## Work with spectra and matrices

Within one grid, `number_spectrum` is a fixed-shape Polars Array column.
Retrieve the coordinates from `product.grids[grid_id]["coordinates"]["diameter"]`.
A repeated AIM header or preamble starts a new segment. When widths differ,
the combined frame exposes a List column and the stored segments retain
their native widths. `MobilityNumberSpectrum.integrated_number()` returns
lazy totals using the reported channels per decade. A missing channel yields
a null total. Vendor warnings remain available for a separate quality decision.

Use `product.flag_summary().collect()` to count each vendor condition and
inspect its first and last occurrence. For a stated analyst exclusion,
`product.quality_summary("total_number", exclude_flags=["Incomplete Scan"])`
returns a lazy sample mean with its denominator, the excluded time range and
the applied flags. Empty selections yield zero contributors and a null mean.

A custom `arrays` field declares an ordered list of named axes with numeric
`values` and `unit`, and one input column per element in row-major order.
For example, two axes of lengths 2 and 3 with six selectors produce a 2 × 3
array per timestamp. Missing readings remain null at their channel positions.
Access units and the full shape definition through `product.manifest["profile"]`.

The worked example includes both source excerpts, the recovered observations
and the plotted spectrum. [Worked imports](../../studies/examples/measurement_import.py)
and [streaming qualification](../../studies/validation/workflows-and-releases/measurement-import.md).
