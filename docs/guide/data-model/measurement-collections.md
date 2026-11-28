# Compare instruments on a common time grid

A counter may report a rate every few seconds while a mobility sizer reports
one spectrum per scan. To compare their time series, choose the time interval
and compute each instrument's statistic separately. A measurement collection
keeps named products together while preserving their clocks, units, coordinate
grids, and source diagnostics. Install the `data` extra and
[import the instrument files](measurements.md) before starting.

## Assign observations to minutes

Open two imported datasets and identify them by their roles in this campaign.
Choose distinct member names without periods, such as `inlet` and `outlet`
for two physical counters with the same profile. Supply timezone-aware bounds
at microsecond precision. An ISO bound with a nonzero sub-microsecond part
raises an error. Imported scan timestamps retain their nanosecond precision.
Here the counter uses Unix seconds and
the mobility export uses the UTC interpretation specified in the donor's
settings. An export that only prints a date and time needs a site clock
interpretation before comparison.

```python
from datetime import UTC, datetime, timedelta

from particula.data import MeasurementCollection, open_measurement

start = datetime(2022, 7, 10, 12, tzinfo=UTC)
stop = start + timedelta(hours=1)
members = {
    "inlet": open_measurement("counter-dataset"),
    "sizer": open_measurement("mobility-dataset"),
}
campaign = MeasurementCollection(members)
hour = campaign.select_time(start, stop)
fields = {"inlet": ["count_rate"], "sizer": ["total_number"]}
minute = hour.average(
    every="1m", origin=start, weighting="sample", fields=fields
)
comparison = minute.align_windows(how="grid")
```

Each sample receives equal weight within its instrument and window. A
reading at exactly 12:01 enters the minute beginning at 12:01. The hour ends
before the 13:00 observation. The lazy comparison has one row per requested
minute. `inlet.count_rate` is a mean in counts/s, and `sizer.total_number` is
a mean in m⁻³ for the supplied profiles. Their respective `.n` columns count
valid contributing observations. A window with zero contributors retains a
null mean. A field declared as actual detector counts sums the pulses in the
window. A rate field averages the reported rates. For a spectrum on one grid,
each channel has its own mean and count of valid observations. Pooling
spectra from different diameter grids requires an explicit regridding method.

Use `comparison.collect()` to inspect this hour or
`comparison.sink_parquet("minute-comparison.parquet")` to write the table.
The comparison's `minute.manifest` records the parent product IDs, field
units, origin, selection bounds and quality policy. Keep this metadata with
an exported table when another analyst will interpret its units or exclusions.

In the [July 10 campaign check](../../studies/validation/workflows-and-releases/measurement-import.md#compare-an-hour-on-a-minute-grid),
the counter contributed to all 60 minutes and the sizer to 40. All 40 sizer
scans carried an incomplete-scan warning. A populated minute means that an
observation started within the minute. The sample means alone cannot tell
how much of the minute either inlet sampled air.

For a comparison of individual readings, use an explicit tolerance such as
`campaign.match_nearest("inlet", "sizer", tolerance="2s")`. Inspect both
qualified `time_utc` columns and `time_offset_ns` (right minus left).
A missing right-hand value can mean the nearest scan fell outside the
tolerance. Choose a tolerance justified by the clock and sampling protocols.
Duplicate right-hand timestamps make the match ambiguous and raise an error.

## Use verified acquisition intervals for exposure weighting

Use `weighting="exposure"` for interval means when the timestamp marks
the start of acquisition and the measured duration is available in seconds.
Pass a mapping such as `exposure={"inlet": "live_seconds"}`, where the named
column records independently verified sample exposure. Each scan contributes
its valid overlap with the window. For example, a 75 s acquisition that starts
at 12:00:45 contributes 15 s before 12:01 and 60 s after it. In those
minutes, `.exposure_s` is 15 and 60 seconds and `.coverage` is 0.25 and 1.0
if this was the only acquisition. Duration and overlap checks apply after
flag and repair exclusions, to scans starting in the selection and earlier
scans with verified exposure extending into it. Overlapping retained exposures
on the same member are ambiguous and raise an error. Recorded pulse totals lack
within-interval timing, so exposure weighting rejects those fields. Retrace
time lies outside sample exposure. An incomplete mobility scan's nominal scan
duration cannot establish its actual exposure or an individual channel's dwell
time.

## Keep field warnings and campaign events distinct

To exclude a vendor condition, give the member name and the exact flag in
`exclude_flags`. For example, adding
`exclude_flags={"sizer": ["Incomplete Scan"]}` to the July calculation leaves
the sizer with zero contributors throughout the hour. The inlet counter
retains all of its samples. Use `exclude_repaired=True` when the analysis
should omit reconstructed logger records.

For a documented change in sampling conditions, record a time-bounded
campaign event. This example represents a hypothetical five-minute zero
check on the inlet counter. Exclude that event in analyses of ambient air.

```python
from particula.data import CampaignEvent

zero_check = CampaignEvent(
    kind="zero_check", start=start, stop=start + timedelta(minutes=5),
    member="inlet", field="count_rate",
)
annotated = MeasurementCollection(members, events=[zero_check])
ambient = annotated.select_time(start, stop).average(
    every="1m", origin=start, weighting="sample", fields=fields,
    exclude_events=("zero_check",),
).align_windows(how="grid")
```

Compare `ambient` and `comparison` to see the effect on the first five
`inlet.count_rate.n` values. The scope identifies the affected member and,
when specified, its field. With exposure weighting the excluded event
splits the valid interval at the event boundaries. Other channels and deployments keep their own
observations. Scan identifiers are local to the source. Use
`(source_sha256, scan_id)` when tracing a reading across products.

## Move and reopen the campaign

Save the collection and move the complete directory, including its member
datasets. Saving a time-selected collection records the selection bounds and
includes the underlying member datasets. Opening the directory restores the
typed products and their diagnostics. Check a member's manifest to identify the original source.

```python
annotated.save("july-campaign")
restored = MeasurementCollection.open("july-campaign")
assert (
    restored["sizer"].manifest["source_sha256"]
    == members["sizer"].manifest["source_sha256"]
)
```

Alignment puts readings on a shared clock. Converting count rate into a
number concentration also requires a compatible sample flow and its
calibration. [Process calibrated measurements](measurement-processing.md).

When a reference pulse establishes a clock error, apply `correct_clock` before
selecting or matching observations. Its signed `offset_seconds` is added to
the reported UTC time within the half-open `start`, `stop` interval. The
returned product retains `reported_time_utc`, the reference `source`, the
offset and the original scan IDs. A measured inlet transport delay describes
the air path and requires its own interpretation.

Keep the returned window object to save the analysis policy with the values.
`minute["inlet"]` selects one member’s summaries. `minute.save("july-minutes")`
writes the lazy comparison and its manifest, and
`WindowComparison.open("july-minutes")` restores the units, grids, UTC bounds,
QC choices and parent product identities. For a sample mean, supplying a
verified exposure field also reports valid seconds and coverage independently
of the equal sample weights.
