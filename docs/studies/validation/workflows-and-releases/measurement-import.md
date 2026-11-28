# Importing 108 million counter readings under a 1 GiB limit

The particle-counter reader streamed a 3,024,000,000-byte CSV file into Parquet.
A subsequent Polars query reopened the dataset and wrote 1,800,000 UTC minute
means. Every minute contained 60 observations and had a mean of 129.5 counts/s.
The import and the query shared a 1 GiB Linux memory limit with swap disabled.

## Reproduce the disk-sized run

Install Particula with the `data` extra and run these commands from the source
checkout. Choose a fresh directory on a disk-backed filesystem with at least
4 GB available. The generator writes one reading per second, beginning at Unix
time 1700000040. The count rate cycles from 100 to 159 counts/s each minute.
Generate the input before starting the timed process group.

```bash
python scripts/qualify_measurement_import.py generate /path/to/run
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 \
    env POLARS_MAX_THREADS=2 "$(command -v python)" \
    scripts/qualify_measurement_import.py qualify /path/to/run
```

The script checks the group's actual limits and validates every output minute.
It sets the time column's sorted flag for this monotonically generated source.
When adapting the query to field measurements, establish time order first.
The published example demonstrates how to inspect a repaired row and a flagged
mobility spectrum. [Worked instrument imports](../../examples/measurement_import.py)

| Measured on Linux x86-64, Python 3.12.12, Polars 2.2.0 | Result |
| --- | ---: |
| Initial available disk space | 263,094,054,912 bytes |
| Maximum input chunk / output budget / scans per batch | 4 MiB / 16 MiB / 100,000 |
| Imported scans | 108,000,000 |
| Raw import and durable Parquet publication | 48.95 s |
| Lazy minute aggregation, disk sink and verification | 1.13 s |
| Whole-group peak memory, including charged file cache | 1,073,741,824 bytes |
| Peak sampled disk allocation, including input and outputs | 3,743,526,912 bytes |
| Incomplete minutes / incorrect means | 0 / 0 |

Disk allocation was sampled every 0.25 s and checked at completion.
The source SHA-256 was
`1b8f47f03d5f94ac80891c07bc1fa03f302e3e01e476f226d9c8f22a4bb97550`.
These timings cover the regular five-column logger workload on two Polars
threads. The reader uses a CSV decoder for irregular field records. Wider
spectra and unsorted campaign aggregations have different resource needs.

## Compare with the original field exports

The [pinned beta checkout](https://github.com/Gorkowski/particula-dev-data/tree/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data)
contains the following source files. To replay the checksum-pinned comparison
from the Particula checkout, run:

```bash
git clone https://github.com/Gorkowski/particula-dev-data .artifacts/particula-dev-data
git -C .artifacts/particula-dev-data checkout 4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8
PARTICULA_BETA_FIXTURES=.artifacts/particula-dev-data/particula_beta/data/tests/example_data \
    python -m pytest particula/data/tests/field_export_test.py
```

The
`cpc-3010` and `smps-aim` profiles declare the columns, units and clock.

| Export | Reopened readings | Independent check |
| --- | ---: | --- |
| CPC, 9 July 2022 | 33,267 | All 13 split pairs recovered. First span 1–2 yields 32,660 counts/s and 290.25 K. |
| CPC, 10 July 2022 | 35,297 | No repairs. Source timestamps and both channels agree with the beta formatter. |
| AIM, 7 July 2022 | 2,854 × 203 | Maximum relative difference between integrated spectra and printed totals is 4.97 × 10⁻⁶. All scans retain the detector flow and incomplete-scan warnings. |
| AIM, 10 July 2022 | 1,015 × 203 | All source spectrum values match after conversion from cm⁻³ to m⁻³. |
| AIM chamber, 25 September 2023 | 336 × 216 | The same profile binds a distinct diameter grid and preserves all 216 channels. |

For July 9, independently join a comma-terminated partial row and its numeric
continuation to recover five fields. Compare the original line numbers, epoch
seconds, count rate and logger temperature with the normalized scans. Imports
using 1-, 7-, 64- and 1,024-byte input chunks agreed with reopened Parquet.
For the AIM check, sum each source row's numeric midpoint columns and divide
by the reported 128 channels per decade. The first integral was
2,168.998164 cm⁻³ against the printed 2,169.00 cm⁻³. Vendor agreement checks
the interpretation of the file. The two warning tokens remain relevant to any
use of those values as calibrated measurements.

The beta scalar and sizer settings were also translated and compared with
the pinned beta formatter for each export. For the split logger, compare its
surviving complete rows. The beta's character filter drops the fragments.
For the spectra, compare channel order and inclusive bounds before converting
units. The packaged tests use small excerpts and synthetic failure cases so
that ordinary test runs work offline.

## Compare an hour on a minute grid

On 10 July 2022, the particle counter reported observations in every minute
from 12:00 to 13:00 UTC. The mobility sizer reported scan starts in 40 of those
60 minutes. Those 40 windows support a comparison of minute summaries even
when individual readings occurred at different times. The remaining 20 retain
the counter mean and a null sizer concentration.

| Source quantity | Observations in the hour | Valid readings per minute | Mean at 12:00 UTC |
| --- | ---: | ---: | ---: |
| Counter detector rate | 1,203 | 20–21 | 17,260.6 counts/s |
| Sizer vendor number concentration | 40 | 0–1 | 1.38240 × 10¹⁰ m⁻³ |

The comparison uses the 10 July files in the [pinned checkout above](#compare-with-the-original-field-exports).
The reference decoder reads the counter's Unix timestamp and the sizer's
printed date and scan start with Python's CSV reader. It interprets the
sizer clock as UTC per the donor settings, converts the printed concentration
from cm⁻³ to m⁻³, and assigns readings to left-closed one-minute windows.
NumPy independently calculates each denominator and arithmetic mean. The
test compares these arrays, including all empty windows, against aligned
Parquet members reopened through the measurement API.

All 40 sizer scans report both `Detector aerosol flow rate error` and
`Incomplete Scan`. Including these numerical records checks the placement
of scan starts. Excluding the incomplete-scan flag leaves 60 null sizer means
and preserves the 1,203 counter samples. Actual exposure within those
windows remains unknown.

Use the same donor checkout and `PARTICULA_BETA_FIXTURES` directory as in the
single-instrument comparison above. This command imports into temporary
datasets, reopens them lazily and checks the entire hour. To reuse archived
datasets produced with the earlier importer, set `PARTICULA_CAMPAIGN_DATASETS`
to a directory containing the two completed datasets named `counter` and
`sizer`. The test verifies the source digest of each.

```bash
PARTICULA_BETA_FIXTURES=.artifacts/particula-dev-data/particula_beta/data/tests/example_data \
    python -m pytest particula/data/tests/campaign_field_test.py
```

[Build a collection and choose weighting and exclusions](../../../guide/data-model/measurement-collections.md).
