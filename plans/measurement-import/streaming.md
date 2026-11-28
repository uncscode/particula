# Streaming raw measurements to Parquet

Part of the [measurement and instruments plan](../measurement-data-and-instruments.md). This is the proposed import contract. The CPC reference check below has been run. The larger-than-RAM qualification is a delivery criterion for the importer.

## Raw input and output batches

Open each raw file as a stream. Read its header once and decode bounded chunks, carrying a partial UTF-8 character, physical line, or quoted CSV record across byte boundaries where applicable. After decoding physical records, apply the instrument profile. For the CPC split-line repair, retain the incomplete fragment and its original line number across input batches. Join only when the profile identifies a unique valid continuation. Report ambiguous or unfinished fragments, including one at end of file. Reset this state at each new source file. Bound individual record size and report oversized records with their source span.

Emit complete observations with stable scan IDs, source spans, units, grid references and quality flags. Buffer output independently of input chunk size. Use a byte budget as well as a row limit so wide spectra cannot silently exceed the batch target. Flush each batch into a Parquet row group or part and release it. Stream repair diagnostics to disk too. Carry only counters and bounded parser state between batches. Compute the source digest incrementally. Mark a dataset complete only after all parts, grid metadata, diagnostics and the final manifest are durable. A retry must expose each successful observation exactly once. Downstream queries should open completed parts with Polars lazy scans.

## Split CPC records across batch boundaries

Use the pinned July 9 CPC logger export at beta revision `4e56f7da`. Its 33,280 physical lines contain 33,254 complete records and 13 uniquely reconstructable pairs. A reference parser that carried one pending fragment produced the same 33,267 records, original field tokens, and source spans at 1, 2, 7, 16, 64 and 1,024 physical lines per batch as a single-batch read. One-line batches split all 13 repairs. The first repaired record covers lines 1–2. Adjacent repairs cover 111–112 and 113–114.

For the implemented importer, repeat that check through the public import API and read back its Parquet output. Compare all scan IDs, timestamps, raw or converted channel values, line spans, repair flags and rejection reasons with independently expected results. Vary input chunk size and output flush size independently. Include a raw byte boundary inside a timestamp, an ambiguous continuation, two consecutive split records, a trailing fragment and a new file following that fragment. Run the profile with repair disabled as a separate control. Logical observations and diagnostic counts should agree across batch sizes even when the Parquet file layout differs. [Original fixture and settings](https://github.com/Gorkowski/particula-dev-data/tree/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/CPC_3010_data)

## Worked larger-than-RAM example to publish with the importer

Generate a synthetic CPC logger file on a disk-backed volume outside the timed process, with Unix epoch seconds, count rate, temperature and two logger fields. Repeat a sixty-second rate pattern with values 100 through 159 counts/s. The following generator writes about 3 GB incrementally and supplies a known result for every complete UTC minute.

```python
start = 1_700_000_040  # Divisible by 60.
minutes = 1_800_000
with open("campaign-cpc.csv", "w", buffering=1 << 20) as output:
    for sample in range(60 * minutes):
        output.write(f"{start + sample},{100 + sample % 60},17.0,1,0.017\n")
```

Run the proposed raw-to-Parquet importer and subsequent query under an enforced **1 GiB memory limit with swap disabled**. Record input bytes, available disk space, input and output batch settings, peak memory of the whole process group, and peak disk use. Confirm input bytes exceed the memory limit. Store the result on disk and inspect only compact summaries. Use the importer’s final documented field names when publishing the example.

```python
import polars as pl

minute = (
    pl.scan_parquet("normalized/cpc/part-*.parquet")
    .select("time_utc", "count_rate")
    .group_by(pl.col("time_utc").dt.truncate("1m").alias("minute"))
    .agg(pl.len().alias("samples"), pl.col("count_rate").mean().alias("mean_rate"))
)
minute.sink_parquet("minute-means.parquet", engine="streaming")

summary = pl.scan_parquet("minute-means.parquet").select(
    pl.len().alias("minutes"),
    (pl.col("samples") != 60).sum().alias("incomplete"),
    (pl.col("mean_rate") != 129.5).sum().alias("wrong_means"),
).collect(engine="streaming")
assert summary.row(0) == (1_800_000, 0, 0)
```

Acceptance requires 108 million imported observations, the exact summary above, and both the raw import and query finishing under the cap. Also run a small prefix against independently generated expected records in CI. Reserve the disk-sized run for qualification. If group state exceeds the cap, partition by UTC date, aggregate within each date, and verify the same oracle and memory bound. Publish the settings and measurements for the path that actually passed. [Polars streaming behavior](https://docs.pola.rs/user-guide/concepts/streaming/)
