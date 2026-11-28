"""Qualify a disk-backed particle-counter import under cgroup v2."""

import argparse
import json
import shutil
import time
from pathlib import Path
from threading import Event, Thread

import polars as pl
from particula.data import import_to_parquet

START = 1_700_000_040
MINUTES = 1_800_000


def generate(source, minutes):
    """Generate the exact sixty-sample rate cycle outside the timed import."""
    with source.open("x") as output:
        for minute in range(minutes):
            start = START + 60 * minute
            output.writelines(
                f"{start + i},{100 + i},17.0,1,0.017\n" for i in range(60)
            )


def _allocated(root):
    total = 0
    for path in root.rglob("*"):
        try:
            total += path.stat().st_blocks * 512
        except FileNotFoundError:
            # The importer atomically moves the staging directory.
            continue
    return total


def qualify(root, minutes):
    """Verify all minute summaries on disk and print measured resource use."""
    group = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split("::")[1].lstrip("/")
    memory_limit = (group / "memory.max").read_text().strip()
    swap = (group / "memory.swap.max").read_text().strip()
    if memory_limit != str(2**30) or swap != "0":
        raise RuntimeError(
            "Qualification requires MemoryMax=1G and MemorySwapMax=0"
        )
    source = root / "campaign-cpc.csv"
    if source.stat().st_size <= 2**30:
        raise ValueError("Input must exceed the memory limit")
    free = shutil.disk_usage(root).free
    if (root / "normalized").exists() or (
        root / "minute-means.parquet"
    ).exists():
        raise ValueError("Use a fresh output directory for qualification")
    peak_disk = [_allocated(root)]
    done = Event()

    def sample_disk():
        while not done.wait(0.25):
            peak_disk[0] = max(peak_disk[0], _allocated(root))

    meter = Thread(target=sample_disk, daemon=True)
    meter.start()
    started = time.monotonic()
    dataset = import_to_parquet(
        source,
        root / "normalized",
        profile="cpc-3010",
        input_chunk_bytes=2**22,
        batch_scans=100000,
        batch_bytes=2**24,
    )
    imported = time.monotonic()
    # The generated UTC times are monotonically increasing. Dynamic windows
    # can stream closed minutes to disk without holding every group at once.
    minute = (
        dataset.lazy_frame.select("time_utc", "count_rate")
        .set_sorted("time_utc")
        .group_by_dynamic("time_utc", every="1m")
        .agg(
            pl.len().alias("samples"),
            pl.col("count_rate").mean().alias("mean_rate"),
        )
    )
    target = root / "minute-means.parquet"
    minute.sink_parquet(target, engine="streaming")
    summary = (
        pl.scan_parquet(target)
        .select(
            pl.len(),
            (pl.col("samples") != 60).sum().alias("incomplete"),
            (pl.col("mean_rate") != 129.5).sum().alias("wrong_means"),
        )
        .collect(engine="streaming")
        .row(0)
    )
    if (
        summary != (minutes, 0, 0)
        or dataset.manifest["scan_count"] != 60 * minutes
    ):
        raise ValueError(f"Incorrect import or minute summaries: {summary}")
    done.set()
    meter.join()
    print(
        json.dumps(
            {
                "polars": pl.__version__,
                "polars_threads": pl.thread_pool_size(),
                "input_bytes": source.stat().st_size,
                "initial_available_disk": free,
                "input_sha256": dataset.manifest["source_sha256"],
                "scan_count": dataset.manifest["scan_count"],
                "minute_summary": summary,
                "cgroup_memory_max_bytes": int(memory_limit),
                "swap_max_bytes": int(swap),
                "cgroup_memory_peak_bytes": int(
                    (group / "memory.peak").read_text()
                ),
                "peak_sampled_disk_bytes": max(peak_disk[0], _allocated(root)),
                "disk_bytes": sum(
                    p.stat().st_blocks * 512
                    for p in root.rglob("*")
                    if p.is_file()
                ),
                "import_seconds": round(imported - started, 2),
                "analysis_seconds": round(time.monotonic() - imported, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "qualify"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True, parents=True)
    if args.mode == "generate":
        generate(args.root / "campaign-cpc.csv", MINUTES)
    else:
        qualify(args.root, MINUTES)
