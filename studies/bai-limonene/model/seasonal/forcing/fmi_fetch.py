"""Fetch the measured daily minute records for prescribed 2022 monthly windows."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import date
import argparse
import gzip
import urllib.request
import urllib.parse
import tempfile
import time

BASE = "https://opendata.fmi.fi/wfs"
SITE = 101932
PARAMS = "GLOB_1MIN,DIR_1MIN,DIFF_1MIN,REFL_1MIN,UVB_U"


def url(day):
    fields = dict(
        service="WFS",
        version="2.0.0",
        request="GetFeature",
        storedquery_id="fmi::observations::radiation::multipointcoverage",
        fmisid=SITE,
        parameters=PARAMS,
        starttime=f"{day.isoformat()}T00:00:00Z",
        endtime=f"{day.isoformat()}T23:59:00Z",
    )
    return BASE + "?" + urllib.parse.urlencode(fields)


def retrieve(item):
    folder, day = item
    out = folder / f"{day.isoformat()}.xml.gz"
    if out.exists():
        return
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url(day), timeout=40) as response:
                raw = response.read()
            if b"doubleOrNilReasonTupleList" not in raw:
                raise ValueError(raw[:600])
            with tempfile.TemporaryDirectory(dir=folder) as tmp:
                pending = Path(tmp) / out.name
                with gzip.open(pending, "wb") as save:
                    save.write(raw)
                pending.replace(out)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("cache", type=Path)
    args.add_argument("--year", type=int, default=2022)
    args.add_argument("--days", type=int, nargs=2, default=(9, 18))
    ns = args.parse_args()
    ns.cache.mkdir(parents=True, exist_ok=True)
    ds = [
        date(ns.year, m, d)
        for d in range(ns.days[0], ns.days[1] + 1)
        for m in range(1, 13)
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, _ in enumerate(pool.map(retrieve, [(ns.cache, d) for d in ds])):
            if i % 12 == 11:
                print("Retrieved", i + 1, "of", len(ds), flush=True)


if __name__ == "__main__":
    main()
