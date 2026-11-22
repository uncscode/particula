"""Size-specific charged inventory immediately before the three early field switches."""

import numpy as np
from functools import lru_cache
from balance import write
from replay import ZipFile, ARCHIVE, measurements, curves


@lru_cache(None)
def charges():
    t, d, e, spec, *_ = measurements()
    with ZipFile(ARCHIVE) as z:
        a = curves(z, "17a")[-2]
        switches = a[1:, 0][np.diff(a[:, 1]) > 0.5]

        def read(name):
            raw = z.read("archive/fig" + name + ".dat").decode()
            data = [
                np.fromstring(l, sep=" ")
                for l in raw.splitlines()
                if l.strip() and not l.startswith("%")
            ]
            return (
                data[0],
                10 ** data[1],
                np.where(np.array(data[2:]) > 0, 10 ** np.array(data[2:]), 0.0),
            )

        time, diam, neg = read("03a")
        tp, dp, pos = read("03d")
        assert np.allclose(tp, time) and np.allclose(dp, diam)
    rows = []
    events = []
    for flip in switches:
        previous = np.flatnonzero((time < flip) & (time >= flip - 5 / 60))
        # Make the finite pre-switch averaging window explicit for the two instruments.
        sm = np.flatnonzero((t < flip) & (t >= flip - 5 / 60))
        den = np.median(spec[:, sm], axis=1)
        charge = np.interp(
            np.log(d),
            np.log(diam),
            np.median((neg + pos)[:, previous], axis=1),
            left=0,
            right=0,
        )
        frac = np.clip(charge / np.maximum(den, 1.0), 0, 1)
        frac[(d < diam.min()) | (d > diam.max())] = 0
        h = flip - t[0]
        events.append((h, d, frac))
        for lo, hi in [(5.0, 9.7), (9.7, 15.0), (15.0, 40.0)]:
            w = np.maximum(
                0,
                np.minimum(e[1:], np.log10(hi))
                - np.maximum(e[:-1], np.log10(lo)),
            )
            rows.append(
                dict(
                    hour=h,
                    lo_nm=lo,
                    hi_nm=hi,
                    pre_total=w @ den,
                    pre_AIS_charged=w @ charge,
                    modeled_fraction=(w @ (frac * den)) / (w @ den)
                    if w @ den
                    else 0,
                )
            )
    write("charge_switches.csv", rows)
    return events
