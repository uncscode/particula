"""A camera-conditioned spatial redistribution, conserving each diameter-class inventory."""

import numpy as np
import json, time
from observations import load, D
from inverse import HERE, Forward, torch, project, train_mask


def run_case(run, gain, mesh):
    start = time.monotonic()
    name = f"{run}-{mesh}-redistribute-{gain:g}"
    folder = HERE / "state-check" / name
    folder.mkdir(parents=True, exist_ok=True)
    fit = HERE / "final-fits" / f"{run}-18-1-0.05-joint-0.5-hold-native-0.05"
    z = np.load(fit / "prediction.npz")
    m = Forward(run, mesh, 1, 0.05, "cuda:0")
    put = m.put
    t, y = load(run)
    h = m.opt / m.f
    norm = (h - h.min(0).values) / (h.max(0).values - h.min(0).values).clamp(min=1e-15)
    visible = (m.model.d >= 11) & (m.model.d < 79)
    log = []

    def update(i, pop, rows):
        if (i + 1) % 5:
            return pop
        left = 2.75 + (i + 1) * 0.05 - 0.25
        if not train_mask(np.array([left + 0.125]), "hold")[0]:
            return pop
        sel = (t >= left) & (t < left + 0.25)
        obs = put((y[sel].mean(0) @ project(D, 1, np.arange(len(m.model.d) + 1))))
        local = (pop * h).sum(0)
        resid = obs - torch.stack(rows[-5:]).mean(0)
        target = local + gain * resid
        total = pop.sum(0)

        def relocate(alpha):
            weighted = pop * torch.exp(norm * alpha)
            return weighted * (total / weighted.sum(0).clamp(min=1e-150))

        low = torch.full_like(total, -np.log(4.0))
        high = -low
        for _ in range(25):
            mid = (low + high) / 2
            below = (relocate(mid) * h).sum(0) < target
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        alpha = torch.where(put(visible), (low + high) / 2, 0.0)
        revised = relocate(alpha)
        error = (revised.sum(0) - total).abs().max().item()
        assert error < 1e-11
        log.append(
            dict(
                end_s=left + 0.25,
                fraction_relocated=((revised - pop).abs().sum() / (2 * total.sum())).item(),
                max_number_error=error,
            )
        )
        return revised

    with torch.no_grad():
        pred, bud = m.predict(
            put(z["logs"]),
            put(z["flow"]),
            put(z["thermal"]),
            True,
            analysis_update=update if gain else None,
        )
        assert bud[:, 6].max() < 1e-15 and bud[:, 4].min() > 0
        np.savez_compressed(
            folder / "prediction.npz",
            t=m.t,
            d=m.model.d,
            spec=pred.cpu().numpy(),
            ledger=bud.cpu().numpy(),
        )
    meta = dict(
        run=run,
        kind="redistribute",
        knot=0.5,
        split="hold",
        objective="native",
        cadence=0.05,
        mesh=mesh,
        width=1,
        dt=0.05,
        gain=gain,
        source=fit.name,
        updates=log,
        seconds=time.monotonic() - start,
        device=torch.cuda.get_device_name(),
    )
    (folder / "fit.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    for r in ("forced", "unforced"):
        for mesh in (18, 108, 576):
            for g in (0, 0.25, 0.5, 1):
                run_case(r, g, mesh)
