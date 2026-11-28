"""Compare two virtual cameras viewing a single evolving water-droplet chamber."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from workspace import WORK

HERE = WORK / "cameras"
from chamber import HEIGHT0, MESHES, run_case, torch
from mesh import hologram

MESHES.update({"1536": (8, 24, 8), "2304": (8, 36, 8)})
VIEWS = {
    "A": (0, 0),
    "below55": (-0.055, 0),
    "below25": (-0.025, 0),
    "above25": (0.025, 0),
    "above55": (0.055, 0),
    "left40": (0, -0.04),
    "right40": (0, 0.04),
}
MECHANISMS = ("control", "import_above", "import_below", "local_growth", "local_mergers")


class Intervention:
    def __init__(self, mechanism, strength=None, start=4.25, end=5.0):
        self.mechanism = mechanism
        default = {"local_growth": 4.0, "local_mergers": 15000.0}.get(mechanism, 0.25)
        self.strength = default if strength is None else strength
        self.start, self.end = start, end
        self.spectra = []
        self.band = []
        self.moved = 0.0
        self.error = 0.0

    def setup(self, grid, model, times):
        self.model = model
        self.diam = model.d
        self.times = times
        z, _, radius = np.meshgrid(grid.zc, grid.ac, grid.rc, indexing="ij")
        z, radius = z.ravel(), radius.ravel()
        # Both the cold pocket and enhanced encounters are confined near camera A.
        self.local = np.exp(-0.5 * ((z - HEIGHT0) / 0.035) ** 2 - 0.5 * (radius / 0.055) ** 2)
        self.local_kernel = model.tensor(self.local[:, None]) * model.kernel["midpairs"]
        source_z = HEIGHT0 + (0.10 if self.mechanism == "import_above" else -0.10)
        parcel = np.exp(-0.5 * ((z - source_z) / 0.03) ** 2 - 0.5 * (radius / 0.055) ** 2) * grid.v
        # Keep the initial excess entirely outside the original optical reconstruction.
        parcel *= np.abs(z - HEIGHT0) > 0.07
        self.parcel = model.tensor(parcel[:, None] / parcel.sum())
        self.donors = model.tensor((radius > 0.17)[:, None] * ((model.d >= 50) & (model.d < 65)))
        self.first = times[np.searchsorted(times, self.start)]
        self.active = (times >= self.start) & (times < self.end)
        self.visible = (model.d >= 10) & (model.d < 80)
        band_overlap = (
            np.maximum(
                0,
                np.minimum(model.d + model.width / 2, 65)
                - np.maximum(model.d - model.width / 2, 50),
            )
            / model.width
        )
        rows = []
        for shift in (0.0, -0.01, 0.01):
            for dz, dx in VIEWS.values():
                w, coverage = hologram(
                    grid, model.d[self.visible], z=HEIGHT0 + dz + shift, lateral=dx
                )
                assert np.allclose(w.sum(0), 1) and min(coverage) > 0.999
                rows.append(w)
        self.weights = model.tensor(np.stack(rows))
        self.tail = model.tensor(band_overlap[self.visible])

    def cooling(self, times):
        if self.mechanism == "local_growth":
            return -self.strength * self.active[:, None] * self.local
        return 0.0

    def kernel(self, t, start, count, physical):
        if self.mechanism == "local_mergers" and self.start <= t < self.end:
            return physical + (self.strength - 1) * self.local_kernel[start : start + count]
        return physical

    def relocate(self, t, pop):
        if self.mechanism.startswith("import") and t == self.first:
            take = self.strength * pop * self.donors
            new = pop - take + self.parcel * take.sum(0)
            self.moved = take.sum().item()
            self.error = (new.sum(0) - pop.sum(0)).abs().max().item()
            assert self.error < 1e-11 and new.min() >= 0
            return new
        return pop

    def observe(self, totals):
        y = (self.weights * totals[:, self.visible]).sum(1)
        self.spectra.append(y[: len(VIEWS)].cpu().numpy())
        self.band.append((y @ self.tail).cpu().numpy())


def simulate(
    mesh, flow, width, dt, mechanism, device, strength=None, start=4.25, end=5.0, destination=None
):
    probe = Intervention(mechanism, strength, start, end)
    begin = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        os.environ["OPTICAL_OUTPUT"] = scratch
        run_case("unforced", str(mesh), flow, "standard", dt, width, device, experiment=probe)
        f = next(Path(scratch).glob("*.npz"))
        with np.load(f) as saved:
            central_error = np.abs(
                saved["spectra"][:, 0, probe.visible] - np.stack(probe.spectra)[:, 0]
            ).max()
            assert central_error < 1e-6
    folder = destination or HERE / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{mesh}-{flow}-{width}-{dt}-{mechanism}"
    np.savez_compressed(
        folder / (name + ".npz"),
        t=probe.times,
        d=probe.diam[probe.visible],
        spectra=np.array(probe.spectra, dtype="f4"),
        tail=np.array(probe.band, dtype="f4"),
        views=list(VIEWS),
        registration_m=[0.0, -0.01, 0.01],
    )
    info = dict(
        mesh=mesh,
        flow=flow,
        diameter_width_um=width,
        dt=dt,
        mechanism=mechanism,
        strength=probe.strength,
        start_s=start,
        end_s=end,
        relocation_conservation_error=float(probe.error),
        moved_number_cm3=probe.moved,
        central_optical_agreement=float(central_error),
        seconds=time.monotonic() - begin,
        device=torch.cuda.get_device_name() if device.startswith("cuda") else device,
    )
    (folder / (name + ".json")).write_text(json.dumps(info, indent=2) + "\n")
    return np.stack(probe.band)[:, 0], probe.times


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mesh", type=int)
    p.add_argument("flow")
    p.add_argument("width", type=int)
    p.add_argument("dt", type=float)
    p.add_argument("mechanisms", nargs="*", default=list(MECHANISMS))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for case in args.mechanisms:
        simulate(args.mesh, args.flow, args.width, args.dt, case, args.device)
