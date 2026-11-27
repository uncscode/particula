"""Weighted particle set models and trajectory diagnostics."""

import json
from pathlib import Path

import numpy as np
import torch
from measures import observables
from torch import nn


def state_input(physical):
    """Bound concentrations for network input while retaining the fp64
    record.
    """
    scales = physical.new_tensor([1e11, 1e-4, 1e-4, 1e-4, 0.01])
    return torch.log1p(physical / scales)


def local_response_targets(data, p):
    """Predict change by the next observation and the end of the current leg."""
    time = data["time"]
    mode = str(data["mode"]) if "mode" in data else "pulse"
    marks = (
        np.flatnonzero(np.diff(time) == 0)
        if mode != "coagulation"
        else np.array([], dtype=int)
    )
    end_times = np.r_[marks, len(time) - 1]
    t = np.arange(len(time))
    end = end_times[np.searchsorted(end_times, t)]
    nxt = np.minimum(t + 1, end)
    water = p[..., 1]
    target = np.stack([water[:, nxt] - water, water[:, end] - water], -1)
    schedule = np.stack(
        [
            (time[nxt] - time[t]) * 100,
            (time[end] - time[t]) * 10,
            np.full(len(t), mode != "coagulation"),
            np.full(len(t), mode in ("coagulation", "coupled")),
        ],
        -1,
    )
    return target, np.broadcast_to(schedule, (*p.shape[:2], 4))


def load_split(path, device, pattern="families-*.npz", grid="pulse"):
    """Load features, physical records, and future labels for one split."""
    records = []
    for file in sorted(Path(path).glob(pattern)):
        with np.load(file) as data:
            mass = data["mass"].swapaxes(0, 1)
            counts = data["counts"]
            counts = (
                np.broadcast_to(counts[:, None, :], mass.shape[:-1])
                if counts.ndim == 2
                else counts.swapaxes(0, 1)
            )
            x, p, d = observables(
                mass,
                counts,
                data["density"],
                float(data["volume"]),
                data["gas"].T,
                grid,
            )
            if grid == "tails":
                future, forcing = local_response_targets(data, p)
            else:
                specs = json.loads(str(data["families"]))
                boundary = mass.shape[1] // 2
                samples = boundary - 1
                known = np.repeat(
                    [[s["wet_rh"], s["dry_rh"]] for s in specs], 2, axis=0
                )
                future = np.empty((*mass.shape[:2], 2))
                # All future labels depend on the current state and known
                # forcing
                # only.
                for t in range(mass.shape[1]):
                    end = samples if t < boundary else 2 * boundary - 1
                    future[:, t, 0] = p[:, end, 1] - p[:, t, 1]
                    future[:, t, 1] = p[:, -1, 1] - p[:, t, 1]
                phase = np.arange(mass.shape[1]) >= boundary
                remain = np.tile(np.linspace(1, 0, boundary), 2)
                forcing = np.concatenate(
                    [
                        np.broadcast_to(known[:, None], (*mass.shape[:2], 2)),
                        np.broadcast_to(
                            np.stack([phase, remain], -1), (*mass.shape[:2], 2)
                        ),
                    ],
                    axis=-1,
                )
            records.append(
                dict(
                    x=x,
                    counts=counts,
                    physical=p,
                    diag=d,
                    forcing=forcing,
                    future=future,
                    ids=np.repeat(data["ids"], 2),
                )
            )
    if not records:
        raise ValueError(f"No trajectories in {path}")
    return {
        k: torch.as_tensor(
            np.concatenate([r[k] for r in records]),
            device=device,
            dtype=torch.float64
            if k == "physical"
            else torch.long
            if k == "ids"
            else torch.float32,
        )
        for k in records[0]
    }


def mlp(widths):
    """Build a feed-forward network with SiLU hidden activations."""
    layers = []
    for a, b in zip(widths[:-1], widths[1:], strict=True):
        if layers:
            layers.append(nn.SiLU())
        layers.append(nn.Linear(a, b))
    return nn.Sequential(*layers)


class WeightedEncoder(nn.Module):
    """Learn one feature map shared by all active particle slots."""

    def __init__(self, width, diag_dim=28):
        """Construct the networks and their persistent scaling buffers."""
        super().__init__()
        self.width = width
        self.phi = mlp([4, 64, 64, width])
        self.decoder = mlp([width + 5, 96, 96, diag_dim])
        self.response = mlp([width + 9, 96, 96, 2])
        for key, dim in (
            ("x", 4),
            ("diag", diag_dim),
            ("future", 2),
            ("physical", 5),
        ):
            self.register_buffer(key + "_mean", torch.zeros(dim))
            self.register_buffer(key + "_std", torch.ones(dim))

    @torch.no_grad()
    def calibrate(self, data):
        """Estimate all scaling statistics using the training split."""
        valid = data["counts"] > 0
        for key in ("x", "diag", "future", "physical"):
            if key == "x":
                values = data[key][valid]
            elif key == "physical":
                values = state_input(data[key]).float().flatten(0, 1)
            else:
                values = data[key].flatten(0, 1)
            getattr(self, key + "_mean").copy_(values.mean(0))
            getattr(self, key + "_std").copy_(values.std(0).clamp_min(1e-10))

    def map_particles(self, x):
        """Evaluate the shared particle map before population reduction."""
        return self.phi((x - self.x_mean) / self.x_std)

    def encode(self, x, counts):
        """Return count-weighted shape coordinates for an unordered
        population.
        """
        active = counts > 0
        x = torch.where(active[..., None], x, self.x_mean)
        values = self.map_particles(x)
        count = counts.sum(-1, keepdim=True)
        weight = counts / torch.where(count > 0, count, torch.ones_like(count))
        return (weight[..., None] * values).sum(-2)

    def state(self, z, physical):
        """Join shape and scaled exact concentrations for the readout."""
        return torch.cat(
            [
                z,
                (state_input(physical).float() - self.physical_mean)
                / self.physical_std,
            ],
            -1,
        )

    def diagnose(self, z, physical):
        """Decode the native distribution diagnostics in physical units."""
        return (
            self.decoder(self.state(z, physical)) * self.diag_std
            + self.diag_mean
        )


class MomentEncoder(WeightedEncoder):
    """Reserve four coordinates for explicit mixing-state moments."""

    def __init__(self, width, diag_dim=28):
        """Keep total stored width equal to the fully learned representation."""
        if width <= 4:
            raise ValueError(
                "The moment encoder needs at least five coordinates"
            )
        super().__init__(width, diag_dim)
        self.phi = mlp([4, 64, 64, width - 4])

    def map_particles(self, x):
        """Combine learned coordinates with compositional moments."""
        dry_fraction = x[..., 2]
        wet_fraction = x[..., 3]
        moments = torch.stack(
            [
                dry_fraction,
                dry_fraction.square(),
                x[..., 0] * dry_fraction.square(),
                wet_fraction * dry_fraction,
            ],
            -1,
        )
        return torch.cat([moments, super().map_particles(x)], -1)


def errors_by_pair(pred, true):
    """Paired signed water-transfer errors on distinct validation families."""
    target = true[1::2] - true[::2]
    delta = pred[1::2] - pred[::2]
    return dict(
        pair_mae_mg_m3=float((delta - target).abs().mean() * 1e6),
        pair_sign_fraction=float(((delta * target) > 0).float().mean()),
        pair_abs_error_over_mean_abs_signal=float(
            (delta - target).abs().mean() / target.abs().mean()
        ),
        member_mae_mg_m3=float((pred - true).abs().mean() * 1e6),
    )
