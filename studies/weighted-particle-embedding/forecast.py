"""Conservative latent forecasts with a common forcing and history contract."""

import json

import numpy as np
import torch
from embedding import WeightedEncoder, mlp, observables
from torch import nn

MODES = ("coagulation", "pulse", "coupled")


@torch.no_grad()
def trajectories(root, split, mode, encoder, device, folder=None):
    """Read native state, exact inventories and interval forcing, grouped
    by box.
    """
    folder = folder or (
        root / "data" / split
        if mode == "pulse"
        else root / "data" / mode / split
    )
    records = []
    for path in sorted(folder.glob("families-*.npz")):
        with np.load(path) as d:
            mass = d["mass"].swapaxes(0, 1)
            counts = d["counts"]
            if counts.ndim == 2:
                counts = np.broadcast_to(counts[:, None, :], mass.shape[:-1])
            else:
                counts = counts.swapaxes(0, 1)
            gas = d["gas"].T
            x, p, diag = observables(
                mass,
                counts,
                d["density"],
                float(d["volume"]),
                gas,
                getattr(encoder, "grid", "pulse"),
            )

            def tensor(v, dtype=torch.float32):
                return torch.as_tensor(
                    np.ascontiguousarray(v), dtype=dtype, device=device
                )

            z = encoder.encode(tensor(x), tensor(counts))
            if "force" in d:
                force = d["force"].swapaxes(0, 1)
            else:
                time = d["time"]
                dt = np.diff(time)
                specs = json.loads(str(d["families"]))
                dry = np.repeat([f["dry_rh"] for f in specs], 2)
                wet = np.repeat([f["wet_rh"] for f in specs], 2)
                saturation = gas[:, 0] / wet
                force = np.zeros((mass.shape[0], len(dt), 5))
                force[..., 0] = dt
                force[..., 1] = mode in ("coagulation", "coupled")
                force[..., 2] = mode in ("pulse", "coupled")
                force[..., 3] = (dt == 0)[None, :] & bool(mode != "coagulation")
                force[..., 4] = np.where(
                    force[..., 3] > 0, (dry * saturation)[:, None], 0.0
                )
            records.append(
                dict(
                    z=z,
                    p=tensor(p, torch.float64),
                    diag=tensor(diag),
                    force=tensor(force, torch.float64),
                    id=tensor(np.repeat(d["ids"], 2), torch.long),
                )
            )
    if not records:
        raise ValueError(f"No native histories in {folder}")
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


def physical_features(p):
    """Express the exact inventories on comparable physical scales."""
    dry = (p[..., 2] + p[..., 3]).clamp_min(1e-25)
    return torch.stack(
        [
            torch.log1p(p[..., 0] / 1e11),
            torch.log1p(dry / 1e-5),
            p[..., 3] / dry,
            p[..., 1] / dry,
            p[..., 4] / 0.017,
        ],
        -1,
    ).float()


class Dynamics(nn.Module):
    """Predict rates for shape, water exchange and Brownian number loss."""

    def __init__(
        self,
        width,
        kind,
        history=1,
        hidden=96,
        feedforward=None,
        thermal_transfer=False,
    ):
        """Construct the networks and their persistent scaling buffers."""
        super().__init__()
        self.width = width
        self.kind = kind
        self.thermal_transfer = thermal_transfer
        output_size = width + (4 if thermal_transfer else 2)
        self.feedforward = feedforward or 2 * hidden
        self.history = history
        self.register_buffer("z_mean", torch.zeros(width))
        self.register_buffer("z_std", torch.ones(width))
        dims = width + 8
        if kind in ("ode", "residual"):
            self.rate = mlp([dims, hidden, hidden, output_size])
        elif kind == "recurrent":
            self.rnn = nn.GRU(dims, hidden, batch_first=True)
            self.readout = nn.Linear(hidden, output_size)
        elif kind == "transformer":
            self.project = nn.Linear(dims, hidden)
            self.position = nn.Parameter(torch.zeros(history, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden,
                4,
                self.feedforward,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(
                layer, 2, enable_nested_tensor=False
            )
            self.readout = nn.Linear(hidden, output_size)
        else:
            raise ValueError(kind)

    def features(self, z, p, f):
        """Combine shape, exact state, elapsed time and enabled processes."""
        time = f[..., 0:1].float()
        step_scale = torch.log1p(time / 0.0025)
        return torch.cat(
            [z, physical_features(p), step_scale, f[..., 1:3].float()], -1
        )

    def rates(self, token, history):
        """Return the shape and physical rates with the updated causal
        history.
        """
        series = torch.cat([history, token[:, None]], 1)[:, -self.history :]
        if self.kind in ("ode", "residual"):
            rates = self.rate(token)
        elif self.kind == "recurrent":
            rates = self.readout(self.rnn(series)[0][:, -1])
        else:
            hidden = self.project(series) + self.position[-series.shape[1] :]
            # Prior tokens cannot access later tokens at either layer.
            mask = torch.ones(
                (series.shape[1], series.shape[1]),
                device=series.device,
                dtype=torch.bool,
            ).triu(1)
            rates = self.readout(self.attention(hidden, mask=mask)[:, -1])
        return rates, series

    def advance(self, z, p, rates, f, fraction):
        """Integrate a bounded exchange and a positive binary-collision rate."""
        dt = f[..., 0].double() * fraction
        coag = f[..., 1].double()
        cond = f[..., 2].double()
        # Normalize elapsed time to the resolved interval for each regime.
        scale = dt / (0.125 * coag + 0.0025 * (1 - coag))
        z_next = z + scale[:, None].float() * rates[..., : self.width]
        # This implicit binary rate law preserves positive number at any
        # interval.
        coefficient = nn.functional.softplus(rates[..., -1]).double() * 1e-15
        number = p[..., 0] / (1 + dt * coag * coefficient * p[..., 0])
        dry = p[..., 2] + p[..., 3]
        if self.thermal_transfer:
            from relaxation import thermal_exchange

            raw = cond * thermal_exchange(p, dt, rates[..., -4:-1])
        else:
            raw = dry * cond * scale * rates[..., -2].double()
        transfer = torch.maximum(-p[..., 1], torch.minimum(raw, p[..., 4]))
        water = p[..., 1] + transfer
        gas = p[..., 4] - transfer
        return z_next, torch.stack(
            [number, water, p[..., 2], p[..., 3], gas], -1
        )

    def step(self, z, p, f, history):
        """Advance one observation interval and apply any prescribed reset."""
        token = self.features(z, p, f)
        rates, series = self.rates(token, history)
        if self.kind == "ode":
            mid_z, mid_p = self.advance(z, p, rates, f, 0.5)
            rates = self.rate(self.features(mid_z, mid_p, f))
        new_z, new_p = self.advance(z, p, rates, f, 1.0)
        imposed = f[..., 3] > 0
        new_p = torch.cat(
            [
                new_p[..., :4],
                torch.where(imposed, f[..., 4], new_p[..., 4])[..., None],
            ],
            -1,
        )
        return new_z, new_p, series

    def rollout(self, z, p, force, history=None):
        """Run without teacher corrections, including the vapor-reset
        observation.
        """
        if history is None:
            history = z.new_zeros((len(z), 0, self.width + 8))
        zs, ps = [z], [p]
        for i in range(force.shape[1]):
            z, p, history = self.step(z, p, force[:, i], history)
            zs.append(z)
            ps.append(p)
        return torch.stack(zs, 1), torch.stack(ps, 1)


def read_encoder(path, device):
    """Load the frozen feature map and diagnostic decoder."""
    info = torch.load(path, map_location=device, weights_only=True)
    dim = info["weights"]["diag_mean"].numel()
    representation = info.get("representation", "mixed")
    if representation in (
        "size-sections",
        "joint-sections",
        "size-material",
        "joint-material",
    ):
        from sections import SectionEncoder

        enc = SectionEncoder(
            info["width"],
            dim,
            representation.startswith("joint"),
            "material" in representation,
        )
    elif representation == "flux":
        from flux_encoder import FluxEncoder

        enc = FluxEncoder(info["width"], dim)
    elif representation.startswith("spectrum"):
        from equilibrium_encoder import EquilibriumEncoder

        enc = EquilibriumEncoder(
            info["width"], dim, int(representation.removeprefix("spectrum"))
        )
    elif representation == "moments":
        from embedding import MomentEncoder

        enc = MomentEncoder(info["width"], dim)
    else:
        enc = WeightedEncoder(info["width"], dim)
    enc = enc.to(device)
    enc.grid = info.get("grid", "pulse")
    enc.representation = representation
    enc.load_state_dict(info["weights"])
    return enc.eval()
