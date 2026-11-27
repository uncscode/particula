"""Fixed-byte sectional controls with learned diagnostic and response
readouts.
"""

import argparse
import json
from pathlib import Path

import torch
from embedding import WeightedEncoder, errors_by_pair, load_split, state_input


def sections(x, counts, width, composition, limits=(0.35, 3.5)):
    """Number fractions on fixed log dry diameter and dry composition edges."""
    classes = 4 if composition else 1
    size_bins = width // classes
    size = (
        (
            (x[..., 0] - torch.log(x.new_tensor(limits[0])))
            / (torch.log(x.new_tensor(limits[1] / limits[0])))
            * size_bins
        )
        .floor()
        .long()
    )
    comp = (x[..., 2] * classes).floor().long().clamp(0, classes - 1)
    idx = size.clamp(0, size_bins - 1) * classes + comp
    total = counts.sum(-1, keepdim=True)
    out = x.new_zeros(*counts.shape[:-1], width)
    return out.scatter_add_(
        -1, idx, counts / torch.where(total > 0, total, torch.ones_like(total))
    )


def material_sections(x, counts, width, composition):
    """Store represented number and all three material fractions by wet size."""
    classes = 2 if composition else 1
    size_bins = width // (4 * classes)
    if width % (4 * classes) or size_bins < 1:
        raise ValueError("Width must hold four inventories in each section")
    # The two dry materials have equal density in the study.
    dry_volume = torch.exp(3 * x[..., 0])
    wet_volume = torch.exp(3 * x[..., 1])
    dry = 1400.0 * dry_volume
    inventories = torch.stack(
        [
            torch.ones_like(dry),
            1000.0 * (wet_volume - dry_volume).clamp_min(0),
            dry * (1 - x[..., 2]),
            dry * x[..., 2],
        ],
        -1,
    )
    represented = counts[..., None] * inventories
    total = represented.sum(-2, keepdim=True)
    normalized = represented / torch.where(
        total > 0, total, torch.ones_like(total)
    )
    size = (
        (
            (x[..., 1] - torch.log(x.new_tensor(0.045)))
            / torch.log(x.new_tensor(100.0))
            * size_bins
        )
        .floor()
        .long()
    )
    group = (x[..., 2] * classes).floor().long().clamp(0, classes - 1)
    idx = (size.clamp(0, size_bins - 1) * classes + group)[..., None].expand_as(
        normalized
    )
    out = x.new_zeros(*counts.shape[:-1], size_bins * classes, 4)
    return out.scatter_add_(-2, idx, normalized).flatten(-2)


class SectionEncoder(WeightedEncoder):
    """Decode fixed dry-size or joint dry-size/composition sections."""

    def __init__(self, width, diag_dim, composition, material=False):
        """Use the same physical record and learned diagnostic readout."""
        super().__init__(width, diag_dim)
        self.width = width
        self.composition = composition
        self.material = material
        self.phi = torch.nn.Identity()

    def encode(self, x, counts):
        """Return each section's fraction of total represented number."""
        if self.material:
            return material_sections(x, counts, self.width, self.composition)
        return sections(x, counts, self.width, self.composition, (0.045, 4.5))


def controls(root, device):
    """Fit the sectional readouts and score their paired forecasts."""
    train = load_split(root / "data/train", device)
    valid = load_split(root / "data/validation", device)
    records = []
    # Identical nonlinear readouts for each sectional representation.
    for width in (8, 16, 32):
        for composition in (False, True):
            from embedding import mlp

            torch.manual_seed(711)
            ti = sections(train["x"], train["counts"], width, composition)
            vi = sections(valid["x"], valid["counts"], width, composition)

            def inputs(d, z):
                return torch.cat(
                    [z, state_input(d["physical"]).float(), d["forcing"]], -1
                )

            train_input = inputs(train, ti)
            valid_input = inputs(valid, vi)
            dry = train["physical"][..., 2:4].sum(-1, keepdim=True).float()
            y = train["future"] / dry
            diag = train["diag"]
            center, scale = (
                diag.mean((0, 1)),
                diag.flatten(0, 1).std(0).clamp_min(1e-8),
            )
            net = mlp([train_input.shape[-1], 96, 96, 2 + diag.shape[-1]]).to(
                device
            )
            opt = torch.optim.AdamW(net.parameters(), lr=0.0015)
            n, t = y.shape[:2]
            cs = (y[1::2] - y[::2]).square().mean(0).sqrt().clamp_min(0.003)
            for _ in range(3000):
                a = 2 * torch.randint(n // 2, (128,), device=device)
                row = torch.stack([a, a + 1])
                obs = torch.randint(t, (128,), device=device)
                obs[:32] = 0
                target = y[row, obs]
                v = net(train_input[row, obs])
                error = (
                    v[1, :, :2] - v[0, :, :2] - target[1] + target[0]
                ) / cs[obs]
                loss = (
                    v[..., :2] - target
                ).square().mean() + 0.25 * torch.nn.functional.smooth_l1_loss(
                    error, torch.zeros_like(error)
                )
                loss = (
                    loss
                    + (v[..., 2:] - (diag[row, obs] - center) / scale)
                    .square()
                    .mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                prediction = net(valid_input)
                p = (
                    prediction[..., :2]
                    * valid["physical"][..., 2:4].sum(-1, keepdim=True).float()
                )
                record = dict(
                    width=width,
                    composition=composition,
                    wet=errors_by_pair(p[:, 0, 0], valid["future"][:, 0, 0]),
                    full=errors_by_pair(p[:, 0, 1], valid["future"][:, 0, 1]),
                    diagnostic_nrmse=float(
                        (prediction[..., 2:] - (valid["diag"] - center) / scale)
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    payload_bytes=width * 4 + 5 * 8,
                )
                records.append(record)
                torch.save(
                    dict(
                        model=net.state_dict(),
                        width=width,
                        composition=composition,
                        center=center,
                        scale=scale,
                    ),
                    root / f"section-w{width}-c{int(composition)}.pt",
                )
                print(record, flush=True)
    (root / "sections.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    controls(a.root, a.device)
