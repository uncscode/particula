"""Learn a weighted particle code that decodes to positive physical nodes."""

import numpy as np
import torch
from embedding import mlp
from torch import nn


def particle_inputs(mass, counts):
    """Compute intensive local features and fp64 normalized number weights."""
    dry = mass[..., 1:].sum(-1)
    number = counts.sum(1, keepdims=True)
    weight = counts / np.where(number > 0, number, 1)
    mean = (dry * weight).sum(1, keepdims=True)
    scale = np.where(dry > 0, dry, 1)
    f = mass[..., 2] / scale
    water = mass[..., 0] / scale
    bulk = (f * weight).sum(1, keepdims=True)
    x = np.stack(
        (
            np.log(np.maximum(dry / np.where(mean > 0, mean, 1), 1e-12)),
            f,
            np.log1p(water),
            (f - bulk) ** 2,
        ),
        axis=-1,
    )
    return np.where(counts[..., None] > 0, x, 0).astype(np.float32), weight


def section_features(x, weights):
    """Pool composition in eight overlapping dry-size bins."""
    size, f, water = (
        x[..., 0].double(),
        x[..., 1].double(),
        x[..., 2].double().expm1(),
    )
    centers = torch.linspace(-3, 3, 8, dtype=x.dtype, device=x.device)
    affinity = (-0.5 * ((size[..., None] - centers) / 0.5).square()).softmax(-1)
    basis = (weights[..., None] * affinity).transpose(1, 2)
    occupancy = basis.sum(-1, keepdim=True)
    dist = basis / occupancy.clamp_min(1e-24)
    values = torch.stack((size, f, f**2, f**3, water, water * f), -1)
    m = dist @ values
    mean = m[..., 1]
    spread = (m[..., 2] - mean**2).clamp_min(0).sqrt()
    moment = m[..., 3] - 3 * mean * m[..., 2] + 2 * mean**3
    skew = torch.where(
        moment.abs() > 1e-12, moment.sign() * moment.abs().pow(1 / 3), 0.0
    )
    features = torch.stack(
        (
            occupancy[..., 0],
            m[..., 0],
            mean,
            spread,
            skew,
            m[..., 4],
            m[..., 5] - mean * m[..., 4],
        ),
        -1,
    )
    return features.flatten(1).float()


class ProfileCodec(nn.Module):
    """Combine exact size moments with learned mixing-state coordinates."""

    def __init__(self, axes, context_mean, context_std, kind, hidden):
        """Initialize the profile-space model at the linear PCA control."""
        super().__init__()
        self.kind = kind
        if kind not in ("direct", "profile", "sections"):
            raise ValueError("Unknown profile encoder")
        for key, array in (
            ("axes", axes),
            ("context_mean", context_mean),
            ("context_std", context_std),
        ):
            self.register_buffer(
                key, torch.as_tensor(array, dtype=torch.float32)
            )
        k, features = axes.shape
        c = len(context_mean)
        input_width = {"direct": 4, "sections": 56, "profile": features + c}[
            kind
        ]
        self.encoder = mlp([input_width, hidden, hidden, k])
        self.decoder = mlp([k + c, hidden, hidden, features])
        if kind == "profile":
            for network in (self.encoder, self.decoder):
                nn.init.zeros_(network[-1].weight)
                nn.init.zeros_(network[-1].bias)

    def standardized_context(self, known):
        """Apply the training split's calibration to exact intensive inputs."""
        return (known - self.context_mean) / self.context_std

    def encode(self, known, x=None, weights=None, profiles=None, sections=None):
        """Pool particle features or compress a physical profile."""
        if self.kind == "direct":
            per_slot = self.encoder(x)
            return (per_slot.double() * weights[..., None]).sum(1).float()
        if self.kind == "sections":
            packed = (
                sections
                if sections is not None
                else section_features(x, weights)
            )
            return torch.where(
                weights.sum(1, keepdim=True) > 0, self.encoder(packed), 0.0
            )
        context = self.standardized_context(known)
        record = profiles @ self.axes.T + self.encoder(
            torch.cat((profiles, context), -1)
        )

        return torch.where(weights.sum(1, keepdim=True) > 0, record, 0.0)

    def decode(self, z, known):
        """Predict standardized local composition and water-loading profiles."""
        context = self.standardized_context(known)
        return z @ self.axes + self.decoder(torch.cat((z, context), -1))
