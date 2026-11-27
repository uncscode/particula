"""Causal recurrent mixture of relaxation heads for vapor exchange."""

import torch
from forecast import physical_features
from relaxation import Relaxation
from torch import nn


class MixtureExchange(Relaxation):
    """Blend learned corrections to a shared mass-conserving exchange law."""

    def __init__(self, *args, experts, memory_width, expert_hidden, **kwargs):
        """Construct a shared predictor, optional GRU and expert heads."""
        super().__init__(*args, **kwargs)
        if experts < 1 or memory_width < 0 or expert_hidden < 1:
            raise ValueError("Invalid mixture or recurrent state dimensions")
        self.memory_width = memory_width
        features = self.width + 7
        if memory_width:
            # Completed leg: initial state, predicted change, process, duration.
            self.cell = nn.GRUCell(2 * (self.width + 5) + 3, memory_width)
        inputs = features + memory_width
        self.router = nn.Linear(inputs, experts) if experts > 1 else None
        self.make_experts(inputs, experts, expert_hidden)

    def make_experts(self, inputs, experts, expert_hidden):
        """Construct independent neural coefficient corrections."""
        outputs = self.head[-1].out_features
        self.expert_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inputs, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, outputs),
            )
            for _ in range(experts)
        )
        for head in self.expert_heads:
            nn.init.normal_(head[-1].weight, std=1e-4)
            nn.init.zeros_(head[-1].bias)

    def zero_corrections(self):
        """Start a corrections-only fit at the original model exactly."""
        for head in self.expert_heads:
            nn.init.zeros_(head[-1].weight)

    def start_memory(self, z):
        """Use a cold start for each independent forecast."""
        return (
            z.new_zeros(len(z), self.memory_width)
            if self.memory_width
            else None
        )

    def advance_memory(
        self, memory, z, p, next_z, next_p, process, age, completed=None
    ):
        """Update after a completed leg and before the gas reset."""
        initial = physical_features(p)
        features = torch.cat(
            [
                z,
                initial,
                next_z - z,
                physical_features(next_p) - initial,
                process.float(),
                torch.log1p(age[:, None] / 0.0025).float(),
            ],
            -1,
        )
        active = age > 0
        if completed is not None:
            active = active & completed
        return torch.where(active[:, None], self.cell(features, memory), memory)

    def head_outputs(self, features, memory):
        """Route using the present aerosol, forcing and completed-leg memory."""
        if memory is None and self.memory_width:
            memory = self.start_memory(features)
        inputs = (
            torch.cat([features, memory], -1)
            if memory is not None
            else features
        )
        if self.router is None:
            correction = self.expert_heads[0](inputs)
        else:
            weights = self.router(inputs).softmax(-1)
            predictions = torch.stack(
                [head(inputs) for head in self.expert_heads], 1
            )
            correction = (weights[..., None] * predictions).sum(1)
        return self.head(features) + correction


def extend_model(
    parent,
    info,
    experts,
    memory_width,
    expert_hidden,
    temporal=False,
    physics_time=False,
):
    """Initialize each architecture from a common relaxation checkpoint."""
    if info.get("mixture"):
        raise ValueError(
            "Start the comparison from a shared single-head checkpoint"
        )
    mixture = dict(
        experts=experts, memory_width=memory_width, expert_hidden=expert_hidden
    )
    if physics_time:
        if not temporal:
            raise ValueError(
                "A physical time grid requires the temporal mixture"
            )
        mixture["physics_time"] = True
    model_class = MixtureExchange
    if temporal:
        from temporal_exchange import TemporalExchange

        model_class = TemporalExchange
    model = model_class(
        info["width"],
        info["hidden"],
        info["terms"],
        info["kelvin"],
        parent.equilibrium_knots,
        parent.rate_scaling,
        parent.match_flux,
        **mixture,
    ).to(parent.z_mean.device)
    model.head.load_state_dict(parent.head.state_dict())
    for name, source in parent.named_buffers():
        model.get_buffer(name).copy_(source)
    return model, dict(info, mixture=mixture, temporal_experts=temporal)
