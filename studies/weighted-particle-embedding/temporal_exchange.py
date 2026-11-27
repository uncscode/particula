"""Mix relaxation, a neural transient and recurrent transfer."""

import torch
from forecast import physical_features
from mixture_exchange import MixtureExchange
from torch import nn


class TemporalExchange(MixtureExchange):
    """Predict a conservative water correction through each closed vapor leg."""

    def __init__(self, *args, physics_time=False, **kwargs):
        """Optionally express temporal expert ages in local exchange times."""
        super().__init__(*args, **kwargs)
        self.physics_time = physics_time

    def expert_ages(self, elapsed, p):
        """Map each physical interval to the shared network time grid."""
        ages = elapsed.reshape(len(p), -1)
        if self.physics_time:
            from relaxation import equilibrium_water, exchange_timescale

            tau = exchange_timescale(p, equilibrium_water(p, 1.0))
            ages = ages * (0.01 / tau[:, None])
        return ages.float()

    def make_experts(self, inputs, experts, expert_hidden):
        """Construct three responses from a common initial state."""
        if experts != 3:
            raise ValueError("The temporal mixture uses three exchange laws")
        self.transient = nn.Sequential(
            nn.Linear(inputs, expert_hidden),
            nn.SiLU(),
            nn.Linear(expert_hidden, 3),
        )
        self.context = nn.Linear(inputs, expert_hidden)
        self.transient_cell = nn.GRUCell(expert_hidden + 1, expert_hidden)
        self.readout = nn.Linear(expert_hidden, 1)
        self.register_buffer("knots_s", 0.0025 * 2.0 ** torch.arange(9))
        self.zero_corrections()

    def zero_corrections(self):
        """Keep the parent forecast exactly at initialization."""
        with torch.no_grad():
            self.transient[-1].weight[0].zero_()
            self.transient[-1].bias[0].zero_()
            self.readout.weight.zero_()
            self.readout.bias.zero_()

    def head_outputs(self, features, memory):
        """Use baseline coefficients for shape and relaxation."""
        return self.head(features)

    def recurrent_curve(self, inputs, ages):
        """Propagate on a fixed grid and evaluate the requested ages."""
        context = self.context(inputs).tanh()
        hidden = context
        levels = [torch.zeros_like(context[:, :1])]
        for t in self.knots_s.unbind():
            phase = torch.log1p(t / 0.0025).expand(len(inputs), 1)
            hidden = self.transient_cell(
                torch.cat((context, phase), dim=1), hidden
            )
            levels.append(self.readout(hidden))
        knots = torch.cat((self.knots_s.new_zeros(1), self.knots_s))
        hi = torch.searchsorted(knots, ages.contiguous()).clamp(
            1, len(knots) - 1
        )
        lo = hi - 1
        level = torch.cat(levels, dim=1)
        fraction = ((ages - knots[lo]) / (knots[hi] - knots[lo])).clamp(0, 1)
        return torch.lerp(level.gather(1, lo), level.gather(1, hi), fraction)

    def propagate(self, z, p, process, elapsed, coeff=None, memory=None):
        """Blend transfers and close the gas and particle water budget."""
        zz, ph = super().propagate(
            z, p, process, elapsed, coeff=coeff, memory=memory
        )
        features = torch.cat((z, physical_features(p), process.float()), 1)
        if memory is None:
            memory = self.start_memory(z)
        x = torch.cat((features, memory), 1) if memory is not None else features
        weights = self.router(x).softmax(1)
        ages = self.expert_ages(elapsed, p)
        amplitude, onset, decay = self.transient(x).unbind(1)
        rise = -torch.expm1(
            -ages / (0.005 * torch.exp(2 * onset.tanh()))[:, None]
        )
        envelope = torch.exp(
            -ages / (0.08 * torch.exp(3 * decay.tanh()))[:, None]
        )
        analytic = amplitude[:, None] * rise * envelope
        recurrent = self.recurrent_curve(x, ages) * envelope
        correction = weights[:, 1:2] * analytic + weights[:, 2:3] * recurrent
        if self.match_flux:
            correction = correction * rise
        delta = (
            correction.double()
            * p[:, 2:4].sum(1, keepdim=True)
            * process[:, 1:2]
        )
        delta = torch.minimum(ph[..., 4], torch.maximum(-ph[..., 1], delta))
        return zz, torch.stack(
            (
                ph[..., 0],
                ph[..., 1] + delta,
                ph[..., 2],
                ph[..., 3],
                ph[..., 4] - delta,
            ),
            -1,
        )
