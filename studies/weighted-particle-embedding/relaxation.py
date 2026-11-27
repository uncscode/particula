"""A latent neural ODE with analytic relaxation during each closed interval."""

import torch
from embedding import mlp
from forecast import physical_features
from torch import nn

SATURATION_WATER_KG_M3 = 0.01713206014665131  # WaterBuck at 293 K
KELVIN_LENGTH_M = 2 * 0.072 * 0.01801528 / (1000.0 * 8.31446261815324 * 293.0)


def equilibrium_water(p, curvature):
    """Solve the finite-box kappa equilibrium with a Kelvin correction."""
    n = p[:, 0].clamp_min(1.0)
    dry_volume = (p[:, 2] + p[:, 3]) / 1400.0
    capacity = 1000.0 / 1400.0 * (0.03 * p[:, 2] + 0.75 * p[:, 3])
    total = p[:, 1] + p[:, 4]
    water = p[:, 1]
    for _ in range(4):
        wet_volume = (dry_volume + water / 1000.0) / n
        radius = (3 / (4 * torch.pi) * wet_volume.clamp_min(1e-33)).pow(1 / 3)
        sat = SATURATION_WATER_KG_M3 * torch.exp(
            KELVIN_LENGTH_M * curvature / radius
        )
        linear = sat - total + capacity
        discriminant = (linear.square() + 4 * capacity * total).sqrt()
        water = 2 * capacity * total / (linear + discriminant).clamp_min(1e-30)
    return water


def exchange_timescale(p, equilibrium):
    """Linearize mean-radius mass transfer in a finite isothermal box."""
    number = p[:, 0].clamp_min(1.0)
    dry = p[:, 2:4].sum(1)
    radius = (
        3
        * (dry / 1400 + equilibrium / 1000).clamp_min(1e-30)
        / (4 * torch.pi * number)
    ).pow(1 / 3)
    # Transition-regime transfer for water vapor at 293 K and 0.2 accommodation.
    diffusivity = 2.4e-5
    mean_speed = (8 * 8.31446261815324 * 293 / (torch.pi * 0.01801528)) ** 0.5
    knudsen = 3 * diffusivity / (mean_speed * radius)
    coefficient = 4 / (3 * 0.2)
    correction = (1 + knudsen) / (
        1 + (coefficient + 0.377) * knudsen + coefficient * knudsen.square()
    )
    conductance = 4 * torch.pi * radius * diffusivity * correction
    capacity = 1000 / 1400 * (0.03 * p[:, 2] + 0.75 * p[:, 3]) / number
    rh = ((p[:, 1] + p[:, 4] - equilibrium) / SATURATION_WATER_KG_M3).clamp(
        0, 0.999
    )
    slope = (
        SATURATION_WATER_KG_M3 * (1 - rh).square() / capacity.clamp_min(1e-30)
    )
    return (1 / (conductance * (number + slope))).clamp(1e-6, 10.0)


def thermal_exchange(p, dt, coefficients):
    """Relax toward finite-box equilibrium with a decaying transient mode."""
    shift, curvature, transient = coefficients.double().tanh().unbind(-1)
    eq = equilibrium_water(p, 1 + 0.85 * curvature)
    tau = exchange_timescale(p, eq) * torch.exp(2 * shift)
    age = dt / tau
    moved = -torch.expm1(-age)
    dry = p[:, 2:4].sum(1)
    return moved * (eq - p[:, 1] + 2 * dry * transient * torch.exp(-age))


def match_initial_slope(amplitudes, timescales, slope):
    """Project amplitudes onto the prescribed derivative at fixed asymptote."""
    rate = timescales.double().reciprocal()
    centered = rate - rate.mean(1, keepdim=True)
    original = amplitudes.double()
    remainder = slope - (rate * original).sum(1)
    shift = remainder / centered.square().sum(1).clamp_min(1e-30)
    return (original + shift[:, None] * centered).to(amplitudes.dtype)


class Relaxation(nn.Module):
    """Infer a spectrum of relaxation rates from the last imposed state."""

    def __init__(
        self,
        width,
        hidden=128,
        terms=8,
        kelvin=False,
        equilibrium_knots=0,
        rate_scaling="pilot",
        match_flux=False,
    ):
        """Use fixed log-time centers and learn their shifts and amplitudes."""
        super().__init__()
        self.width = width
        self.terms = terms
        self.kelvin = kelvin
        self.equilibrium_knots = equilibrium_knots
        self.rate_scaling = rate_scaling
        self.match_flux = match_flux
        if match_flux and (not kelvin or terms < 2):
            raise ValueError(
                "Flux matching requires equilibrium and two or more modes"
            )
        self.register_buffer("z_mean", torch.zeros(width))
        self.register_buffer("z_std", torch.ones(width))
        self.register_buffer("time_centers", torch.logspace(-4, 1, terms))
        self.head = mlp(
            [
                width + 7,
                hidden,
                hidden,
                hidden,
                (width + 2) * terms + 1 + int(kelvin and not equilibrium_knots),
            ]
        )

    def start_memory(self, z):
        """Initialize optional recurrent state for a fresh aerosol forecast."""
        return None

    def head_outputs(self, features, memory):
        """Predict relaxation coefficients from one leg's initial state."""
        return self.head(features)

    def coefficients(self, z, p, process, memory=None):
        """Calculate rates from physical state at the start of the leg."""
        outputs = self.head_outputs(
            torch.cat([z, physical_features(p), process.float()], -1), memory
        )
        amps = outputs[:, : self.terms * (self.width + 1)].reshape(
            -1, self.terms, self.width + 1
        )
        shifts = (
            outputs[
                :, self.terms * (self.width + 1) : self.terms * (self.width + 2)
            ].tanh()
            * 1.5
        )
        tau = self.time_centers * shifts.exp()
        binary = (
            nn.functional.softplus(
                outputs[:, self.terms * (self.width + 2)]
            ).double()
            * 1e-15
        )
        if self.kelvin:
            if self.equilibrium_knots:
                from equilibrium_encoder import spectral_equilibrium

                dry_shape = (z * self.z_std + self.z_mean)[
                    :, : self.equilibrium_knots
                ].double()
                equilibrium = spectral_equilibrium(p, dry_shape)
            else:
                correction = (1.0 + 0.85 * torch.tanh(outputs[:, -1])).double()
                equilibrium = equilibrium_water(p, correction)
            dry = p[:, 2:4].sum(-1).clamp_min(1e-24)
            total = ((equilibrium - p[:, 1]) / dry).float()
            # Close the slowest mode at the finite-box equilibrium.
            water = amps[..., -1]
            water = torch.cat(
                [water[:, :-1], (total - water[:, :-1].sum(-1))[:, None]], -1
            )
            amps = torch.cat([amps[..., : self.width], water[..., None]], -1)
        return amps, tau, binary

    def propagate(self, z, p, process, elapsed, coeff=None, memory=None):
        """Solve the modes and balance gas and particle water."""
        amps, tau, binary = (
            self.coefficients(z, p, process, memory) if coeff is None else coeff
        )
        time = elapsed.reshape(len(z), -1)
        basis = -torch.expm1(-time[..., None].float() / tau[:, None])
        response = torch.bmm(basis, amps)
        pred_z = z[:, None] + response[..., : self.width]
        if self.equilibrium_knots:
            k = self.equilibrium_knots
            # Water exchange preserves every dry-spectrum coordinate.
            fixed = torch.where(
                process[:, None, 0:1] > 0, pred_z[..., :k], z[:, None, :k]
            )
            pred_z = torch.cat([fixed, pred_z[..., k:]], -1)
        number = p[:, 0:1] / (
            1 + time * binary[:, None] * p[:, 0:1] * process[:, 0:1]
        )
        dry = p[:, 2:4].sum(1)
        water_mode = response[..., -1]
        if self.kelvin:
            if self.rate_scaling == "diffusion":
                equilibrium = p[:, 1] + dry * amps[..., -1].sum(1).double()
                characteristic = exchange_timescale(p, equilibrium)[
                    :, None
                ].float()
                water_tau = characteristic / (0.01 / tau + 0.125)
            else:
                volume = dry / (1400.0 * p[:, 0].clamp_min(1.0))
                radius = (3 / (4 * torch.pi) * volume.clamp_min(1e-33)).pow(
                    1 / 3
                )
                cap = (0.018 * (radius / 0.65e-6).pow(1.5)).clamp(0.0001, 0.08)
                water_tau = 1 / (1 / tau + 1 / cap[:, None].float())
            water_basis = -torch.expm1(
                -time[..., None].float() / water_tau[:, None]
            )
            water_amplitudes = amps[..., -1]
            if self.match_flux:
                from flux_encoder import bulk_rate

                flux_shape = (z * self.z_std + self.z_mean)[:, :2].double()
                slope = bulk_rate(flux_shape, p) / dry.clamp_min(1e-30)
                water_amplitudes = match_initial_slope(
                    water_amplitudes, water_tau, slope
                )
            water_mode = torch.bmm(water_basis, water_amplitudes[..., None])[
                ..., 0
            ]
        raw = water_mode.double() * dry[:, None] * process[:, 1:2]
        transfer = torch.minimum(p[:, 4:5], torch.maximum(-p[:, 1:2], raw))
        view = p[:, None].expand(-1, time.shape[1], -1)
        pred_p = torch.stack(
            [
                number,
                view[..., 1] + transfer,
                view[..., 2],
                view[..., 3],
                view[..., 4] - transfer,
            ],
            -1,
        )
        return pred_z, pred_p

    def rollout(self, z, p, force):
        """Continue the solution until the next reservoir reset."""
        length = force.shape[1]
        zs, ps = [z[:, None]], [p[:, None]]
        if length == 0:
            return zs[0], ps[0]
        # All outputs inside a closed leg use the same analytic coefficients.
        # Transfer the list of boundaries just once for the whole batch.
        resets = force[..., 3] > 0
        changes = torch.zeros_like(resets)
        changes[:, :-1] = (force[:, 1:, 1:3] != force[:, :-1, 1:3]).any(-1)
        boundary = (resets | changes).any(0)
        boundary[-1] = True
        ends = (boundary.nonzero().flatten() + 1).tolist()
        anchor_z, anchor_p = z, p
        age = p.new_zeros(len(p))
        memory = self.start_memory(z)
        start = 0
        for end in ends:
            segment = force[:, start:end]
            times = age[:, None] + segment[..., 0].cumsum(1)
            zz, pp = self.propagate(
                anchor_z, anchor_p, segment[:, 0, 1:3], times, memory=memory
            )
            last = pp[:, -1]
            reset = resets[:, end - 1]
            gas = torch.where(reset, segment[:, -1, 4], last[:, 4])
            last = torch.cat([last[:, :4], gas[:, None]], 1)
            zs.append(zz)
            ps.append(torch.cat([pp[:, :-1], last[:, None]], 1))
            renew = reset | changes[:, end - 1]
            if memory is not None and end < length:
                memory = self.advance_memory(
                    memory,
                    anchor_z,
                    anchor_p,
                    zz[:, -1],
                    pp[:, -1],
                    segment[:, 0, 1:3],
                    times[:, -1],
                    renew,
                )
            anchor_z = torch.where(renew[:, None], zz[:, -1], anchor_z)
            anchor_p = torch.where(renew[:, None], last, anchor_p)
            age = torch.where(renew, 0.0, times[:, -1])
            start = end
        return torch.cat(zs, 1), torch.cat(ps, 1)
