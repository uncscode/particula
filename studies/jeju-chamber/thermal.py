"""Thermometry, Newton wall exchange and conservative heat transport."""

import numpy as np
import pandas as pd
from flow import WIDTH
from observations import MANIFEST, START, sat, source
from scipy.ndimage import gaussian_filter1d

CP = 1005.0
R_AIR, R_WATER = 287.05, 461.52
BOXED_VOLUME = 22.4
THERMAL_KEYS = (
    "speed_ms",
    "exchange_m2s",
    "h_wm2k",
    "h_pressure_exponent",
    "ceiling_w",
    "wall_horizontal_half_difference_k",
    "wall_vertical_half_difference_k",
)
STAGES = ("conditioning", "cloud_formation", "mixed_phase_window")
STAGE_WEIGHTS = np.array([0.20, 0.35, 0.45])


def stage(second):
    """Define the common-clock windows before looking at the fits."""
    return np.searchsorted([600, 1080], second, side="right")


def archives():
    """Open every archived pressure and temperature table after checksum checks."""
    runs, schema = [], []
    for run in range(1, 7):
        name = f"Chamber_P,T_exp{run}.csv"
        file = next(item for item in MANIFEST["files"] if item["name"] == name)
        table = pd.read_csv(source(name), parse_dates=["measure datetime"])
        clock = table["measure datetime"]
        start = pd.Timestamp(f"{clock.iloc[0].date()} {START[run]}")
        second = (clock - start).dt.total_seconds().astype(int)
        assert not second.duplicated().any() and second.is_monotonic_increasing
        part = table.assign(second=second, run=run).rename(
            columns={
                "Twall": "wall_c",
                "Tair": "air_mean_c",
                "Tair_minimum": "air_min_c",
                "Pressure_hPa": "pressure_hpa",
            }
        )
        part["wall_minus_air_k"] = part.wall_c - part.air_mean_c
        part["air_deficit_k"] = part.air_mean_c - part.air_min_c
        part["stage"] = np.take(STAGES, stage(part.second))
        runs.append(part.drop(columns="measure datetime"))
        schema.append(
            (
                run,
                name,
                file["id"],
                len(table),
                int(second.iloc[0]),
                int(second.iloc[-1]),
                int(table.isna().sum().sum()),
                int(np.sum(np.diff(second) != 1)),
                file["computed_md5"],
            )
        )
    return pd.concat(runs, ignore_index=True), pd.DataFrame(
        schema,
        columns=[
            "run",
            "file",
            "figshare_file_id",
            "rows",
            "start_s",
            "stop_s",
            "missing_values",
            "irregular_steps",
            "verified_md5",
        ],
    )


class HeatModel:
    """Compute thermal contrasts with a shared measured mean-temperature anchor."""

    def __init__(self, data):
        n = int(data.second.max()) + 1
        fields, validity = [], []
        for e, h in data.groupby("run"):
            assert e == len(fields) + 1
            h = h.set_index("second").sort_index()
            cols = ["air_mean_c", "air_min_c", "wall_c", "pressure_hpa"]
            observed = h.reindex(np.arange(n))[cols]
            validity.append(observed.notna().all(axis=1).to_numpy())
            fields.append(
                observed.interpolate(limit_direction="both").to_numpy()
            )
        packed = np.stack(fields)
        self.air, self.minimum, self.wall = np.moveaxis(
            packed[..., :3] + 273.15, -1, 0
        )
        self.pressure = packed[..., 3] * 100
        self.density = self.pressure / (R_AIR * self.air)
        self.valid = np.stack(validity)
        self.seconds = np.arange(n)
        self.phase = stage(self.seconds)
        self.mean_rate = gaussian_filter1d(
            np.gradient(self.air, axis=1), 8, axis=1
        )
        self.dry_rate = (
            (R_AIR / CP)
            * self.air
            * gaussian_filter1d(
                np.gradient(np.log(self.pressure), axis=1),
                8,
                axis=1,
            )
        )

    @staticmethod
    def geometry(grid, operator):
        """Return each box’s area on the six walls and a mean-probe operator."""
        nx, nz = grid.nx, grid.nz
        ix = np.tile(np.arange(nx), nz)
        iz = np.repeat(np.arange(nz), nx)
        face = np.column_stack(
            (
                (ix == 0) * WIDTH * grid.dz,
                (ix == nx - 1) * WIDTH * grid.dz,
                np.full(grid.size, grid.dx * grid.dz),
                np.full(grid.size, grid.dx * grid.dz),
                (iz == 0) * WIDTH * grid.dx,
                (iz == nz - 1) * WIDTH * grid.dx,
            )
        )
        top = np.where(
            iz == nz - 1,
            np.exp(-0.5 * ((grid.x[ix] - WIDTH / 2) / 0.60) ** 2),
            0,
        )
        mean = np.ones(grid.size)
        if operator == "interior":
            # Sensitivity for the paper's four-probe mean excluding warm regions.
            mean = np.where(iz == nz - 1, 0.2, 1.0)
            mean *= np.where(ix == nx // 2, 0.6, 1.0)
        return face, top / top.sum(), mean / mean.sum()

    def solve(
        self,
        parameters,
        grid,
        operator="volume",
        dt=5,
        wall_bias=0.0,
        retain=False,
        always_cool=False,
        mixing_matrices=None,
    ):
        """Integrate mixing plus Newton wall exchange and archive local forcing."""
        p = np.atleast_2d(parameters)
        if p.shape[1] != len(THERMAL_KEYS):
            raise ValueError("Incomplete thermal parameter vector.")
        n = len(p)
        cells = grid.size
        face, top, mean = self.geometry(grid, operator)
        surface = face.sum(axis=1)
        volume = BOXED_VOLUME / cells
        matrix = (
            np.stack([grid.matrix(row[0], row[1], dt) for row in p])
            if mixing_matrices is None
            else mixing_matrices
        )
        cap = CP * volume * self.density
        offsets = np.zeros((n, 6))
        offsets[:, 0], offsets[:, 1] = -p[:, 5], p[:, 5]
        offsets[:, 4], offsets[:, 5] = -p[:, 6], p[:, 6]
        assert np.allclose(offsets.mean(axis=-1), 0)
        local_offset = offsets @ face.T / surface
        index = self.seconds[::dt]
        coeff = (
            p[:, 2, None, None]
            * (self.pressure[None, :, ::dt] / 1e5) ** p[:, 3, None, None]
            / cap[None, :, ::dt]
        )
        target = self.wall[:, ::dt] - self.air[:, ::dt] + wall_bias
        if always_cool:
            target = -abs(target)
        theta = np.zeros((n, 6, cells))
        sq = np.zeros((n, 6, 3))
        error = np.zeros_like(sq)
        counts = np.zeros((6, 3))
        warm = np.zeros((n, 6))
        expanding_count = np.zeros(6)
        extra_cooling = np.zeros((n, 6))
        late_count = np.zeros(6)
        lower = np.repeat(np.arange(grid.nz) < grid.nz - 1, grid.nx).astype(
            float
        )
        lower /= max(1.0, lower.sum())
        if retain:
            forcing = np.zeros((n, 6, len(index), 3 * cells), dtype=np.float32)
            rates = np.zeros((n, 6, len(index), 6), dtype=np.float32)
        pattern = p[:, 4, None] * top[None, :]
        film_conductance = coeff[..., None] * surface
        alpha = -np.expm1(-film_conductance * dt)
        heat_factor = np.divide(
            alpha,
            film_conductance,
            out=np.full_like(alpha, dt),
            where=film_conductance > 0,
        )
        for j, k in enumerate(index):
            ph = self.phase[k]
            alive = self.valid[:, k]
            residual = theta.min(axis=-1) - (
                self.minimum[:, k] - self.air[:, k]
            )
            sq[:, :, ph] += (residual * alive) ** 2
            error[:, :, ph] += residual * alive
            counts[:, ph] += alive
            if k >= 600:
                warm += ((theta @ top) > (theta @ lower)) * alive
                expanding_count += alive
            wall_difference = (
                target[None, :, j, None] + local_offset[:, None, :] - theta
            )
            heat = pattern[:, None, :] / cap[None, :, k, None]
            transport = matrix[k // 60] if matrix.ndim == 5 else matrix[:, None]
            transported = (transport @ theta[..., None])[..., 0]
            wall_after_exchange = wall_difference + (theta - transported)
            wall_change = alpha[:, :, j] * wall_after_exchange
            upper_change = heat * heat_factor[:, :, j]
            uncentered = transported + wall_change + upper_change
            mean_change = uncentered @ mean
            common = (
                self.mean_rate[:, k] - self.dry_rate[:, k] - mean_change / dt
            )
            if k >= 1080:
                extra_cooling += np.maximum(0.0, -common * 60) * alive
                late_count += alive
            if retain:
                forcing[:, :, j, :cells] = theta
                tw = (
                    self.air[None, :, k, None]
                    + target[None, :, j, None]
                    + offsets[:, None, :]
                )
                _, saturated = sat(tw)
                saturation = saturated / (R_WATER * tw) * 1000
                forcing[:, :, j, cells : 2 * cells] = (
                    saturation @ face.T / surface
                )
                forcing[:, :, j, 2 * cells :] = film_conductance[:, :, j]
                rates[:, :, j, 0] = (wall_change @ mean) / dt
                rates[:, :, j, 1] = (upper_change @ mean) / dt
                rates[:, :, j, 2] = self.dry_rate[:, k]
                rates[:, :, j, 3] = common
                rates[:, :, j, 4] = (transported @ mean) / dt
                rates[:, :, j, 5] = wall_change.mean(axis=-1) / dt
            theta = uncentered - mean_change[..., None]
        rmse = np.sqrt(sq / counts)
        bias = error / counts
        result = {
            "rmse": rmse,
            "bias": bias,
            "run_loss": np.sqrt((rmse**2) @ STAGE_WEIGHTS),
            "warm_ceiling_fraction": warm / expanding_count,
            "unresolved_late_cooling_kmin": extra_cooling / late_count,
        }
        if retain:
            result.update(forcing=forcing, rates=rates)
        return result
