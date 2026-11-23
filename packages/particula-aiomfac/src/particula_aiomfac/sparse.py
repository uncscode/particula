"""Portable float64 sparse interpolation of positive neutral activities.

Levels zero and one contain endpoint and midpoint hat functions. Subsequent
levels bisect each interval. A downward-closed set of tensor increments
preserves continuity while allowing different resolution along each axis.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from hashlib import sha256
from importlib.resources import files
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np

from . import UPSTREAM_REVISION, AIOMFACModel, MixtureSpec, __version__
from ._equilibrium import ActivityDomainError

DATA = files(__package__).joinpath("data/sparse")
FORMAT = 1
REASONS = ("hit", "absent", "dry_ratio", "bounds", "unqualified", "flux")


def _json(obj):
    return json.dumps(
        obj, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def _groups(items):
    """Parse integer subgroup IDs, including integer JSON object keys."""
    pairs = items.items() if hasattr(items, "items") else items
    groups = {}
    for key, count in pairs:
        if isinstance(key, str) and key.isdecimal():
            key = int(key)
        if type(key) is not int or key in groups:
            raise ValueError(f"invalid or duplicate subgroup ID {key!r}")
        groups[key] = count
    return tuple(sorted(groups.items()))


@dataclass(frozen=True)
class TableRecipe:
    """Mass coordinates use unit dry mass and ordered conditional dry shares.

    The first axis is natural log(water mass / dry mass), the second K.
    Each additional named dry component takes its share of the remaining
    dry mass. The last group retains the recipe's dry mass ratios.
    """

    name: str
    components: tuple
    dry_weights: tuple
    targets: tuple
    variable_dry: tuple
    bounds: tuple
    log_tolerances: tuple
    max_level: int

    def __post_init__(self):
        object.__setattr__(
            self,
            "components",
            tuple((name, _groups(groups)) for name, groups in self.components),
        )
        for key in ("dry_weights", "targets", "variable_dry", "log_tolerances"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))
        names = tuple(self.mixture.components)
        waters = [
            i
            for i, g in enumerate(self.mixture.components.values())
            if g == {16: 1}
        ]
        if len(names) != len(self.components) or len(waters) != 1:
            raise ValueError(
                "require unique component names and one water component"
            )
        object.__setattr__(self, "water", waters[0])
        weights = np.asarray(self.dry_weights)
        if (
            weights.shape != (len(names),)
            or not np.isfinite(weights).all()
            or weights[self.water] != 0
            or np.any(np.delete(weights, self.water) <= 0)
        ):
            raise ValueError("require positive weights for every dry component")
        if (
            not self.targets
            or len(set(self.targets)) != len(self.targets)
            or not set(self.targets) <= set(names)
            or len(self.targets) != len(self.log_tolerances)
            or not np.all(np.isfinite(self.log_tolerances))
            or min(self.log_tolerances) <= 0
        ):
            raise ValueError("invalid neutral targets or log tolerances")
        if (
            len(self.variable_dry) > 2
            or len(set(self.variable_dry)) != len(self.variable_dry)
            or not set(self.variable_dry) <= set(names) - {names[self.water]}
        ):
            raise ValueError("invalid variable dry shares")
        axes = np.asarray(self.bounds, float)
        if (
            axes.shape != (2 + len(self.variable_dry), 2)
            or not np.isfinite(axes).all()
            or np.any(axes[:, 0] >= axes[:, 1])
            or axes[0, 0] < np.log(np.finfo(float).tiny)
            or axes[0, 1] > np.log(np.finfo(float).max)
            or axes[1, 0] <= 0
            or np.any(axes[2:, 0] <= 0)
            or np.any(axes[2:, 1] >= 1)
            or type(self.max_level) is not int
            or not 1 <= self.max_level <= 16
            or not len(self.fixed)
        ):
            raise ValueError(
                "invalid table bounds, refinement limit or fixed group"
            )

    @cached_property
    def mixture(self):
        """Return the ordered analytical mixture bound by this recipe."""
        return MixtureSpec({k: dict(v) for k, v in self.components})

    @cached_property
    def variable(self):
        """Component indices with independent dry-share coordinates."""
        return tuple(
            tuple(self.mixture.components).index(k) for k in self.variable_dry
        )

    @cached_property
    def fixed(self):
        """Ordered indices of dry components with constrained ratios."""
        return tuple(
            i
            for i in range(len(self.components))
            if i not in (self.water, *self.variable)
        )

    @cached_property
    def fixed_ratio(self):
        """Normalized dry weights within the fixed-ratio group."""
        w = np.asarray(self.dry_weights, dtype=float)[list(self.fixed)]
        if not np.isfinite(w.sum()):
            w = w / w.max()
        ratio = w / w.sum()
        ratio.setflags(write=False)
        return ratio

    def states(self, unit):
        """Return analytical masses and temperature from unit-cube coordinates."""
        lo, hi = np.asarray(self.bounds).T
        x = lo + np.asarray(unit) * (hi - lo)
        mass = np.zeros((len(x), len(self.components)))
        mass[:, self.water] = np.exp(x[:, 0])
        rem = np.ones(len(x))
        for axis, i in enumerate(self.variable, 2):
            mass[:, i] = rem * x[:, axis]
            rem *= 1 - x[:, axis]
        mass[:, self.fixed] = rem[:, None] * self.fixed_ratio
        return mass, x[:, 1]

    def coordinates(self, mass, temperature):
        """Recover unit coordinates and a fallback reason for each analytical row."""
        m = np.asarray(mass, dtype=float)
        if m.ndim != 2 or m.shape[1] != len(self.components):
            raise ValueError("expected rows of ordered analytical components")
        t = np.broadcast_to(temperature, (len(m),))
        good = np.isfinite(m).all(axis=1) & (m > 0).all(axis=1)
        reason = np.where(good, 0, 1)
        # Scale before summing to support arbitrary common mass units.
        m = np.where(good[:, None], m, 1.0)
        m = m / m.max(axis=1, keepdims=True)
        dry = m[:, self.fixed].sum(axis=1)
        wrong = np.any(
            abs(m[:, self.fixed] - dry[:, None] * self.fixed_ratio)
            > 2e-12 * dry[:, None],
            axis=1,
        )
        reason[(reason == 0) & wrong] = 2
        shares = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in reversed(self.variable):
                dry = dry + m[:, i]
                shares.append(m[:, i] / dry)
            x = np.column_stack(
                [np.log(m[:, self.water]) - np.log(dry), t, *reversed(shares)]
            )
        lo, hi = np.asarray(self.bounds).T
        unit = (x - lo) / (hi - lo)
        inside = np.all(
            np.isfinite(unit) & (unit >= -1e-14) & (unit <= 1 + 1e-14), axis=1
        )
        reason[(reason == 0) & ~inside] = 3
        return np.clip(np.nan_to_num(unit), 0, 1), reason


def load_recipe(name):
    """Read a shipped recipe name or a user-supplied JSON recipe."""
    if isinstance(name, TableRecipe):
        return name
    path = Path(name)
    raw = (
        json.loads(path.read_text())
        if path.is_file()
        else json.loads(DATA.joinpath("recipes.json").read_text())[str(name)]
    )
    return TableRecipe(**raw)


def _shape(level):
    return tuple(2 if depth == 0 else 1 << (depth - 1) for depth in level)


def _nodes(level):
    axes = [
        np.arange(2)
        if depth == 0
        else (2 * np.arange(1 << (depth - 1)) + 1) / (1 << depth)
        for depth in level
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(level)
    )


def _interpolate(unit, blocks, ntarget):
    values = np.zeros((len(unit), ntarget))
    for level, coeff in blocks.items():
        base = np.zeros(len(unit), dtype=np.intp)
        weight = np.ones(len(unit))
        endpoints = []
        for j, (depth, size) in enumerate(
            zip(level, _shape(level), strict=True)
        ):
            base *= size
            if depth:
                z = unit[:, j] * size
                cell = np.minimum(z.astype(np.intp), size - 1)
                base += cell
                weight *= 1 - abs(2 * (z - cell) - 1)
            else:
                endpoints.append(
                    (j, int(np.prod(_shape(level)[j + 1 :], dtype=int)))
                )
        for bits in product((0, 1), repeat=len(endpoints)):
            idx = base
            w = weight
            for bit, (j, stride) in zip(bits, endpoints, strict=True):
                idx = idx + bit * stride
                w = w * (unit[:, j] if bit else 1 - unit[:, j])
            values += w[:, None] * coeff[idx]
    return values


class NodeEvaluator:
    """Evaluate ordered neutral lanes on the explicitly selected device."""

    def __init__(self, recipe, device, *, profile=False):
        self.profile = (
            {
                "kernel_s": 0.0,
                "transfer_s": 0.0,
                "upload_bytes": 0,
                "download_bytes": 0,
                "equilibrium_evaluations": 0,
            }
            if profile
            else None
        )
        self.recipe = recipe
        self.model = AIOMFACModel.prepare(
            recipe.mixture,
            execution="cpu" if device == "cpu" else "cuda",
            device=device,
        )
        prepared = getattr(self.model, "prepared", self.model)
        if missing := set(recipe.targets) - set(prepared.neutrals):
            raise ValueError(
                f"table targets must be dissolved neutral species: {sorted(missing)}"
            )
        self.indices = [prepared.neutrals.index(k) for k in recipe.targets]
        if any(
            recipe.mixture.components[k] == {173: 1} for k in recipe.targets
        ):
            raise ValueError(
                "aqueous CO2 has no generic volatile reference state"
            )
        self.device = device

    def evaluate(self, mass, tk):
        """Return dimensionless target activities and direct row status."""
        if self.device != "cpu":
            from ._sparse_cuda import evaluate_nodes

            return evaluate_nodes(
                self.model, mass, tk, self.indices, profile=self.profile
            )
        r = self.model._evaluate_rows(
            mass, tk, self.profile is not None, collect_failures=True
        )
        if self.profile is not None:
            self.profile["equilibrium_evaluations"] += int(
                np.sum(r.diagnostics.get("evaluations", 0))
            )
        values = r.neutral_activities[:, self.indices]
        values[r.solver_status < 0] = np.nan
        return values, r.solver_status


def _identity():
    return dict(
        format=FORMAT,
        version=__version__,
        revision=UPSTREAM_REVISION,
        parameters=sha256(
            files(__package__).joinpath("data/parameters.npz").read_bytes()
        ).hexdigest(),
    )


def _digest(metadata, arrays):
    h = sha256(_json(metadata).encode())
    for key, a in sorted(arrays.items()):
        h.update(_json([key, a.dtype.str, a.shape]).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class ActivityTable:
    """Immutable coefficients and cell-wise empirical log-activity errors."""

    def __init__(self, metadata, arrays, digest=None):
        self.recipe = TableRecipe(**metadata["recipe"])
        if metadata["model"] != _identity():
            raise ValueError(
                "incompatible sparse table format or model identity"
            )
        if set(arrays) != {"levels", "coefficients", "errors"}:
            raise ValueError("invalid sparse table array inventory")
        # Own immutable bytes so later edits to a caller's dictionary or NumPy
        # buffer cannot change an already-bound checkpoint identity.
        arrays = {
            key: np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(
                a.shape
            )
            for key, a in arrays.items()
        }
        self.digest = _digest(metadata, arrays)
        if digest is not None and self.digest != digest:
            raise ValueError("sparse table content hash mismatch")
        self._metadata_json = _json(metadata)
        self.arrays = MappingProxyType(arrays)
        levels = arrays["levels"]
        coeff = arrays["coefficients"]
        errors = arrays["errors"]
        dim = len(self.recipe.bounds)
        nt = len(self.recipe.targets)
        if (
            levels.ndim != 2
            or levels.dtype != np.dtype("<i4")
            or levels.shape[1] != dim
            or np.any(levels < 0)
            or np.any(levels > self.recipe.max_level)
            or coeff.ndim != 2
            or coeff.dtype != np.dtype("<f8")
            or coeff.shape[1] != nt
            or not np.isfinite(coeff).all()
        ):
            raise ValueError("invalid sparse levels or coefficients")
        if (
            errors.ndim != dim + 1
            or errors.dtype != np.dtype("<f8")
            or errors.shape != (4,) * dim + (nt,)
            or np.any(np.isnan(errors))
            or np.any(errors < 0)
        ):
            raise ValueError("invalid sparse qualification cells")
        blocks = {}
        start = 0
        for row in levels:
            level = tuple(map(int, row))
            if level in blocks or any(
                level[:j] + (v - 1,) + level[j + 1 :] not in blocks
                for j, v in enumerate(level)
                if v
            ):
                raise ValueError(
                    "sparse levels must have unique nodes and precede their children"
                )
            size = prod(_shape(level))
            blocks[level] = coeff[start : start + size]
            start += size
        if not blocks or start != len(coeff):
            raise ValueError("invalid sparse coefficient offsets")
        self.blocks = MappingProxyType(blocks)

    @property
    def metadata(self):
        """Return a detached copy of recipe, generator and model identity."""
        return json.loads(self._metadata_json)

    def prepare(self, *, device="cpu"):
        """Bind neutral lookup and direct fallback on one simulation device."""
        return PreparedTable(self, device)

    def save(self, path):
        """Serialize numerical arrays with a hash covering metadata and payload."""
        import tempfile

        path = Path(path)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False
        ) as stream:
            tmp = Path(stream.name)
        try:
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f,
                    metadata=_json(self.metadata),
                    digest=self.digest,
                    **self.arrays,
                )
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, name):
        """Load and verify a named resource or a portable NPZ file."""
        path = Path(name)
        src = path if path.is_file() else DATA.joinpath(str(name) + ".npz")
        with src.open("rb") as f, np.load(f, allow_pickle=False) as data:
            return cls(
                json.loads(str(data["metadata"])),
                {k: data[k] for k in data if k not in ("metadata", "digest")},
                str(data["digest"]),
            )

    @cached_property
    def interpolants(self):
        """Combine tensor increments into equivalent regular linear interpolants."""
        from collections import Counter

        weights = Counter()
        dim = len(self.recipe.bounds)
        levels = tuple(self.blocks)
        finest = tuple(np.max(levels, axis=0))
        # Fuse low-dimensional grids only when the redundant knots stay small.
        if prod((1 << int(n)) + 1 for n in finest) <= 4 * len(
            self.arrays["coefficients"]
        ):
            weights[finest] = 1
        else:
            for level in levels:
                for diff in product(
                    *[(0, 1) if depth else (0,) for depth in level]
                ):
                    weights[
                        tuple(a - b for a, b in zip(level, diff, strict=True))
                    ] += (-1) ** sum(diff)
        grids = []
        for level, w in weights.items():
            if not w:
                continue
            intervals = 2 ** np.asarray(level)
            unit = np.stack(
                np.meshgrid(
                    *[np.linspace(0, 1, n + 1) for n in intervals],
                    indexing="ij",
                ),
                axis=-1,
            )
            data = _interpolate(
                unit.reshape(-1, dim), self.blocks, len(self.recipe.targets)
            )
            grids.append(
                (intervals, data.T.reshape((-1, *(intervals + 1))) * w)
            )
        for scale, grid in grids:
            scale.setflags(write=False)
            grid.setflags(write=False)
        return tuple(grids)

    def lookup(self, mass, temperature):
        """Return approximate activities, log-error estimates and integer reasons."""
        u, reason = self.recipe.coordinates(mass, temperature)
        bins = self.arrays["errors"].shape[0]
        cell = np.minimum((u * bins).astype(int), bins - 1)
        error = self.arrays["errors"][tuple(cell.T)]
        reason[(reason == 0) & ~np.isfinite(error).all(axis=1)] = 4
        hit = reason == 0
        values = np.zeros((len(u), len(self.recipe.targets)))
        if hit.any():
            from scipy.ndimage import map_coordinates

            coords = u[hit].T
            logs = np.zeros((len(self.recipe.targets), coords.shape[1]))
            for scale, grid in self.interpolants:
                scaled = coords * scale[:, None]
                for j, field in enumerate(grid):
                    logs[j] += map_coordinates(
                        field, scaled, order=1, prefilter=False, mode="nearest"
                    )
            values[hit] = np.exp(logs.T)
        return values, np.where(hit[:, None], error, 0.0), reason


class PreparedTable:
    """Evaluate table targets in recipe order with same-device direct misses."""

    def __init__(self, table, device):
        self.table = table
        self.oracle = NodeEvaluator(table.recipe, device)
        if device != "cpu":
            from ._sparse_cuda import DeviceTable

            self.device_table = DeviceTable(
                table, self.oracle.model, table.recipe.targets
            )

    def evaluate(self, masses, temperature_K, *, active=None):
        """Return activities, log-error estimates and reason codes (-1 vacant)."""
        m = np.asarray(masses, dtype=float)
        if m.ndim != 2 or m.shape[1] != len(self.table.recipe.components):
            raise ValueError("expected a matrix in recipe component order")
        tk = np.broadcast_to(temperature_K, (len(m),))
        occupied = (
            np.ones(len(m), bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if occupied.shape != (len(m),):
            raise ValueError("invalid occupancy shape")
        if self.oracle.device != "cpu":
            from ._sparse_cuda import evaluate_table

            return evaluate_table(
                self.device_table, self.oracle.model, m, tk, occupied
            )
        values = np.zeros((len(m), len(self.table.recipe.targets)))
        errors = np.zeros_like(values)
        reason = np.full(len(m), -1, dtype=np.int32)
        rows = np.flatnonzero(occupied)
        values[rows], errors[rows], reason[rows] = self.table.lookup(
            m[rows], tk[rows]
        )
        miss = np.flatnonzero(reason > 0)
        if len(miss):
            try:
                values[miss] = self.oracle.model.evaluate(
                    m[miss], tk[miss]
                ).neutral_activities[:, self.oracle.indices]
            except ActivityDomainError as exc:
                raise type(exc)(
                    exc.reason, batch_index=(int(miss[exc.batch_index[0]]),)
                ) from exc
        return values, errors, reason


def build_table(recipe, *, device="cpu", tolerance_scale=1.0, profile=False):
    """Precompute sparse log activities and test local interpolation error.

    Construction probes determine admission. Independent study holdouts are
    evaluated only after the artifact is frozen.
    """
    recipe = load_recipe(recipe)
    if not np.isfinite(tolerance_scale) or tolerance_scale <= 0:
        raise ValueError("tolerance scale must be positive and finite")
    t0 = perf_counter()
    oracle = NodeEvaluator(recipe, device, profile=profile)
    setup = perf_counter() - t0
    dim = len(recipe.bounds)
    nt = len(recipe.targets)
    tol = np.array(recipe.log_tolerances) * tolerance_scale
    cache = {}
    failures = []
    elapsed = 0.0

    def exact(points):
        nonlocal elapsed
        keys = list(map(tuple, points))
        new = list(dict.fromkeys(k for k in keys if k not in cache))
        if new:
            start = perf_counter()
            masses, tk = recipe.states(np.asarray(new))
            values, codes = oracle.evaluate(masses, tk)
            elapsed += perf_counter() - start
            for i, (k, v, code) in enumerate(
                zip(new, values, codes, strict=True)
            ):
                if code < 0 or not (np.isfinite(v) & (v > 0)).all():
                    failures.append(
                        dict(
                            node_index=len(cache),
                            unit_coordinates=list(map(float, k)),
                            analytical_masses=masses[i].tolist(),
                            temperature_K=float(tk[i]),
                            backend_status=int(code),
                            reason="direct_failure"
                            if code < 0
                            else "undefined_log_activity",
                        )
                    )
                    cache[k] = np.full(nt, np.nan)
                else:
                    cache[k] = np.log(v)
        return np.asarray([cache[k] for k in keys])

    blocks = {}
    # Compute frontier surplus against its proper ancestors only. Distinct
    # dyadic increments have disjoint nodes; incomparable levels vanish there.
    base = (0,) * dim
    frontier = {base}
    accepted = set()
    while frontier:
        next_candidates = set()
        for level in sorted(frontier):
            if len(cache) + prod(_shape(level)) > 1_000_000:
                raise ValueError(
                    "sparse build exceeds one million nodes; narrow the domain or reduce max_level"
                )
            points = _nodes(level)
            ancestors = {
                depth: v
                for depth, v in blocks.items()
                if all(i <= j for i, j in zip(depth, level, strict=True))
            }
            surplus = exact(points) - _interpolate(points, ancestors, nt)
            clean = np.nan_to_num(surplus, nan=0.0)
            blocks[level] = clean
            significant = np.any(abs(clean) > tol / 8)
            if significant or sum(level) < 2:
                accepted.add(level)
                for d in range(dim):
                    if level[d] < recipe.max_level:
                        next_candidates.add(
                            level[:d] + (level[d] + 1,) + level[d + 1 :]
                        )
        frontier = {
            depth
            for depth in next_candidates
            if depth not in blocks
            and all(
                depth[:j] + (v - 1,) + depth[j + 1 :] in accepted
                for j, v in enumerate(depth)
                if v
            )
        }
    bins = 4
    # Cell centers, axis faces and diagonal probes are construction data.
    cells = np.array(list(product(range(bins), repeat=dim)))
    probes = np.vstack(
        [
            np.full(dim, 0.5),
            np.zeros(dim),
            np.ones(dim),
            0.5 + 0.5 * np.eye(dim),
            0.5 - 0.5 * np.eye(dim),
            np.random.default_rng(701).random((8, dim)),
        ]
    )
    u = ((cells[:, None, :] + probes) / bins).reshape(-1, dim)
    truth = exact(u)
    err = abs(truth - _interpolate(u, blocks, nt)).reshape(
        len(cells), len(probes), nt
    )
    err = np.maximum(3 * np.max(err, axis=1), tol / 4)
    err[np.any(~np.isfinite(err) | (err > tol), axis=1)] = np.inf
    # Failed training nodes invalidate every incident qualification cell.
    for failure in failures:
        scaled = np.array(failure["unit_coordinates"]) * bins
        err[np.all((cells <= scaled) & (scaled <= cells + 1), axis=1)] = np.inf
    meta = dict(
        model=_identity(),
        recipe=asdict(recipe),
        device=device,
        tolerance_scale=tolerance_scale,
        unique_nodes=len(cache),
        failures=failures,
    )
    arrays = dict(
        levels=np.asarray(list(blocks), dtype="<i4"),
        coefficients=np.concatenate(list(blocks.values())).astype("<f8"),
        errors=err.reshape((*[bins] * dim, nt)).astype("<f8"),
    )
    table = ActivityTable(meta, arrays)
    table.timings = dict(
        setup_s=setup,
        node_s=elapsed,
        total_s=perf_counter() - t0,
        **(oracle.profile or {}),
    )
    return table


def main():
    """Build a portable sparse table from an installed or JSON recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True).add_parser(
        "build"
    )
    cmd.add_argument("recipe")
    cmd.add_argument("--device", default="cpu")
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--tolerance-scale", type=float, default=1.0)
    args = parser.parse_args()
    table = build_table(
        args.recipe, device=args.device, tolerance_scale=args.tolerance_scale
    )
    t0 = perf_counter()
    table.save(args.output)
    print(
        json.dumps(
            dict(
                **table.timings,
                serialization_s=perf_counter() - t0,
                nodes=table.metadata["unique_nodes"],
                sha256=table.digest,
                device=args.device,
            )
        )
    )


if __name__ == "__main__":
    main()
