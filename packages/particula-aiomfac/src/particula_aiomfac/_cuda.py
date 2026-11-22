"""Float64 Warp CUDA evaluation with caller-owned device output and workspace."""

from importlib.resources import files
from threading import Lock

import numpy as np
import warp as wp

from ._equilibrium import CARBONATE_EVALUATIONS_PER_DIMENSION
from .model import ActivityResult, ConvergenceError

# The same runtime layout is used by every mixture on a device. Parameter
# slices are uploaded once. Generated aliases refer only to offsets, never
# to a host pointer or to a particular composition.
DIMS = "N I C G M D sulfate carbonate water co2 h hs so hc co oh ca evaluations_per_dimension".split()
TABLES = (
    "conversion mw charge counts q area vol interaction main_index group_mw "
    "main_counts bgroup cgroup b_pair_b c_pair_b decay_b b_pair_c c_pair_c "
    "decay_c like three lambda_co2 pair_counts"
).split()
WORK = "initial amounts x mol sr mr lr lng psi theta den group ref mainx mainmw groupion".split()
OUTPUT = (
    "x activity neutral_log mol ion_log mean iap neutral_mask ion_mask pair_mask "
    "warning moles sr mr lr strength scale errors evaluations retries solid"
).split()


def _snippet():
    header = [
        f"const int {name}=layout.data[{j}];" for j, name in enumerate(DIMS)
    ]
    header += [
        "const double T=temperature.data[row];",
        "const double *mass=masses.data+row*D;",
    ]
    pos = len(DIMS)
    for names, array, prefix in (
        (TABLES, "parameters", ""),
        (WORK, "scratch", ""),
        (OUTPUT, "out", "out_"),
    ):
        for name in names:
            offset = f"layout.data[{pos}]"
            if array != "parameters":
                offset += f"+row*{array}.shape[1]"
            header.append(f"double *{prefix}{name}={array}.data+{offset};")
            pos += 1
    resources = files(__package__).joinpath("kernels")
    return "\n".join(
        header
        + [
            resources.joinpath(k + ".inc").read_text()
            for k in ("activity", "equilibrium", "output")
        ]
    )


@wp.func_native(_snippet())
def _calculate(
    row: int,
    layout: wp.array(dtype=wp.int32),
    parameters: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    scratch: wp.array2d(dtype=wp.float64),
): ...


@wp.kernel(enable_backward=False)
def _batch(
    layout: wp.array(dtype=wp.int32),
    parameters: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    scratch: wp.array2d(dtype=wp.float64),
):
    _calculate(
        wp.tid(), layout, parameters, masses, temperature, out, status, scratch
    )


@wp.kernel(enable_backward=False)
def _masked_batch(
    layout: wp.array(dtype=wp.int32),
    parameters: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    scratch: wp.array2d(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
):
    if active[wp.tid()] == 0:
        status[wp.tid()] = 0
        return
    _calculate(
        wp.tid(), layout, parameters, masses, temperature, out, status, scratch
    )


class CudaModel:
    """A prepared mixture on one CUDA device. Columns are in output_slices order."""

    def __init__(self, p, device, capacity=4096):
        if not wp.is_cuda_available():
            raise ValueError(
                "CUDA execution requested but no CUDA device is available"
            )
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("CUDA execution requires an NVIDIA CUDA device")
        self.prepared = p
        n, i, c, g, m, d = (
            len(p.neutrals),
            len(p.ions),
            p.ncation,
            len(p.q),
            len(p.bgroup),
            p.ncomponents,
        )
        s = n + i
        self.stream = wp.Stream(self.device)
        self.lock = Lock()
        transform = np.zeros((d + p.virtual_count, s))
        transform[p.neutral_indices, :n] = np.diag(1 / p.neutral_mw)
        transform[p.salt_indices, n:] = p.stoich / p.salt_mw[:, None]
        arrays = (
            transform[:d],
            p.neutral_mw,
            p.charge,
            p.counts,
            p.q,
            p.area,
            p.volume,
            p.interaction,
            p.group_map.argmax(axis=1),
            p.group_mw,
            p.main_counts,
            p.bgroup,
            p.cgroup,
            *p.bpair,
            *p.cpair,
            p.like,
            p.three,
            p.co2_lambda if p.carbonate else [],
            p.pair_counts,
        )
        layout = [
            n,
            i,
            c,
            g,
            m,
            d,
            int(p.sulfate),
            int(p.carbonate),
            p.water,
            p.co2,
        ]
        layout.extend(
            n + p.ions.index(j) if j in p.ions else -1
            for j in (205, 248, 261, 250, 262, 247, 221)
        )
        layout.append(CARBONATE_EVALUATIONS_PER_DIMENSION)
        flat = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
        layout.extend(np.cumsum([0] + [a.size for a in flat[:-1]]))
        sizes = [s, s, s, i, s, s, s, s, g * g, g, g, g, n, m, m, m]
        layout.extend(np.cumsum([0] + sizes[:-1]))
        self.work_width = sum(sizes)
        pairs = len(p.pairs)
        widths = [
            n,
            n,
            n,
            i,
            i,
            pairs,
            pairs,
            n,
            i,
            pairs,
            1,
            s,
            s,
            s,
            s,
            1,
            1,
            4,
            1,
            1,
            1,
        ]
        starts = np.cumsum([0] + widths[:-1])
        layout.extend(starts)
        self.output_width = sum(widths)
        self.output_slices = {
            key: slice(off, off + w)
            for key, off, w in zip(OUTPUT, starts, widths, strict=True)
        }
        self.capacity = min(
            capacity,
            max(
                1,
                (32 * 1024**2)
                // (8 * (self.work_width + self.output_width + d + 1)),
            ),
        )
        with wp.ScopedStream(self.stream):
            self.layout = wp.array(
                np.asarray(layout, np.int32), device=self.device
            )
            self.parameters = wp.array(np.concatenate(flat), device=self.device)
            self.scratch = wp.empty(
                (self.capacity, self.work_width),
                dtype=wp.float64,
                device=self.device,
            )
            self.output = wp.empty(
                (self.capacity, self.output_width),
                dtype=wp.float64,
                device=self.device,
            )
            self.status = wp.empty(
                self.capacity, dtype=wp.int32, device=self.device
            )
        wp.synchronize_stream(self.stream)

    def evaluate_into(
        self,
        masses,
        temperature,
        out,
        status,
        workspace,
        *,
        stream,
        active=None,
    ):
        """Enqueue contiguous float64 rows with status 0=fixed, 1=converged, negative=failure."""
        rows = masses.shape[0]
        args = (masses, temperature, out, status, workspace)
        p = self.prepared
        shapes = (
            (rows, p.ncomponents),
            (rows,),
            (rows, self.output_width),
            (rows,),
            (rows, self.work_width),
        )
        if stream.device != self.device:
            raise ValueError("stream device mismatch")
        for arr, shape in zip(args, shapes, strict=True):
            dtype = wp.int32 if arr is status else wp.float64
            if (
                arr.device != self.device
                or arr.shape != shape
                or arr.dtype != dtype
                or not arr.is_contiguous
            ):
                raise ValueError(
                    "CUDA array device, shape, dtype or stride mismatch"
                )
        if active is not None and (
            active.shape != (rows,)
            or active.device != self.device
            or active.dtype != wp.int32
            or not active.is_contiguous
        ):
            raise ValueError("invalid CUDA occupancy mask")
        if rows:
            wp.launch(
                _batch if active is None else _masked_batch,
                dim=rows,
                inputs=[
                    self.layout,
                    self.parameters,
                    *args,
                    *([] if active is None else [active]),
                ],
                device=self.device,
                stream=stream,
            )

    def evaluate(self, masses, temperature_K, *, diagnostics=False):
        p = self.prepared
        mass = np.asarray(masses, dtype=np.float64)
        if mass.ndim < 1 or mass.shape[-1] != p.ncomponents:
            raise ValueError("expected components on the last axis")
        shape = mass.shape[:-1]
        temp = np.broadcast_to(
            np.asarray(temperature_K, dtype=np.float64), shape
        )
        data = np.empty((temp.size, self.output_width))
        flags = np.empty(temp.size, dtype=np.int32)
        with self.lock, wp.ScopedStream(self.stream):
            for start in range(0, temp.size, self.capacity):
                stop = min(start + self.capacity, temp.size)
                rows = stop - start

                def dev(a):
                    return wp.array(a, dtype=wp.float64, device=self.device)

                inp = dev(
                    mass.flat[
                        start * p.ncomponents : stop * p.ncomponents
                    ].reshape(rows, p.ncomponents)
                )
                tk = dev(temp.flat[start:stop])
                dest = self.output[:rows]
                status = self.status[:rows]
                self.evaluate_into(
                    inp,
                    tk,
                    dest,
                    status,
                    self.scratch[:rows],
                    stream=self.stream,
                )
                flags[start:stop] = status.numpy()
                bad = np.flatnonzero(flags[start:stop] < 0)
                if len(bad):
                    k = start + bad[0]
                    idx = tuple(map(int, np.unravel_index(k, shape)))
                    if flags[k] == -5:
                        raise ConvergenceError(
                            "CUDA equilibrium failed", batch_index=idx
                        )
                    raise ValueError(
                        f"CUDA activity status {flags[k]} at batch index {idx}"
                    )
                data[start:stop] = dest.numpy()

        def get(key):
            sl = self.output_slices[key]
            return data[:, sl].reshape((*shape, sl.stop - sl.start))

        terms = None
        if diagnostics:
            terms = {k: get(k) for k in ("sr", "mr", "lr")}
            terms.update(
                ionic_strength=get("strength")[..., 0],
                molality_conversion=get("scale")[..., 0],
            )
            if p.carbonate or p.sulfate:
                terms.update(
                    precipitated_caso4_mol_per_kg_input=get("solid")[..., 0],
                    equilibrium_residuals=get("errors"),
                    evaluations=get("evaluations")[..., 0].astype(int),
                    retries=get("retries")[..., 0].astype(int),
                )
        return ActivityResult(
            p.neutrals,
            p.ions,
            p.pairs,
            *(get(k) for k in OUTPUT[:7]),
            *(get(k).astype(bool) for k in OUTPUT[7:10]),
            get("warning")[..., 0].astype(int),
            terms,
            get("moles"),
            flags.reshape(shape),
        )
