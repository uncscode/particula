"""CUDA transfers for offline sparse-grid node evaluation."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _select(
    out: wp.array2d(dtype=wp.float64),
    lanes: wp.array(dtype=int),
    selected: wp.array2d(dtype=wp.float64),
):
    i, j = wp.tid()
    selected[i, j] = out[i, lanes[j]]


def evaluate_nodes(model, mass, tk, indices, *, profile=None):
    """Return only neutral target lanes and row status to the host controller."""
    out = np.empty((len(mass), len(indices)))
    status = np.empty(len(mass), np.int32)
    from time import perf_counter

    width = len(indices) + int(profile is not None)
    columns = np.array(indices) + model.output_slices["activity"].start
    if profile is not None:
        columns = np.append(columns, model.output_slices["evaluations"].start)
    with model.lock, wp.ScopedStream(model.stream):
        lanes = wp.array(
            columns,
            dtype=wp.int32,
            device=model.device,
        )
        for start in range(0, len(mass), model.capacity):
            end = min(start + model.capacity, len(mass))
            n = end - start
            begin = perf_counter()
            row = wp.array(
                mass[start:end], dtype=wp.float64, device=model.device
            )
            temp = wp.array(
                tk[start:end], dtype=wp.float64, device=model.device
            )
            result = model.output[:n]
            flags = model.status[:n]
            if profile is not None:
                wp.synchronize_stream(model.stream)
                profile["transfer_s"] += perf_counter() - begin
                profile["upload_bytes"] += (mass.shape[1] + 1) * n * 8
                first = wp.Event(model.device, enable_timing=True)
                last = wp.Event(model.device, enable_timing=True)
                wp.record_event(first)
            model.evaluate_into(
                row, temp, result, flags, model.scratch[:n], stream=model.stream
            )
            selected = wp.empty(
                (n, width), dtype=wp.float64, device=model.device
            )
            wp.launch(
                _select,
                selected.shape,
                inputs=[result, lanes, selected],
                device=model.device,
            )
            if profile is not None:
                wp.record_event(last)
                profile["kernel_s"] += (
                    wp.get_event_elapsed_time(first, last) * 1e-3
                )
            begin = perf_counter()
            data = selected.numpy()
            status[start:end] = flags.numpy()
            out[start:end] = data[:, : len(indices)]
            if profile is not None:
                profile["download_bytes"] += data.nbytes + n * 4
                profile["transfer_s"] += perf_counter() - begin
                profile["equilibrium_evaluations"] += int(
                    data[status[start:end] >= 0, -1].sum()
                )
    return out, status


@wp.func_native("""
    const int dim = levels.shape[1], component_count = masses.shape[1];
    const int water = mapping.data[0], target_count = coeff.shape[1];
    if (!active.data[row]) {
        miss.data[row] = 0;
        return;
    }
    // Recipe validation bounds coordinates to four dimensions.
    double u[4], largest = 0., dry = 0.;
    int reason = 0;
    for (int j = 0; j < component_count; j++) {
        double m = masses.data[row * component_count + j];
        if (!(m > 0.) || !isfinite(m))
            reason = 1;
        largest = fmax(largest, m);
    }
    for (int j = dim - 1; j < mapping.shape[0]; j++)
        dry +=
            masses.data[row * component_count + mapping.data[j]] / largest;
    if (!reason)
        for (int j = dim - 1; j < mapping.shape[0]; j++) {
            int k = mapping.data[j];
            if (fabs(masses.data[row * component_count + k] / largest -
                     dry * ratio.data[k]) > 2e-12 * dry)
                reason = 2;
        }
    // Undo the conditional dry shares from the last component back to the
    // first.
    for (int j = dim - 1; j >= 2; j--) {
        double m =
            masses.data[row * component_count + mapping.data[j - 1]] /
            largest;
        dry += m;
        u[j] = m / dry;
    }
    u[0] = log(masses.data[row * component_count + water] / largest) -
           log(dry);
    u[1] = temperature.data[row];
    int cell = 0;
    for (int j = 0; j < dim; j++) {
        double lo = bounds.data[2 * j], hi = bounds.data[2 * j + 1];
        u[j] = (u[j] - lo) / (hi - lo);
        if (!reason &&
            (!isfinite(u[j]) || u[j] < -1e-14 || u[j] > 1. + 1e-14))
            reason = 3;
        u[j] = fmax(0., fmin(1., u[j]));
        cell = cell * bins + min(bins - 1, (int)(bins * u[j]));
    }
    for (int k = 0; k < targets.shape[0]; k++) {
        int lane = targets.data[k];
        if (!reason && !isfinite(errors.data[cell * target_count + lane]))
            reason = 4;
    }
    for (int k = 0; k < targets.shape[0]; k++) {
        const int lane = targets.data[k], col = columns.data[k];
        estimates.data[row * targets.shape[0] + k] =
            reason ? 0. : errors.data[cell * target_count + lane];
        if (reason)
            continue;
        double value = 0.;
        for (int g = 0; g < levels.shape[0]; g++) {
            int base = 0, stride = 1, count = 0, axes[4], strides[4];
            double weight = 1.;
            for (int j = dim - 1; j >= 0; j--) {
                int level = levels.data[g * dim + j];
                if (level == 0) {
                    axes[count] = j;
                    strides[count++] = stride;
                    stride *= 2;
                } else {
                    int size = 1 << (level - 1);
                    double z = u[j] * size;
                    int index_in_axis = min((int)z, size - 1);
                    base += index_in_axis * stride;
                    stride *= size;
                    weight *= 1. - fabs(2. * (z - index_in_axis) - 1.);
                }
            }
            // Interior levels contribute one hat per axis; level zero has
            // two endpoints.
            for (int b = 0; b < (1 << count); b++) {
                int index = offsets.data[g] + base;
                double w = weight;
                for (int s = 0; s < count; s++) {
                    bool high = (b >> s) & 1;
                    index += high ? strides[s] : 0;
                    w *= high ? u[axes[s]] : 1. - u[axes[s]];
                }
                value += w * coeff.data[index * target_count + lane];
            }
        }
        out.data[row * out.shape[1] + col] = exp(value);
    }
    miss.data[row] = reason != 0;
    reason_out.data[row] = reason;
""")
def _lookup_row(
    row: int,
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    active: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
    ratio: wp.array(dtype=wp.float64),
    bounds: wp.array(dtype=wp.float64),
    levels: wp.array2d(dtype=int),
    offsets: wp.array(dtype=int),
    coeff: wp.array2d(dtype=wp.float64),
    errors: wp.array(dtype=wp.float64),
    bins: int,
    targets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    out: wp.array2d(dtype=wp.float64),
    estimates: wp.array2d(dtype=wp.float64),
    reason_out: wp.array(dtype=int),
    miss: wp.array(dtype=int),
): ...


@wp.kernel(enable_backward=False)
def _lookup(
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    active: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
    ratio: wp.array(dtype=wp.float64),
    bounds: wp.array(dtype=wp.float64),
    levels: wp.array2d(dtype=int),
    offsets: wp.array(dtype=int),
    coeff: wp.array2d(dtype=wp.float64),
    errors: wp.array(dtype=wp.float64),
    bins: int,
    targets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    out: wp.array2d(dtype=wp.float64),
    estimates: wp.array2d(dtype=wp.float64),
    reason_out: wp.array(dtype=int),
    miss: wp.array(dtype=int),
):
    _lookup_row(
        wp.tid(),
        masses,
        temperature,
        active,
        mapping,
        ratio,
        bounds,
        levels,
        offsets,
        coeff,
        errors,
        bins,
        targets,
        columns,
        out,
        estimates,
        reason_out,
        miss,
    )


class DeviceTable:
    """Bind portable coefficients to a device, target subset and direct layout."""

    def __init__(self, table, model, names):
        recipe = table.recipe

        def arr(x, dtype):
            return wp.array(
                np.ascontiguousarray(x), dtype=dtype, device=model.device
            )

        ratio = np.zeros(len(recipe.components))
        ratio[list(recipe.fixed)] = recipe.fixed_ratio
        self.columns = [
            model.output_slices["activity"].start
            + model.prepared.neutrals.index(k)
            for k in names
        ]
        self.inputs = [
            arr([recipe.water, *recipe.variable, *recipe.fixed], int),
            arr(ratio, wp.float64),
            arr(np.ravel(recipe.bounds), wp.float64),
            arr(table.arrays["levels"], int),
            arr(
                np.cumsum([0] + [len(v) for v in table.blocks.values()])[:-1],
                int,
            ),
            arr(table.arrays["coefficients"], wp.float64),
            arr(table.arrays["errors"].ravel(), wp.float64),
            table.arrays["errors"].shape[0],
            arr([recipe.targets.index(k) for k in names], int),
            arr(self.columns, int),
        ]
        self.device = model.device

    def lookup_into(
        self, mass, temp, active, out, errors, reason, miss, stream
    ):
        wp.launch(
            _lookup,
            mass.shape[0],
            inputs=[
                mass,
                temp,
                active,
                *self.inputs,
                out,
                errors,
                reason,
                miss,
            ],
            device=self.device,
            stream=stream,
        )


def evaluate_table(table, model, mass, tk, active):
    """Host-facing batched lookup. A mask dispatches direct rows on device."""
    from ._equilibrium import ActivityDomainError, ConvergenceError

    nt = len(table.columns)
    values = np.zeros((len(mass), nt))
    errors = np.zeros_like(values)
    reasons = np.full(len(mass), -1, np.int32)
    with model.lock, wp.ScopedStream(model.stream):
        for start in range(0, len(mass), model.capacity):
            stop = min(start + model.capacity, len(mass))
            n = stop - start
            sl = slice(start, stop)

            def arr(a, dtype):
                return wp.array(
                    np.ascontiguousarray(a), dtype=dtype, device=model.device
                )

            def buf(shape, dtype):
                return wp.empty(shape, dtype=dtype, device=model.device)

            m, t, mask = (
                arr(mass[sl], wp.float64),
                arr(tk[sl], wp.float64),
                arr(active[sl], int),
            )
            err, reason, miss = (
                buf((n, nt), wp.float64),
                buf(n, int),
                buf(n, int),
            )
            out, flags = model.output[:n], model.status[:n]
            table.lookup_into(m, t, mask, out, err, reason, miss, model.stream)
            model.evaluate_into(
                m,
                t,
                out,
                flags,
                model.scratch[:n],
                stream=model.stream,
                active=miss,
            )
            result = buf((n, nt), wp.float64)
            wp.launch(
                _select,
                result.shape,
                inputs=[out, table.inputs[-1], result],
                device=model.device,
            )
            codes = flags.numpy()
            bad = np.flatnonzero(codes < 0)
            if len(bad):
                k = int(bad[0])
                exc = (
                    ConvergenceError if codes[k] == -5 else ActivityDomainError
                )
                raise exc(
                    f"CUDA activity status {codes[k]}", batch_index=(start + k,)
                )
            values[sl] = np.where(active[sl, None], result.numpy(), 0.0)
            errors[sl] = np.where(active[sl, None], err.numpy(), 0.0)
            reasons[sl] = np.where(active[sl], reason.numpy(), -1)
    return values, errors, reasons
