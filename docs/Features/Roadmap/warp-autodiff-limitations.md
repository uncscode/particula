# Warp Autodiff: Limitations and Stochastic Process Handling

Reference notes for the differentiable-simulation goal in the
[Data-Oriented Design and GPU Roadmap](data-oriented-gpu.md#differentiability-for-global-optimization-warp-autodiff).

This page records how NVIDIA Warp automatic differentiation works, the
constraints it imposes on kernel authoring, and the options for making
stochastic aerosol processes (especially particle-resolved coagulation)
compatible with gradient-based global optimization.

- Source: NVIDIA Warp 1.14 user guide, Differentiability and Limitations pages.
  The relevant patterns are reproduced inline below so this page is usable
  offline without web access.
- Code references are to the current particula GPU kernels.
- Scope: informs, but does not replace, the roadmap milestones. Treat the
  stochastic-coagulation approach as an open decision.

## How `wp.Tape` Works

Warp generates a forward and a backward (adjoint) version of every kernel by
default. Gradients are computed with reverse-mode automatic differentiation
recorded on a `wp.Tape`.

- Arrays that participate in gradient computation must be allocated with
  `requires_grad=True`, for example
  `wp.zeros(n, dtype=wp.float64, device="cuda", requires_grad=True)`.
- Kernel launches are recorded inside a `with tape:` block (the forward pass).
- `tape.backward(loss)` runs the reverse pass and populates `array.grad` for
  each input array.
- Output gradients are consumed (zeroed) by default during backward. Use
  `retain_grad=True` only on arrays where each element is written at most once,
  or gradients will be double-counted.
- `wp.copy()`, `wp.clone()`, and `array.assign()` are differentiable.
- Warp provides `wp.autograd.gradcheck()` and `jacobian_fd()` to compare
  autodiff gradients against finite differences. Use these as parity tests for
  every differentiable kernel.

Because Warp uses source-code transformation, the backward pass must reconstruct
or store intermediate values from the forward pass. This is the root cause of
most limitations below.

## Self-Contained Code Patterns

These are the minimal Warp patterns needed to implement and validate a
differentiable kernel. They are reproduced from the Warp 1.14 docs so no web
access is required.

### Forward and backward pass with a tape

```python
import warp as wp

@wp.kernel
def kernel_func(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] ** 2.0 + 3.0 * x[tid] + 1.0

x = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True)
y = wp.zeros_like(x)

tape = wp.Tape()
with tape:                                   # forward pass, recorded
    wp.launch(kernel_func, x.shape, inputs=[x], outputs=[y])

tape.backward(grads={y: wp.ones_like(y)})    # reverse pass
print(x.grad)                                # -> [5. 7. 9.]

# reuse buffers for another backward pass only after:
tape.zero()
```

Notes:

- Only arrays with `requires_grad=True` accumulate gradients.
- `tape.backward(loss)` takes either a scalar loss array or a `grads=` dict that
  seeds output adjoints (useful for Jacobian rows).
- Disable backward entirely with `wp.config.enable_backward = False` or
  `@wp.kernel(enable_backward=False)` to save memory when gradients are not
  needed.

### Validating gradients (parity test)

```python
import warp.autograd

passed = wp.autograd.gradcheck(
    my_kernel, dim=n, inputs=[a, b], outputs=[out],
    raise_exception=False, show_summary=True,
)
assert passed
```

`gradcheck` compares autodiff gradients against finite differences
(`jacobian_fd`). Make this the standard test for every differentiable kernel.

### Guarding non-finite gradients with a custom adjoint

```python
@wp.func
def safe_sqrt(x: float):
    return wp.sqrt(x)

@wp.func_grad(safe_sqrt)
def adj_safe_sqrt(x: float, adj_ret: float):
    # Without this guard, x = 0 yields an inf gradient that poisons the graph.
    if x > 0.0:
        wp.adjoint[x] += 1.0 / (2.0 * wp.sqrt(x)) * adj_ret
```

The custom-grad signature is the forward inputs plus the adjoints of the forward
outputs. Update input partials through the `wp.adjoint[...]` dictionary. This is
the tool for Kelvin, diffusivity, and any `sqrt`/division term that can hit zero.

### Custom replay for per-thread state (atomics)

Atomic scatter breaks gradients because the adjoint of `wp.atomic_add` cannot map
threads back to indices. Store the index in the forward pass and replay it:

```python
@wp.func
def reversible_increment(buf: wp.array(dtype=int), buf_index: int, value: int,
                         thread_values: wp.array(dtype=int), tid: int):
    next_index = wp.atomic_add(buf, buf_index, value)
    thread_values[tid] = next_index          # remember mapping
    return next_index

@wp.func_replay(reversible_increment)
def replay_reversible_increment(buf: wp.array(dtype=int), buf_index: int,
                                value: int, thread_values: wp.array(dtype=int),
                                tid: int):
    return thread_values[tid]                 # reuse stored mapping in backward
```

Relevant if a differentiable coagulation ever uses atomic scatter to write
merged mass.

### In-place math rules (worked example)

```python
@wp.kernel
def inplace(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    i = wp.tid()
    a[i] -= b[i]     # OK: += and -= are differentiable
    # a[i] *= b[i]   # WRONG gradients: *= and /= are not supported
```

Vector/matrix/quaternion components may be assigned once, then only updated with
`+=`/`-=`; a second `=` assignment invalidates their gradients.

### Dynamic loop pitfall and workaround

```python
# Data-dependent bound; backward pass does NOT replay this loop.
# Safe ONLY when array adjoints do not need loop intermediates (pure +=/-=).
@wp.kernel
def sum_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float), iters: int):
    s = float(0.0)
    for _ in range(iters):     # dynamic loop
        s += x[0]
    out[0] = s                 # correct grads (add-only)

# A product (prod *= x[i]) over a dynamic loop gives WRONG gradients,
# because adjoints depend on intermediate prod values that are not recomputed.
```

Workarounds: use a static loop (`range(3)`) that Warp unrolls up to
`max_unroll`; or store each iteration's intermediate in a preallocated array; or
move an add-only loop body into a `@wp.func`.

### Detecting silent overwrites in tests

```python
wp.config.verify_autograd_array_access = True   # warns on write-after-read
```

Enable this in the differentiable test suite. It disables kernel caching, so
keep it test-only. It does not yet cover arrays packed inside `@wp.struct`, so
grad-tracked fields of `WarpParticleData`/`WarpGasData` still need manual review.

## Kernel-Authoring Limitations for Differentiable Code

| Concern | Rule | Implication for particula |
| --- | --- | --- |
| In-place add/sub | `+=` and `-=` are differentiable | Safe for mass accumulation |
| In-place mul/div | `*=` and `/=` are **not** differentiable; wrong adjoints | Avoid in optimization-path kernels |
| Component assignment | Vector/matrix/quat component may be assigned once; then only `+=`/`-=`. Re-assignment with `=` invalidates gradients | Applies if we adopt vector types |
| Dynamic loops | Loops with data-dependent bounds are **not** replayed/unrolled in backward; intermediate values are not recomputed | The per-box coagulation loop is a problem (see below) |
| Static loops | Loops up to `max_unroll` are unrolled and store intermediates | Prefer fixed-count loops on the optimization path |
| Atomics | `wp.atomic_add` adjoint does not know thread-to-index mapping; can need a custom replay function | Relevant if we use atomic scatter for merges |
| Array overwrite | Writing to an array after reading it breaks gradients (write-after-read). Set `wp.config.verify_autograd_array_access=True` to detect | Applies to in-place mass merges |
| Non-finite grads | `wp.sqrt(0)`, division by zero produce `inf`/`nan` adjoints; guard with a custom `@wp.func_grad` | Kelvin/diffusivity terms need safe grads |
| Structs | `requires_grad` set on an array after storing it in a struct may not propagate; arrays-in-structs are not fully covered by overwrite tracking | `WarpParticleData`/`WarpGasData` grad flags need care |
| fp64 | Differentiable, but fp64 throughput is heavily reduced on consumer CUDA | Precision and speed trade-off, see roadmap |

### Custom gradients and replay functions

Warp lets us override the generated adjoint of a `@wp.func`:

- `@wp.func_grad(fn)` supplies a custom adjoint. Use it to guard non-finite
  gradients, for example only propagating `1/(2*sqrt(x))` when `x > 0`.
- `@wp.func_replay(fn)` supplies a custom forward computation replayed during
  the backward pass. Use it when the forward pass stores per-thread state that
  the backward pass must reproduce (the documented pattern for reversible
  atomic-index increments).
- `@wp.func_native` allows custom C++/CUDA with a paired adjoint snippet.

These are the primary tools for keeping physically-motivated kernels
differentiable without rewriting them as naive arithmetic.

## Randomness and Gradients

Warp's differentiability documentation does **not** define gradients for the
`wp.rand*` family. In practice, random draws are treated as constants in the
backward pass: no gradient flows through `wp.randf`, `wp.randi`, or the
acceptance/rejection branch that consumes them.

Consequences for stochastic aerosol dynamics:

- A discrete accept/reject event (`wp.randf(state) < p`) is a step function in
  `p`. Its gradient with respect to `p` is zero almost everywhere, so
  parameters that only affect `p` receive no useful gradient.
- Integer index sampling (`wp.randi`) for collision-pair selection is
  non-differentiable with respect to any physical input.
- A hard state change gated on a random event (zeroing a merged particle's
  mass) produces gradients only along the branch actually taken, not the
  distribution over outcomes.

This is a general property of Monte Carlo simulation, not a Warp defect. To get
useful gradients through a stochastic process, the estimator must be chosen
deliberately.

## Optimization Target: Initial State, Not Parameters

The planned inverse problem does **not** fit physical process parameters.
Coagulation and condensation parameters (accommodation coefficient, diffusivity,
and similar) are prescribed and fixed. The unknowns are the **initial aerosol
state**, and the loss is on the **final state** after the simulation. Concrete
targets include:

- initial versus final size distribution,
- initial versus final hygroscopicity,
- initial versus final mixing state.

This reframes the differentiability requirement:

- Gradients flow from a final-state loss back to the initial state, **through**
  each process operator. The operators must be differentiable with respect to
  the state they act on (per-bin or per-particle masses, concentrations, and
  composition), not with respect to their parameters.
- Prescribed parameters remove the need for parameter adjoints, which simplifies
  custom gradients. It does **not** remove the need for a state-differentiable
  coagulation operator: coagulation is size-dependent, and size depends on the
  state being optimized, so the stochastic accept/reject and discrete pair
  selection still block the gradient from final state to initial state.
- Mixing state and hygroscopicity are composition-dependent. Coagulation
  redistributes composition when particles of different composition merge, so
  the state representation must carry per-species composition through the
  differentiable operator. This favors composition-resolved sectional bins
  (size by composition) or a differentiable particle-resolved surrogate.

## Techniques for Differentiating Stochastic Processes

Four standard approaches, from most to least aligned with our use case. Here the
gradient of interest is with respect to the initial state (bin or particle
concentrations and composition), since parameters are fixed.

### 1. Deterministic mean-field surrogate (recommended default)

Replace the stochastic process with its expected-value population-balance
formulation for the gradient path, and keep the stochastic particle-resolved
model for high-fidelity forward simulation.

- Coagulation has a well-known deterministic form: the Smoluchowski coagulation
  equation on a binned/sectional or moment representation. It is smooth in the
  kernel and concentrations, so it is directly differentiable.
- Condensation mass transfer is already deterministic and rate-based, so it is
  naturally differentiable.
- The sectional operator maps an initial binned distribution to a final one as a
  smooth function of the bin concentrations, which is exactly the state gradient
  the inverse problem needs. Prescribed kernel parameters are constants in it.
- This gives a clean split: particle-resolved stochastic model for forward
  physics, binned deterministic model for the initial-state gradient path. The
  two must be validated to agree in the mean.
- For mixing-state or hygroscopicity targets, use composition-resolved bins so
  the operator tracks how coagulation mixes composition across sizes.
- Cost: two code paths to maintain and reconcile.

### 2. Reparameterization / pathwise gradients

For continuous random inputs, express the sample as a deterministic function of
parameters plus parameter-free noise, for example
`x = mu + sigma * eps`, `eps ~ N(0, 1)`. Gradients then flow through `mu` and
`sigma`.

- Works for continuous stochastic terms (for example noisy rates), not for
  discrete collision events.
- Low variance when applicable.

### 3. Score-function / REINFORCE estimator

Estimate `d/dθ E[f] = E[f * d/dθ log p(x; θ)]`, where here `θ` is the initial
state (bin or particle concentrations), not a process parameter. Works for
discrete events like accept/reject, because it differentiates the sampling
probability rather than the sampled outcome.

- Unbiased but high variance; typically needs variance reduction (baselines)
  and many samples or many boxes to be usable.
- Fits naturally with a large multi-box ensemble, where each box is a sample.
- Requires the kernel to expose the event log-probability as a function of the
  initial state, not just the boolean outcome.
- Relevant only if a stochastic particle-resolved gradient path is required
  instead of the deterministic sectional surrogate.

### 4. Continuous relaxation (Gumbel-softmax / straight-through)

Replace the hard discrete choice with a smooth, temperature-controlled
relaxation during the backward pass while keeping (optionally) the hard choice
in the forward pass.

- Biased gradients; bias shrinks with temperature but variance grows.
- More intrusive to implement inside a Warp kernel than options 1-3.
- Useful mainly if a per-event differentiable path is required and the
  mean-field surrogate is not acceptable.

### Summary guidance

All rows assume the gradient is with respect to the initial state; process
parameters are fixed.

| Process | Recommended state-gradient path |
| --- | --- |
| Condensation (deterministic) | Direct autodiff, no change needed |
| Gas partitioning / rates | Direct autodiff or reparameterization |
| Coagulation, size distribution target | Deterministic sectional Smoluchowski surrogate |
| Coagulation, mixing-state/hygroscopicity target | Composition-resolved sectional surrogate |
| Coagulation, particle-resolved grads required | Score-function or relaxation (open decision) |

## Particula-Specific Assessment

Nothing in the production code uses gradients today. All GPU arrays are
allocated with the default `requires_grad=False`. The physics building blocks
are already gradient-friendly; the stochastic control flow is not.

### Differentiable-friendly today

- All `@wp.func` physics kernels are pure `wp.float64` arithmetic: the Fuchs
  Brownian kernel, diffusivity, g-term, thermal speed, Kelvin term, diffusion
  coefficient, and mass-transfer rate. These are structurally differentiable
  once wrapped in a `wp.Tape`, subject to non-finite-gradient guards on `sqrt`
  and division.
- The condensation path is the closest template for a differentiable process.
  `condensation_mass_transfer_kernel` computes a smooth rate and
  `apply_mass_transfer_kernel` applies it with `+=`-style updates plus a clamp
  (`particula/gpu/kernels/condensation.py`). This is the model to follow for a
  differentiable coagulation rate kernel.

### Non-differentiable points in coagulation

All three are in `particula/gpu/kernels/coagulation.py`:

1. Discrete accept/reject: `wp.randf(state) < kernel_value / k_max`. Hard 0/1
   event; zero gradient with respect to the kernel value.
2. Integer pair sampling: `wp.randi(state, 0, n_particles)` selects colliding
   particles; non-differentiable with respect to inputs.
3. Hard in-place merge: `apply_coagulation_kernel` folds particle `j` mass into
   `i` and zeroes `j`. This is a write-after-read discontinuity, exactly the
   pattern Warp flags as gradient-breaking.

Additional structural issue: the kernel is one thread per box and loops over a
data-dependent number of trials. Warp does not replay dynamic loops in the
backward pass, so even the smooth arithmetic inside that loop would not
differentiate correctly as written.

### Recommended direction

- Prove the autodiff loop on deterministic condensation first: allocate
  `requires_grad=True` inputs, record a multi-step condensation loop on a tape,
  and validate with `wp.autograd.gradcheck`.
- For coagulation, build a differentiable deterministic binned coagulation
  kernel (Smoluchowski) as the gradient path, distinct from the stochastic
  particle-resolved forward kernel. Validate that the two agree in the mean
  before using gradients.
- Keep optimization-path kernels to static loops, `+=`/`-=` updates, and
  guarded gradients. Avoid `*=`, `/=`, hard index scatter, and data-dependent
  loop bounds on that path.
- Turn on `wp.config.verify_autograd_array_access=True` in tests to catch
  write-after-read overwrites early.
- Define the state loss functions (size-distribution distance, hygroscopicity
  distribution distance, mixing-state metric) and confirm each has a
  differentiable path from the final state back to the initial state. Process
  parameters stay prescribed; do not build parameter-gradient machinery.

## Open Decisions

- The state representation for the gradient path: size-only sectional bins,
  composition-resolved sectional bins (for mixing state and hygroscopicity), or
  particle-resolved with a differentiable surrogate.
- Whether the first optimization milestone targets only the size distribution
  (simplest differentiable path) before adding mixing-state and hygroscopicity
  targets.
- Whether coagulation needs a particle-resolved gradient path at all, or whether
  the deterministic sectional surrogate is sufficient for the intended targets.
- How the deterministic sectional gradient model is kept consistent with the
  particle-resolved stochastic forward model.
- fp64 versus mixed precision on the differentiable path, given fp64 throughput
  limits on consumer CUDA hardware.

## References

The essential Warp patterns are reproduced inline in
[Self-Contained Code Patterns](#self-contained-code-patterns) so this page can be
used without internet access. The external links below are for optional
follow-up only and are not required to implement the guidance here.

- Roadmap: [Data-Oriented Design and GPU Roadmap](data-oriented-gpu.md)
- Warp version referenced: 1.14 (`warp-lang`). Verify against the installed
  version with `python -c "import warp; print(warp.__version__)"`, since autodiff
  behavior can change between releases.
- External (optional, may require web access):
  - Warp Differentiability user guide
    (`nvidia.github.io/warp/stable/user_guide/differentiability.html`).
  - Warp Limitations user guide
    (`nvidia.github.io/warp/stable/user_guide/limitations.html`).
  - Warp autograd API: `gradcheck`, `jacobian`, `jacobian_fd`
    (`nvidia.github.io/warp/stable/api_reference/warp_autograd.html`).
