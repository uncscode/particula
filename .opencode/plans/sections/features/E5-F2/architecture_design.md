# Architecture Design

## High-Level Design

```text
E5-F1 normalized mechanism contract
  -> E5-F2 scalar charged helpers (no sampling side effects)
       radius/mass/charge + environment
       -> friction and reduced pair properties
       -> Coulomb potential with stable neutral/repulsive limits
       -> approved dimensionless model
       -> finite non-negative dimensioned pair rate
  -> E5-F3 consumes pair-rate helpers and proves sampling majorants

coagulation_step_gpu preflight
  -> validate masses/concentration/density/volume/charge on active device
  -> existing one-pass selector writes disjoint accepted pairs
  -> apply_coagulation_kernel once
       recipient.mass[s] += donor.mass[s]; donor.mass[s] = 0
       recipient.charge += donor.charge; donor.charge = 0
       donor.concentration = 0
```

Pair helpers remain scalar `@wp.func` building blocks in the existing dynamics
module so E5-F3 and E5-F6 can sum rates inside one Warp kernel. CPU functions
are independent expected-value references, not runtime dependencies. Numerical
branches must explicitly handle a zero Coulomb potential, clip the repulsive
potential consistently with the approved CPU reference, use safe exponentials,
and return finite non-negative rates for the supported domain.

Accepted pairs are already disjoint within one call because the selector
removes both active ranks. The merge kernel therefore keeps its current
parallel `(box, collision)` launch and extends the same recipient/donor update
with charge. Conservation is assessed per box: sum of every species' masses and
sum of particle charge remain unchanged, while inactive donor fields are zero.

## Data / API / Workflow Changes

- **Data Model:** No schema change. Continue using
  `WarpParticleData.charge: wp.array2d(dtype=wp.float64)` with shape
  `(n_boxes, n_particles)` and dimensionless elementary-charge counts.
- **Pair API:** Add concrete scalar helpers in
  `particula.gpu.dynamics.coagulation_funcs`; names and arguments should mirror
  physical quantities rather than Python strategy objects.
- **Merge API:** Extend the private `apply_coagulation_kernel` argument list with
  the existing charge array. The public step return tuple and output-buffer
  ownership remain unchanged.
- **Validation:** Extend particle-array and device checks with charge shape,
  dtype, device, and finite-domain validation before any launch. All-zero charge
  remains the neutral compatibility path.
- **Workflow Hooks:** E5-F2 depends on E5-F1's configuration/model decision and
  hands pair helpers to E5-F3. It does not register charged execution or alter
  the one-pass sampler itself.

## Security & Compliance

There is no network or permissions surface. Robustness is the compliance
boundary: fail before mutation/RNG advancement, avoid hidden CPU reads and
device transfers, avoid divide-by-zero/overflow/NaN propagation, preserve
fixed-shape caller ownership, and reject unapproved model identifiers. Formula
ports retain citations from their CPU sources. CUDA is optional; Warp CPU is
release-blocking when Warp is installed.
