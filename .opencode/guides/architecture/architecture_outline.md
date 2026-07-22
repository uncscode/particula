# Architecture Outline

## GPU Package

`particula/gpu/` contains Warp-backed data containers, explicit CPU↔GPU
transfer helpers, device-side physics helpers, and kernel entry points.

### particula/gpu/

**Key Components:**
- `__init__.py` - Public GPU exports
- `conversion.py` - Explicit CPU↔GPU transfer helpers only
- `warp_types.py` - Warp container schemas only
- `dynamics/` - GPU physics helper functions
- `properties/` - GPU property helper functions
- `kernels/` - GPU kernel entry points and private kernel support helpers
- `tests/` - Test coverage

### particula/gpu/kernels/

GPU kernel entry points own launch-time orchestration and may depend on shared
private helpers for cross-kernel setup.

**Key Components:**
- `condensation.py` - Condensation GPU entry points and kernels
- `coagulation.py` - Coagulation GPU entry points and kernels
- `dilution.py` - Concrete P1 GPU dilution input boundary; validation scans may
  allocate or launch, but rejected calls have no update-kernel launch or caller
  mutation
- `environment.py` - Shared private normalization and validation for kernel environment inputs
- `tests/` - Test coverage
