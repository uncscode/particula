# Documentation Updates

## P6 Shipped Documentation

P6 documents the concrete-only selected-Brownian coagulation adapter without
adding a public export or altering the direct-kernel contract. The strategy
guide now distinguishes the CPU reference route from resident-Warp
particle-resolved dispatch; it specifies exact marker validation, both Warp
thermo forms, optional volume, caller-owned diagnostics and persistent RNG,
and the no-transfer/no-fallback/no-rollback boundary.

The explicit-transfer example constructs the selected Warp state and execution
carrier for two adapter dispatches. It allocates the direct kernel's effective
default pair capacity, initializes its caller-owned RNG sidecar once, reuses it,
synchronizes before restoration, and contains no direct-kernel mechanism or
collision-capacity arguments.

Hardware-free regression coverage verifies the guide boundary, imports,
deferrals, links, and forced-no-Warp example path. Focused validation passed:

```bash
pytest particula/tests/backend_selected_coagulation_docs_test.py -q -Werror
pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q -Werror
pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror
mkdocs build --strict
```

E7-F4 resident sessions, E7-F5 scheduling, E7-F8 checkpoint/restart stream
policy, public exports, fallback, and changes to the multi-mechanism direct
kernel remain deferred. This completed P6 publication evidence contributes to
the authoritative shipped/completed E7-F3 feature record without shipping those
separate deferred scopes.
