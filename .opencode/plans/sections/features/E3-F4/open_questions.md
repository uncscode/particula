# Open Questions

1. Should the final public quick-start import from `particula.gpu.kernels` only,
   or should `particula.gpu` add narrow top-level re-exports for the two step
   functions?
2. If top-level re-exports are added, should raw lower-level symbols such as
   `apply_coagulation_kernel` remain excluded from `particula.gpu.__all__`?
3. Should the quick-start live beside existing data-container examples, or in a
   new GPU direct-kernels example path?
4. After `E3-F1`, what is the minimum coagulation snippet that clearly shows
   persisted `rng_states` without overcomplicating the quick-start?
5. Should troubleshooting live in the example, the feature docs, or both?

Default recommendations: keep the documented low-level import path under
`particula.gpu.kernels`, exclude raw kernel internals from broad public docs,
and place troubleshooting in both the runnable example comments and the
feature documentation.
