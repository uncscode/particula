# E3-F2 Open Questions

- Should acceptance diagnostics be implemented as test-only helper kernels or as
  optional output buffers on a private/internal path?
- What exact mixed NPF/droplet radii and particle counts best expose acceptance
  collapse while keeping tests fast and deterministic?
- How many fixed bins are safe within the current one-thread-per-box Warp kernel
  before overhead outweighs improved acceptance?
- What acceptance-rate threshold qualifies as "improved" versus "explicitly
  bounded" for the final decision?
- Should documentation include benchmark-style evidence, or are statistical and
  conservation tests sufficient for this feature?
