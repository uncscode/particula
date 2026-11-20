# Full resident cyclic trajectory and output recovery

The initial resident campaign ran on an NVIDIA RTX 5060 at revision `9c548d9d617a93cd23a45773426f751e9e6ed9ec`. Later reruns retain their own revisions and observations below.


The first complete resident cyclic run at `9c548d9d61` passed all 600 one-second
cycles on the RTX 5060, with 32 boxes, 8,192 slots per box, 196,608 initially
active particles and twelve prescribed source pulses. It finished with 37,064
active particles. The maximum observed box/species inventory relative residual
was **1.19682042e-13**, below the unchanged `rtol=2e-12` gate. Charge, unit
counts and finite-state checks passed at all thirty observation times.

The 300-second checkpoint reproduced the complete second half with byte-exact
final state and random continuation. The original trajectory used **620,558
accepted** and **378 rejected** condensation steps, with zero domain rejections,
1,200 control readbacks and **zero physical-state coupling uploads/downloads**.
These results are retained in the captured completion stdout; the first attempt
did not publish a result bundle. Its subsequent plot-data writer rejected NumPy
arrays, and kernel cleanup released the in-memory observations. No missing
trajectory data have been reconstructed or substituted.

The observation boundary now converts detached arrays to plain lists. This
changes output representation only. A regression using the actual observation,
JSON writer and CPU-only plotting path reproduces the failure before the fix
and passes after it. Together with the publisher tests, **24 tests passed with
warnings treated as errors**. The timeout is now 14,400 seconds to accommodate
the measured full workload with enforced deadlines. The fresh full publication
from `3d3753d2b` completed in **6,307.33 s**, using identical library bytes and
scientific settings. It reproduced all work counts and accounting results above,
including exact same-device restart and zero stage transfers. The retained
**659,430-byte** data bundle contains all **31 frames** from 0 through 600 s.
The four-panel figure was inspected before and after CPU-only replot. Replot
preserved the data hash and original CUDA execution provenance; no simulation
was rerun for rendering. The initial displayed point precedes the first forcing
update, and the 20-second observations do not resolve every jump or peak. This
completion time is not an isolated performance benchmark.
