# Record and infer aerosol properties from optics

Dense optical histories retain the quantities seen by an optical instrument.
They can follow a changing population, test a physical model against spectra,
and support retrievals with a much smaller output than a particle history.
Use the worked examples to check each of those steps. The numerical
scattering model is Lorenz–Mie theory for homogeneous spheres.

- [Record extinction at 10 Hz](optical-recording/dense_optical_history.py).
  Compare continuous optical recording with sparse offline checks during
  hygroscopic growth.
- [Retrieve effective index and growth](optical-retrieval/infer_refractive_index.py).
  The synthetic κ interval narrows from 0.15–0.43 with one humidified RH to
  0.26–0.30 with dry and two humidified conditions at 2% prescribed errors.
  The Suchdol example retains 700 nm for a field prediction. The November
  fit overshoots that channel by 12–17%, while the August fit undershoots
  by up to 7%. These results depend on the assumed size calibration,
  homogeneous spheres and one effective index across the visible band.
- [Ask whether the optics resolve mixing state](optical-mixing-state/infer_mixing_state.py).
  Two populations can share dry sizes and each species inventory while
  distributing the components differently among particles. The example
  fits a constrained family and uses measurements on selected sizes to
  test whether both size-band fractions can be resolved.
- [Fit initial radius from the optical trajectory](../condensation/optical-trajectory/infer_optical_trajectory.py).
  The fixed-step closed-box inverse retains physical radius and vapor
  for an independent check and compares CUDA autodiff with finite
  differences of the optical loss.

The [optical guide](../../../guide/optics/index.md) defines the optical
quantities and recording interfaces. The [reference corpus](sphere-optics/index.md)
and [CUDA qualification](cuda-optics/index.md) establish forward accuracy
for their respective numerical domains. Each retrieval example reports
its own sensitivity or profile and held-out prediction.
