# Differentiating a fixed Manchester reaction network

The bulk calibration has two errors per experiment. One is the root-mean-square suspended organic mass residual divided by the measured peak, with a 1 µg m⁻³ floor. The other is the root-mean-square count residual divided by the initial particle number. The optimization minimizes an upper bound on all 30 errors. This permits the run or observable with the largest error to change during a fit.

## What the derivatives include

Each experiment retains its measured precursor-disappearance history. The dose scale multiplies that run's precursor-consumption increments and changes its isoprene-dependent product diversion. Eleven common parameters control the parent-specific yields and formation times, isoprene competition and eddy deposition. The selected reaction rates are also common to the campaign. The derivative passes through primary and secondary vapor supply, product maturation, reachable first- and second-order transfers, the half-retained fragmentation convention, suspended organic mass, the monodisperse equivalent radius, size-dependent deposition, Brownian coagulation, count, and interpolation to each observation time.

The saved parameter vector `x` uses the original archive ordering. Entries 0–2 give α-pinene, o-cresol and isoprene formation times in hours. Entries 3–5 and 6–8 give their two sets of effective source yields. Entry 9 controls wall eddy deposition and entry 12 controls isoprene-dependent diversion. Other slots in the 16-entry prefix retain their archived values. Rates begin at entry 16. The optimizer uses the full, locally generated candidate bank. In the published `size_inputs.json`, the rates follow the ordered `channels` array of each frozen calibration. The run-specific dose vector follows runs 1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19, 21 and 24.

The CUDA replay propagates double-precision dual numbers, one sensitivity direction per lane. Each channel applies the archived effective-mass kinetic law. Area-dependent rates (h⁻¹) refer to 0.001 m² of particles per m³ of air. Second-order rates use m³ of air per µmol of equivalent reactant per hour. A zero rate has a feasible right derivative, including new chemistry whose parent had no such transfer. At an exactly vanishing residual norm the optimization uses its zero subgradient. The measured precursor-history shape, gas transport properties, chamber conditions and discrete network choices stay fixed. The resolved size-distribution replay and terminal mass diagnostics are evaluated separately by the original physical simulators.

## Independent checks and matched controls

The original Warp simulator supplies independent forward evaluations for every optimizer candidate promoted for comparison. The compiled derivative implementation reproduces native mass and count scores. Finite differences vary each shortlisted rate positively from zero and in both directions at positive rates. Additional checks vary the precursor dose and representative shared parameters at three increments. A simultaneous-direction check varies all active rates, all free shared parameters and all local doses at an interior point.

The common fit starts have identical shared coordinates across parent and child, and each has its own local dose parameter per run in the published ±15% bounds. The coarse fit uses the archive's 15 s forward split. The finer fit minimizes the maximum over both split orders at 3.75 s. A second start with connected candidate rates above zero allows downstream products to form. Each saved start includes its objective-call count and stopping condition. Scores at 1.875 s retain fitted parameters and doses. Separate profiles evaluate the archive's grid of 61 possible doses for each run at each resolution and order. For admission, compare parent and child on the training score and on each of the two independent fine-order scores. Use the smallest gain across those three comparisons, retaining the archived incumbent and any previously tested finer parent as eligible controls. The gain must exceed the parent restart spread, the child restart spread and the maximum shift from training to a refined score in either fitted network. Scores in the JSON files are fractions. A difference of 0.0001 is 0.01 percentage points.

## Reproduction

Use the study worktree and a CUDA-enabled Particula Python environment with Warp, SciPy, NumPy and the archive's pandas dependency. The CUDA Toolkit compiler `nvcc` must be on the path. The Python driver compiles the local CUDA source into a private temporary directory once per process. The recorded runs used an RTX 5060, Python 3.12.12, Warp 1.11.1, NumPy 2.3.4, SciPy 1.16.2 and CUDA Toolkit 13.4 for the forward-mode kernel.

From this directory, run the sensitivity qualification and coarse refits after the search has written `shortlist_01.json`.

```bash
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --extra 22,32,64,65 --calls 100
```

Use `--checks-only` to reproduce only the derivative, native parity and timing checks. Results retain all 15 experiments, finite-difference perturbations and signs, per-start vectors and per-run doses. The extra channel IDs refer to the separately screened four-transfer α-pinene uptake-product sequence.

For a gain smaller than the restart spread, a longer matched continuation starts both topologies at the best fine parent and at the best fine child projected into the relevant topology. Each common parameter and dose starts at exactly the donor value. In the reduced parent the newly added coefficient is set to zero.

The finer comparison uses the area-dependent vapor return and the α-pinene vapor self-accretion finalist. The deletion challenge removes the best refined addition at its fitted shared coordinates and doses, then gives its parent another fit from that projected point. The labels in the decision record identify each time step and reaction order.

```bash
OPENBLAS_NUM_THREADS=1 python ad_fine.py --sub 16 --calls 180 --extra 76 --extra 80
OPENBLAS_NUM_THREADS=1 python ad_fine.py --sub 16 --calls 300 --extra 76 --continue-pair
python ad_refine.py ad_cont_76_01.json
python ad_decisions.py
OPENBLAS_NUM_THREADS=1 python ad_ablate.py
```

At the time of the 3.75 s continuation, the 39-channel fit cleared the available controls by 0.003007 percentage points against a 0.002504-point numerical threshold. Replaying the subsequently optimized 38-channel deletion parent removes that early advantage. The first addition ultimately qualifies through the separately refitted 1.875 s comparison below. The marginal pair therefore received a second matched comparison at 1.875 s, with validation at 0.9375 s and a further order-by-order check at 0.46875 s. Both topologies receive the best preceding child and the strongest preceding simpler fit as projected starts. The parent and child each have a cap of 240 objective/Jacobian calls per start. Recompute the decision record after this comparison.

```bash
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --extra 22,32,64,65 --checks-only --check-sub 32
OPENBLAS_NUM_THREADS=1 python ad_adjudicate.py --round 1 --calls 240
python ad_decisions.py
OPENBLAS_NUM_THREADS=1 python ad_ablate.py --adjudicated
python ad_decisions.py
```

Follow the [reproduction sequence](README.md#compare-starts-time-steps-and-competing-reactions) to generate the provisional seed locally. Round two reads the exact x and per-run doses in `growth_seed_02.json` and starts from the 39-channel state that passed the first matched continuation. Its simultaneous-dose assignment is preserved exactly. A second start perturbs the shared precursor formation times by 1% with dose fixed and gives added rates a positive feasible start. Every challenger receives the same two starts and call budget. The finer round-one adjudication is a separate challenge to the lineage used here.

```bash
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --round 2 --checks-only
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --round 2 --checks-only --check-sub 16
OPENBLAS_NUM_THREADS=1 python ad_fine.py --round 2 --sub 16 --calls 100 \
  --extra 71 --extra 100 --extra 38 --extra 66,67 --extra 82 \
  --extra 88 --extra 81 --extra 79 --extra 91
python ad_decisions.py --round 2
```

Of the nine second-round fits, channel 100 had the largest positive minimum gain against its simultaneously fitted parent at 3.75 s and 1.875 s. The independently refined first-round parent reduced the score further. To test channel 100 directly against that parent, refit both topologies at 1.875 s, using the better incoming parent at that resolution and the best 40-channel fit as common starts. Freeze the two fitted solutions for evaluation at 0.9375 s and 0.46875 s.

```bash
OPENBLAS_NUM_THREADS=1 python ad_adjudicate.py --round 2 --extra 100 --calls 100
python ad_decisions.py --round 2
OPENBLAS_NUM_THREADS=1 python ad_ablate.py --round 2 --adjudicated
python ad_decisions.py --round 2
```

The third round uses the four entries in `shortlist_03.json`, the frozen 40-channel parent and a common second start with its precursor formation times shifted by 1%. The cap is 40 calls per start in each topology at 1.875 s. Independent replays at 0.9375 s use both channel orders, fixed continuous doses and additional 61-dose profiles.

```bash
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --round 3 --checks-only
OPENBLAS_NUM_THREADS=1 python ad_shortlist.py --round 3 --checks-only --check-sub 32
OPENBLAS_NUM_THREADS=1 python ad_fine.py --round 3 --sub 32 --verify-sub 64 \
  --calls 40 --extra 66,67 --extra 38 --extra 82 --extra 79
python ad_decisions.py --round 3
```

## Derivatives at individual observations

The frozen export differentiates each predicted mass (µg m⁻³) and count (particles cm⁻³) with respect to that experiment's dose multiplier and each admitted rate. The CUDA kernel retains minute-endpoint tangents. The exporter applies the same linear interpolation as the measurement residual and writes one row per instrument time in `ad_observation_jacobian.csv.gz`. Independent native trajectory differences at two perturbation sizes check the mass and count sensitivities over every minute of all five hours. At an active parameter bound the finite difference uses the feasible side. The accompanying JSON records the actual selected vector, doses and normalization used to assess the numerical checks.

The third-round decision retained 40 channels. Its two added area-dependent rates are 0.290405 h⁻¹ for α-pinene vapor return and 0.262512 h⁻¹ for uptake of secondary o-cresol product vapor at the reference area. The export checks every active rate, dose and channel ID against the inputs used for the size and trajectory comparisons. Its JSON contains SHA-256 digests for that frozen input file and the compressed Jacobian table. The exported per-observation sensitivities are also combined analytically into the 30 normalized RMSE derivatives and checked against the directly differentiated campaign scores.

```bash
OPENBLAS_NUM_THREADS=1 python ad_trajectories.py --round 3 --sub 64
```
