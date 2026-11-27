# Boreal parcel checkpoint challenge

For the published plots alone, use the [compact CSV bundle and replot command](../README.md). The full runs below produce the additional trial tables and checkpoints locally.

The study runs locally on CPU using the checkout's Particula environment, NumPy, SciPy, pandas, Matplotlib, a C++17 compiler and scattnlay 2.4. Run from the repository root. The recorded environment used Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2 and GCC 16.2.1. The optical dependency is isolated in a project-specific directory. Allow approximately 50 GiB for working reference checkpoints. Lowering the worker count reduces peak memory use.

```bash
export COATED_ENV="$PWD/.artifacts/boreal/deps"
python -m pip install --no-deps --target "$COATED_ENV" scattnlay==2.4
WORKERS=4 bash studies/parcel-composition/boreal/run.sh
```

Use a new output directory for a new numerical experiment. The runner writes the chapter's compact evidence beside its source and caches full trajectories under the directory given as its first argument. Each campaign marker records original particles per parcel, step in seconds and stochastic repeat. The seed order in `model.py` fixes the calibration split. The separate four-species numerical control is implemented in the sibling `aging` directory.

## Replay the saved cloudy restart

```bash
export PYTHONPATH=".:$COATED_ENV" OPENBLAS_NUM_THREADS=1
export BOREAL_TREE="$PWD/.artifacts/boreal/tree.so"
study=studies/parcel-composition/boreal
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
python "$study/checkpoint.py" "$study/cloudy-restart.npz"
```

The NPZ contains both parcels at hour three of history 1562, plus the generator state, forcing, thermodynamics, source gases and lineage. Each representative stores eight dry species volumes and liquid-water volume as nine float64s. Volumes are expressed as six times physical volume divided by π, in nm³. Rows additionally store the dry bin and exact integer count as two int32s. A uint16 member-to-row index restores the original slot order for common-random-stream comparisons, a bit vector records pre-existing membership, and each slot holds its unresolved cohort multiplicity as int32. Archive metadata and compression add a file-level cost. The encoder checks the decoded state against direct in-memory reconstruction field by field.

## Read the results

- `caps.csv` gives the calibration-selected snapshot cap for every grid and grouping. The complete integer sweep appears in `cap-sweep.csv.gz` as worst error at each cap. Grid 0 is the earlier 12-bin grid. Rules 0, 1 and 2 select generic splitting, BC first, and BC followed by new-cohort splitting. `snapshot-scores.csv.gz` contains individual event and hourly errors at 16 and the selected cap.
- `restart-scores.csv.gz` gives every evaluated future score with initialization time, evaluation time, seed, parcel and stochastic run. The peak and cohort gates apply over the entire remaining interval through hour six. `screen-*.csv` contains an independent necessary-condition test for every integer from 1 through 256 at a single calibration checkpoint. `water-control.csv` separates averaging water and averaging dry species.
- `shape-scores.csv.gz` reports whole-profile and three decade errors and the location of the largest residual. `resolved-profiles.csv.gz` gives held-out occupancy and absorption on the 16-bin-per-decade grid, including explicit underflow (bin 0), overflow (49) and blank undefined χ. The float64 heights in `held-out-chi.npz` directly reproduce the adaptive fits. Interpolate linearly at log10(dry diameter / 1.7 nm), clip to [0, 1] and use the packed defined-bin mask. Keys identify seed, zero-based regime index, seconds and parcel. The first interior bin is [1.7, 1.7 × 10^(1/16)) nm.
- `particle-allocation.csv` and `knot-allocation.csv` contain the calibration search. `storage.csv.gz` gives the actual occupied rows, snapshot record bytes, original count and fraction of BC mass in each held-out decade. The independent forward test of those same budgets is in `allocated-restarts.csv.gz`. The common count spectrum, identity of a frozen grid, and global metadata are side information for χ-only fits.
- `checkpoint-bytes.csv` measures the serialized particle buffers in the supplied restart by dry decade, including slot indices and lineage. `forecast-*.csv` gives each calibration rejection before the first passing clear-air forecast cap. `limiting-cases.csv` repeats the exact held-out failure boundaries and `knot-boundary.csv` refines the near-threshold movable-knot case.
- `optics.csv.gz` separates integrated 550 nm absorption for dry coated spheres, wet coated spheres and dry homogeneous spheres. `reference-checks.csv` gives repeat, population and time-step variations. `cloud-error.csv` contains the plotted droplet errors by size and time.

## Reproduce stage two

With the same isolated scattnlay 2.4 directory and Python environment, run
`WORKERS=4 bash studies/parcel-composition/boreal/run-stage-two.sh` from the
project root. Its optional argument sets the working output directory. It can
build its own optical table. The earlier campaign under `base/` is needed only
to regenerate the retrospective burst-peak audit. Use a fresh working directory
after changing the physical model or grouping rule to avoid reusing old scores.

`stage-two-forcing.csv` records the full-resolution response of each physical
history. The factor codes −1, 0 and +1 multiply sulfuric acid supply by 0.85,
1 or 1.15, exchange by 0.65, 1 or 1.35 and ascent height by 0.80, 1 or 1.20.
`stage-two-histories.csv` reports each rule’s worst error within a history.
`stage-two-scores.csv.gz` retains each paired comparison. Start and end times
are seconds. Equal times indicate snapshot reconstruction. Peak water and
droplet numbers are for the ascending parcel, cumulative from restart to the
specified end, and are repeated on the two parcel rows. Births and survival
are likewise repeated and refer to both parcels combined. Birth counts cover
only the remaining interval. Survival includes all lineages born by the
evaluation time. Zero reference droplet peaks are scored in cm⁻³ in the history
table. A blank relative error indicates an inapplicable denominator.

`stage-two-profiles.csv.gz` contains the 48 interior dry-size bins, underflow
and overflow for each reference parcel and checkpoint. It records dry χ,
particle count, liquid-water mass and count in the growing branch, water in
haze, BC volume (in d³ units), new-only particle count, pending birth lineages
and conditional 10th, 50th and 90th diameter percentiles in nm. Water masses
are kg per simulated parcel. The volume of each parcel is its initial particle
count divided by 1,000 in cm³. Undefined χ and empty cohort quantiles are
blank. `stage-two-controls.csv` compares each numerical refinement with the
nominal full-resolution history. `stage-two-burst-audit.csv` revisits the
four earlier held-out histories, including raw and tie-stabilized CDF error.

## Reproduce the six BC absorption failures and local refinements

Run stage two first. From the same environment and repository root use
`WORKERS=4 bash studies/parcel-composition/boreal/run-coating.sh`. The optional
working-root argument defaults to `.artifacts/boreal`. Remove no prior
evidence: choose a fresh working root for changed physics or grouping rules.

`coating-peaks.csv` identifies the worst original observation in each failed
history. `coating-cells.csv.gz` includes the starting and failing distributions
for all ten misses. `failure_end` distinguishes a start shared by several
failures. The dry-bin index uses the original ten-bin-per-decade grid with
underflow at zero. Core-diameter edges in nm are 50, 80, 120, 180 and 250.
Outer/core diameter ratio edges are 1.25, 1.5, 2 and 3. Both axes use category
zero below the first edge and include a final overflow category. Each BC cell
reports count, BC d³-equivalent volume in nm³, summed absorption cross section
in nm² and summed radial shell thickness in nm for reference and prediction.
Divide the latter by the corresponding count to obtain the mean thickness.
All signed optical contributions use the full reference parcel as denominator.

`coating-targets.csv` gives the six dry-size and coating classes selected on
seed 2039. `coating-inventory.csv` measures the fraction of target-*bin* BC
misplaced between coating categories at restart. `coating-restarts.csv.gz`
contains all deterministic-budget comparisons, including snapshots, with
signed future errors and both optical-additivity terms. `coating-histories.csv`
reduces these to worst future errors per history and rule. Gas errors use
total reference gas plus particulate inventory of the same organic class as
denominator. `coating-repeats.csv.gz` contains same-stream reduced versus
full comparisons on three alternate stochastic streams, and the difference
between that uncompressed stream and the original full reference. The extra_rows column distinguishes the four- and
eight-row paired runs. Seed,
perturbation factors, time step, population, start, end and parcel identify a
comparison. At a start equal to the end, cloud peak differences are undefined.

## Qualify a random redraw

From the repository root, with the development Python and pinned scattnlay
environment described above, run
`WORKERS=8 bash studies/parcel-composition/boreal/run-redraw.sh`. The runner
builds its tree and optical table as needed. Use a fresh working-root argument
when changing physics, scoring, design or sampling. Completed trajectories are
cached under `redraw-qualification/` within that root. Calibration and test
seeds are disjoint. The selection JSON is written before the held-out run.
Four smaller representations are reported as comparisons with the frozen rule.

To regenerate a population from the default 16-row checkpoint, run
`python studies/parcel-composition/boreal/redraw_replay.py --draw 0`
with the same PYTHONPATH, BOREAL_TREE and optical table used by the runner.
The default archive is `redraw-cloud-16.npz`. Pass `redraw-cloud.npz` by path
to replay the historical 32-row calibration choice. Both contain the hour-three
state of seed 8329 at 10,000 initial particles per parcel. Draws 0–9 reproduce
the study streams. The decoder builds exact integer counts, balanced random
composition deviations and a fresh slot ordering without the original arrays.

The archive stores float64 dry species and water in d³-equivalent nm³. Multiply
by π/6 for physical volume. The columns in `label` are dry bin, growing branch,
pre-existing ancestry, pending birth multiplicity and presence of BC, in order.
The integer `count` array and the `mean` and optional `delta` arrays use the same
row ordering. The header includes forcing, case metadata, checkpoint time and
the original random state. The dry bin edges are defined in `metrics.py`.
The model's rising parcel is A. Both parcels have volume n/1000 cm³.

`redraw-calibration.csv` compares the six candidates and full control. A recipe
name gives bins/decade, base row cap and whether a direction vector is stored.
`redraw-storage.csv` gives the held-out row and realized particle count in each
decade and the uncompressed row bytes. Decades −1 and 3 mark underflow and
overflow. `redraw-trajectories.csv.gz` gives all frozen-rule and full-repeat
realizations, reference values, joint component errors and gates at each lead.
Absorption is in Mm⁻¹ and organic transfer is integrated over both parcels in
d³-equivalent nm³. Infinite relative errors mark an event against a zero
reference. `redraw-reliability.csv` and `redraw-events.csv` summarize single
draws, and `redraw-paired.csv` compares like downstream streams. The comparative
table gives per-size success and actual file bytes for cheaper budgets.
`redraw-ensembles.csv.gz` scores mean observables, including the equally weighted
CCN mixture curves. `redraw-step-check.csv` reports the fixed-lead sensitivity
at 25 s for one previously specified held-out history at both particle counts.

## Reproduce the independent qualification and memory scaling

With the same activated environment and `COATED_ENV`, run the following from the checkout root.

```bash
WORKERS=6 bash studies/parcel-composition/boreal/run-redraw-scaling.sh
```

The optional first argument selects a working directory. Use a fresh directory after changing the physics, sampling rules or seeds. The script builds the tree splitter, prepares the optical table, evolves the resolved histories, runs the two checkpoint recipes and full-resolution controls, measures memory, then generates the tables and figures in the qualification chapter. The memory measurement uses Linux procfs and glibc. Forecast workers can run in parallel. The memory sweep uses one process and one BLAS thread.

- `redraw-independent.csv` gives joint pass counts, history-level uncertainty and the four family pass rates by forecast horizon. `redraw-independent-events.csv` gives rates per event, with 40 trajectories per cell.
- `redraw-independent-leads.csv.gz` retains the physical observables, original reference, endpoint errors and paired-stream gates for each draw and lead. The method ID gives bins per decade, row cap and spread-vector flag.
- `redraw-memory.csv` records the initial count per parcel (`n`), checkpoint time in seconds (`start`), realized two-parcel particle count (`slots`), rows, particle-array and portable archive bytes. Recipe 0 stores the original particles. Recipes 16 and 32 use those row caps. The encode/decode columns give wall time in seconds, baseline resident bytes and peak resident bytes. Subtract baseline from peak for incremental working memory. The archive includes nonparticle fields and metadata. The array columns count only particle fields.

The burst replication evolves eight additional seeds to burst onset and repeats the 30-minute course there. `redraw-burst-replication.csv` reports scores against the original continuation and the same-stream full repeat. `redraw-burst-leads.csv.gz` contains the individual runs. The scaling runner also writes `redraw-failure-matrix.csv` with the count of failed 16-row forecasts and the median and worst absolute errors among failures for each property, start time and population. Each forecast contributes its maximum error over the four positive leads. The companion figure displays the six properties with observed failures.

To replay the limiting 50,000-particle burst checkpoint, run this from the checkout root with the environment above. The results stage replays draw zero directly from the exported file and checks every reported physical observable against the original experiment output at all five leads.

```bash
study=studies/parcel-composition/boreal
python "$study/redraw_replay.py" "$study/redraw-burst-16.npz" --draw 0
```

## Reproduce the first 75 seconds of burst formation

From the checkout root, use the same Particula development Python and native tree splitter as in the scaling experiment. This follow-up rebuilds the onset states from seeds. Its only external input table for validation is the earlier replication's first-step count in `redraw-burst-leads.csv.gz`.

```bash
study=studies/parcel-composition/boreal
root=.artifacts/boreal/burst-counts
export PYTHONPATH=".:$COATED_ENV" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export BOREAL_TREE="$PWD/.artifacts/boreal/tree.so"
mkdir -p "$root"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
python "$study/burst_counts.py" "$root" --workers 6
python "$study/burst_counts.py" "$root/five-second" --five-second --workers 6
python "$study/burst_substeps.py" "$study/burst-substeps.csv.gz" --workers 5
python "$study/burst_count_results.py" "$root" "$study" docs/images/parcel-burst-counts.png
```

`burst-count-draws.csv.gz` reports new births and integrated intensities by parcel. `full` and `redraw` start with the same event stream. `independent` uses a second full-resolution stream. `original` replays the resolved history's saved stream. `same_rng` compares complete generator states just before an active parcel's formation draw on the 75 s grid. Empty values indicate an inactive parcel or a subdivided run. `burst-count-histories.csv` summarizes the two parcels combined. Its `expected` field gives the integrated full-reference rate and `full_mean` is the empirical average count. `burst-count-cdf.csv` gives discrete conditional and mixture CDF distances. The 75 s reference is the model law, calibrated with independent simulated ensembles. The 5 s reference is the second full-resolution ensemble, calibrated by permuting method labels within each history. `burst-substeps.csv.gz` gives the intrawindow traces from the numerical sensitivity experiment. All counts refer to the two 50 cm³ parcels when summed. Each worker uses a distinct seed or time-step task and a single BLAS thread.

## Qualify the default checkpoint separately for each air mass

With the environment and tree splitter from the preceding section, run
`python studies/parcel-composition/boreal/burst_qualification.py studies/parcel-composition/boreal --workers 8 --figure docs/images/parcel-burst-conditional.png`.
The test rebuilds the 16 onset states, fits the default 16-row recipe, decodes
each archive, and uses fresh streams. It measures total 1.7 nm births across
the two parcels over fifteen 5 s steps. Each history receives 192 paired runs.
A history whose upper bound exceeds 0.05 receives 384 additional pairs.
The two-look, 16-history adjustment gives simultaneous one-sided 95% coverage.
The criterion and its scope are explained in the qualification chapter.

`burst-conditional-pairs.csv.gz` contains the two parcel counts in each arm of
every trial. `burst-conditional-qualification.csv` gives the result, empirical
quantiles, largest observed CDF gap and upper bound for each history. Means and
quantiles use the final sample for each history. The probability bound is valid
at either of the two planned sample sizes. The two control columns compare the
first 192 coupled full counts with the 32 independent, standard-stream
full-resolution counts for the same history in `burst-count-draws.csv.gz`.
The tied-count permutation check uses 1,999 random label permutations when
the observed CDF distance exceeds zero.
Replace `--workers 8` with `--from-pairs` to regenerate the table and figure
from the stored draws and check the stopping decision for each history.


## Qualify additional measurements from the same 16-row checkpoint

With the development environment active and `COATED_ENV` pointing to the pinned
scattnlay 2.4 install, run `bash studies/parcel-composition/boreal/run-responses.sh`
from the checkout root. Set `WORKERS` for available CPU cores and memory. The
optional first argument selects a cache directory. Use a fresh directory when
changing the model, design or checkpoint recipe. Each air mass has two 50 cm³
parcels with 50,000 initial particles each. The runner regenerates three
checkpoints from the air-mass seed, fits and serializes the default checkpoint,
then decodes it before drawing new populations.

* `response-qualification.csv` gives the joint and component results by air
  mass. The unit of replication is one paired trial evaluated at both short
  restart times. The half-hour ascent and independent full repeats are
  descriptive controls.
* `response-pairs.csv.gz` retains each paired error, selected measurements
  and matching full-resolution values. A family score of one corresponds to
  its physical tolerance. Error suffixes `pct` and `pp` mean relative percent
  and percentage points. Optical coefficients are in Mm⁻¹ at 550 nm.
* `response-distributions.csv.gz` reports conditional full and redrawn
  quantiles and the empirical, unshifted CDF distance for each measured
  quantity, checkpoint and lead. The paired trial table preserves dependence
  between observables.
* `response-scattering-check.csv` compares lookup and direct coated-sphere
  scattering on identical weighted samples stratified by size and BC presence
  in both resolved and redrawn snapshots.
* `response-step-check.csv` repeats 12 pairs at both short windows for one
  history in each forcing combination using 25 s and 75 s integration steps.
  The saved starting state is common to both time grids.

To rebuild the figure, confidence bounds and distribution summaries from the
locally regenerated trials, pass `studies/parcel-composition/boreal/response-pairs.csv.gz`
as the first argument to `response_results.py`, followed by an output directory
and a PNG path. The published run uses a 75 s step, including during ascent.
The separate burst-onset qualification uses its own 5 s grid.

## Moving the dry-size boundaries

Run `bash studies/parcel-composition/boreal/run-adaptive.sh` from the checkout
root with the development environment active and `COATED_ENV` pointing to
scattnlay 2.4. The optional first argument selects the cache root and `WORKERS`
limits CPU workers. The script can regenerate missing full-resolution states
and optical tables. It uses the locally generated `response-pairs.csv.gz` from the
preceding experiment to recover the fixed-grid scores and full counter counts
on exactly the same downstream random streams. Use a fresh cache if you
change the design or solver.

`adaptive_grid.py` defines the frozen density and cutoff rules.
`adaptive_trials.py` records new paired counter measurements and a smaller
set of six-process diagnostics. `adaptive_spectra.py` integrates each fitted
row to obtain its expected count in independent instrument channels.
`adaptive_results.py` checks design completeness, calculates per-air-mass
binomial bounds and generates the comparison figure. To reproduce the plot
and summary tables from the locally retained simulation outputs, run
`python studies/parcel-composition/boreal/adaptive_results.py studies/parcel-composition/boreal docs/images/parcel-adaptive-grid.png`.
The comparison uses a 75 s model step.

* `adaptive-edges.csv.gz` gives all 31 outer and inner boundaries (nm) for
  each method, parcel and snapshot, including row counts and whole-archive bytes.
* `adaptive-pairs.csv.gz` reports raw simulated counts in each 50 cm³ parcel,
  the worst percentage error between the two parcels, and process-family
  errors divided by their tolerances. Missing family scores mark trials
  designated only for the target counter.
* `adaptive-qualification.csv` scores the counter over both short windows
  simultaneously. `adaptive-spectra.csv.gz` gives channel limits, reference
  counts and expected redraw counts at each snapshot. The compact
  `adaptive-channel-summary.csv` summarizes sufficiently sampled channels
  outside 150–160 nm, weighting each parcel/time/channel equally.
* `adaptive-steps.csv` repeats the first four seeds, both short windows and
  four draws per window with one 75 s or three 25 s steps from identical
  saved states. `redraw-adaptive-pinned.npz` is the portable density-plus-
  cutoffs archive for air mass 10111 at hour one. Pass it to
  `redraw_replay.py` to generate particles and run a half-hour forecast.
