#!/usr/bin/env bash
# Run stages from the study root. Pass "all" or one or more stage names in order.
set -euo pipefail
cd "$(dirname "$0")"
py=${PYTHON:-python}
fit_py=${FIT_PYTHON:-$py}
workers=${WORKERS:-4}
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

inputs() {
    "$py" prepare.py
    "$py" rate_sources/extract_reference_figures.py
    "$py" companion/link_neutral_records.py
    "$py" companion/extract_run1833.py
    "$py" analysis/reference.py
    "$py" analysis/benchmarks.py
}
fits() {
    # The second pass extends promising starts that exhausted the first budget.
    "$fit_py" analysis/fit.py --stage full --workers "$workers"
    "$fit_py" analysis/fit.py --stage full --workers "$workers"
    for step in joint_fit fraction_profile design paired_scan; do
        "$fit_py" "analysis/$step.py"
    done
}
nominal() {
    "$py" analysis/chemistry.py
    "$py" analysis/dimers.py
    "$py" analysis/particles.py --tracks 2048 --reps 4
    "$py" analysis/particles.py --attraction --tracks 2048 --reps 4
    for step in population warm_chamber summarize figures; do
        "$py" "analysis/$step.py"
    done
}
sulfur() {
    for step in search sensitivity plausibility bulk neutralization figure; do
        "$py" "simba/$step.py"
    done
    for step in balance sulfur_inputs experiment sectional refinement equal_gas wall_photo figure; do
        "$py" "simba/wall-sulfur/$step.py"
    done
}
formation() {
    "$py" monte-carlo/design.py
    "$py" monte-carlo/archive.py
    for phase in primary extension ions; do
        "$py" monte-carlo/formation.py "$phase" --workers "$workers"
    done
    for step in formation_profile prospective formation_summary overview; do
        "$py" "monte-carlo/$step.py"
    done
}
clusters() { "$py" monte-carlo/cluster.py; }
transport() {
    "$py" monte-carlo/transport.py --oracle
    "$py" monte-carlo/transport.py
    "$py" monte-carlo/transport_population.py
    "$py" monte-carlo/transport_results.py
}
warm() {
    "$py" monte-carlo/warm.py --workers "$workers"
    "$py" monte-carlo/warm_led.py
    "$py" monte-carlo/warm_summary.py
}
plots() {
    "$py" monte-carlo/review_comparisons.py
    "$py" analysis/export_plot_summaries.py results/plot-data
    renderer=../../website/public/study-data/dunne-cloud/replot.py
    cp "$renderer" results/plot-data/replot.py
    "$py" results/plot-data/replot.py --output results/figures
}
if [[ "$#" == 0 ]]; then
    echo "Usage: bash run.sh {all|inputs|fits|nominal|sulfur|formation|clusters|transport|warm|plots} ..."
    exit 2
fi
for stage in "$@"; do
    case "$stage" in
        all) inputs; fits; nominal; sulfur; formation; clusters; transport; warm; plots ;;
        inputs|fits|nominal|sulfur|formation|clusters|transport|warm|plots) "$stage" ;;
        *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
done
