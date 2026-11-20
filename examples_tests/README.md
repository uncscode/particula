# Example test suites

Source example test modules live here, including the original 13 modules
with their original basenames and the native migration regressions.
They exercise runnable studies under `docs/studies/`, including CPU, direct
GPU, and resident GPU workflows. Package tests live under `particula/`.
Release infrastructure and marker-policy tests live under `scripts/tests/`.

## Source-checkout validation

Install `particula[extra]` from the checkout plus `pytest PyYAML` for the
source test environment (or use both development and extra dependency groups).
Native examples exercise plotting, unit conversion, chemistry and progress
reporting, so the base runtime dependencies alone are insufficient.

Run these suites separately from the repository root:

```bash
# Default testpaths selects only particula.
pytest -q -Werror

# All source example tests, including GPU coverage.
pytest examples_tests -q -Werror

# Infrastructure and pytest_marker_policy_test.py.
pytest scripts/tests -q -Werror
```

Source CI invokes all three suites explicitly. The example invocation must not
use the release-only CPU allowlist: source GPU example coverage is retained.
Warp CPU is the baseline when Warp is installed; optional CUDA validation
skips cleanly when unavailable. Existing per-example capability requirements
still apply; CPU execution is not a fallback for native-CUDA-only examples.

`native_quick_start_docs_test.py` runs the published first CPU workflow and
its guide snippets for reloading and resuming a simulation. The documentation
checks separately validate the study catalog, rendered links and saved outputs.

## Installed-package release validation

Use the release runner for isolated package and example checks.

```bash
# Fresh wheel, package tests only; no docs tree in the test workspace.
python scripts/run_release_tests.py

# Fresh wheel, only the declared CPU example tests and matching scripts.
python scripts/run_release_tests.py --suite examples
```

`--suite package` is the default. It stages package tests, fixtures, conftests,
and pytest configuration, but neither this directory nor documentation inputs.
`--suite examples` stages root conftest/config and only these test/script pairs:

| Test in this directory | Script under `docs/studies/examples/particle-processes/` |
| --- | --- |
| `dilution_example_test.py` | `sources-and-dilution/cpu_dilution.py` |
| `nucleation_example_test.py` | `sources-and-dilution/cpu_nucleation.py` |
| `condensation_latent_heat_example_test.py` | `condensation/condensation_latent_heat.py` |

`CPU_EXAMPLE_TESTS` in
[`scripts/run_release_tests.py`](../scripts/run_release_tests.py) is the
authoritative release-test allowlist; `CPU_EXAMPLES` lists the matching scripts.
Missing inputs or a declared module contributing no selected tests fail the
release check. Neither suite stages the full documentation tree, planning
records, infrastructure tests, or package application sources.

The conda recipe uses the already-installed package and runs both commands:

```bash
python scripts/run_release_tests.py --installed
python scripts/run_release_tests.py --installed --suite examples
```

See [conda release validation and feedstock handoff](../conda/README.md)
for the coupled CPU suite, release exclusions, version-specific external
recipe inputs and qualification evidence.
