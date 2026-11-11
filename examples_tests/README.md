# Example test suites

All 13 source example test modules live here with their original basenames.
They exercise runnable examples under `docs/Examples/`, including CPU, direct
GPU, and resident GPU workflows. Package tests remain under `particula/`;
release infrastructure and marker-policy tests live under `scripts/tests/`.

## Source-checkout validation

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

These tests retain runtime, numerical, and API-boundary checks. Documentation
prose/link, notebook publication-state, and planning-reference assertions are
not part of the suite; MkDocs validation remains a separate documentation task.

## Installed-package release validation

The 0.2.x release runner has two isolated suites:

```bash
# Fresh wheel, package tests only; no docs tree in the test workspace.
python scripts/run_release_tests.py

# Fresh wheel, only the declared CPU example tests and matching scripts.
python scripts/run_release_tests.py --suite examples
```

`--suite package` is the default. It stages package tests, fixtures, conftests,
and pytest configuration, but neither this directory nor documentation inputs.
`--suite examples` stages root conftest/config and only these test/script pairs:

| Test in this directory | Script under `docs/Examples/` |
| --- | --- |
| `dilution_example_test.py` | `cpu_dilution.py` |
| `nucleation_example_test.py` | `Nucleation/cpu_nucleation.py` |
| `condensation_latent_heat_example_test.py` | `Dynamics/Condensation/Condensation_Latent_Heat.py` |

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

GPU/Warp/CUDA and benchmark release exclusions are explicitly limited to 0.2.x
and require review for v0.3. `warp-lang` remains a runtime dependency; these
release exclusions do not narrow source-CI GPU coverage.

The new layout and suite option require a new source archive or reviewed recipe
patch for external conda adoption. An unpatched v0.2.14 archive requires its
earlier seven-input, one-command contract, not these new test paths. See
[conda release validation and feedstock handoff](../conda/README.md) for both
contracts, isolation details, and evidence requirements.
