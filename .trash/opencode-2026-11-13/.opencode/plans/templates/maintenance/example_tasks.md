<!-- TEMPLATE: Replace this entire file with representative tasks -->

Provide representative tasks agents can spin into GitHub issues. These help
illustrate the scope and complexity of the maintenance work.

**Example (M23):**
1. Update `run_pytest` parsing so coverage-gate failures cannot produce PASS
   when pytest reports fail-under errors.
2. Label single-file and single-folder pytest runs as scoped validation and
   surface exit-code-4 stdout/stderr details.
3. Fix `adw_spec` failure envelopes to distinguish state-missing vs
   write-format errors with stdout, stderr, and exit context.
4. Rework `git_operations` continue handling so rebases advance after staged
   conflict resolution or return a bounded next-step diagnostic.
5. Ship `adw_plans analytics` wrapper parity with bun tests and
   wrapper-facing docs.
