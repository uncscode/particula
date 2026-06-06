<!-- TEMPLATE: Replace this entire file with your implementation task breakdown -->

Break tasks down by discipline. Keep each checkbox actionable and specific.

**Required subsections (include only those that apply):**

### Backend
- [ ] Task with specific file path and function name

### Frontend / CLI
- [ ] Task with specific command or UI element

### Tooling / Tests
- [ ] Task with specific test file and coverage target

**Example (E17-F5):**

### Backend
- [ ] Add `migrate_plans()` function in `adw/plans/migration.py`
- [ ] Add `validate_migration()` round-trip checker
- [ ] Wire `--source`, `--output`, `--dry-run` Click options in CLI

### Tooling / Tests
- [ ] Add `adw/plans/tests/migration_test.py` with round-trip validation
- [ ] Add dry-run parity tests (parse/validate without filesystem writes)
- [ ] Add error-handling tests for malformed markdown input
