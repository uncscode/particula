# Wrapper test harness notes

This directory is internal Bun test infrastructure for wrapper contract testing.

- Run all wrapper tests: `cd .opencode && bun test tools/__tests__`
- Helpers under `helpers/` intentionally monkeypatch `Bun.$` / `Bun.spawnSync` and must call restore functions in `afterEach`.
