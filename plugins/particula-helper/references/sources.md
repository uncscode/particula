# Version-matched documentation

Use the user's selected Particula interpreter to locate local documentation
before looking up APIs on the live site:

```bash
python -m particula_docs --json
python -m particula_docs
rg "ParticleData|GasData" "$(python -m particula_docs)/guide"
```

`particula-docs` is the equivalent console command. The locator uses only the
standard library; it does not import simulation code or require a working GPU.
Its JSON reports the version, path, mode, file count and uncompressed byte size.
Wheel installs resolve to version-matched bundled docs. Editable installs resolve
to the selected checkout's maintained `docs/` tree. Run the command outside an
unrelated Particula checkout to avoid Python's current-directory shadowing.

Read/search the returned path using available file tools or Python's
`pathlib.Path.rglob` if `rg` is unavailable. Start with `guide/data-model/`,
`guide/processes/`, `guide/gpu/`, or `guide/migration/v0.3.0/` as appropriate;
`examples/` and `theory/` contain readable Python sources. Relative links between
bundled files retain their structure. Generated API pages, website outputs and
repository tooling are not bundled. Do not write results into installed docs.

Older installations may lack `particula_docs`. Report that limitation, inspect
the installed distribution version, and use matching source documentation or
version-compatible online sources. Do not silently upgrade or switch environments.
Local docs establish the release's intended contract; verify actual signatures
and preserve the bounded numerical evidence in [compatibility](compatibility.md).
A doc bundle is not evidence that every example or device path is qualified.

## Online sources

Use these for missing material or explicitly requested current information,
checking compatibility with the selected installed version:

- Documentation: https://uncscode.github.io/particula/guide/
- Source and version history: https://github.com/uncscode/particula
- Examples: https://uncscode.github.io/particula/examples/
- Scientific citation: https://doi.org/10.5281/zenodo.6634653
