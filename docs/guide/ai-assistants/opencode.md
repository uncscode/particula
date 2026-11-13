# OpenCode

## Native OpenCode 2 package

The adapter targets OpenCode 2.1.3 and pins `@opencode/plugin` 2.1.3.
It registers eleven skills from its own installed resources. Loading it reads
package files only: it does not run Python, install dependencies into your
project, or start simulations.

```bash
opencode plugin add 'github:uncscode/particula#main::path:plugins/particula-helper/opencode'
opencode plugin list
```

Start a new session in the intended project and ask:

> @particula-setup Check this project's existing .venv before running anything.

For a fixed release, replace `RELEASE_TAG` with a release containing the adapter:

```bash
opencode plugin add 'github:uncscode/particula#RELEASE_TAG::path:plugins/particula-helper/opencode'
```

Use only one revision of the plugin at a time. Remove the old specification
before changing between branch and release installs.

```bash
opencode plugin check
opencode plugin update 'github:uncscode/particula#main::path:plugins/particula-helper/opencode'
opencode plugin remove 'github:uncscode/particula#main::path:plugins/particula-helper/opencode'
```

Existing skills with the same IDs are preserved, and unrelated skills are never
changed. Reload reconstructs registrations; unloading removes the adapter's
registration without deleting files. Restart after updating package contents.

## Local checkout or downloaded package

For local development, add the absolute directory to your project's existing
`opencode.json` **without replacing its other settings**:

```json
{
  "plugins": ["/absolute/path/to/particula/plugins/particula-helper/opencode"]
}
```

The release OpenCode ZIP contains a `particula-helper/` native package that can
be referenced the same way. Local packages need their SDK dependencies installed
first: run `npm ci --ignore-scripts` inside that package. No Python dependency
installation is triggered by this command.

## Standalone skills alternative

For OpenCode versions without the OpenCode 2 plugin API, extract the ZIP into a
temporary folder. Copy its eleven directories from `.opencode/skills/` into your
project's `.opencode/skills/`, preserving existing directories. If any name
already exists, stop and compare it; do not overwrite it blindly.

Every copied skill contains its own scripts and references. Start a new session
and ask it to load `particula-setup`. Remove only those eleven directories to undo
this manual installation. Do not install both the manual and native copies
unless you intentionally want the project skills to take precedence.

See [OpenCode plugin installation](https://opencode.ai/docs/plugins),
[skill discovery](https://opencode.ai/docs/skills/), and the
[shared first-run guide](index.md).
