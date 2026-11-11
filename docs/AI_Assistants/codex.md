# Codex and ChatGPT

## Codex CLI

Command syntax verified with Codex CLI 0.167.0:

```bash
codex plugin marketplace add uncscode/particula
codex plugin add particula-helper@particula
codex plugin list --marketplace particula --json
```

Start a new session in your project, then ask it to use `particula-setup` and
your existing `.venv`. The repository marketplace advertises the plugin without
automatically enabling it for every contributor.

For a release-pinned installation, replace `RELEASE_TAG` with an actual tag:

```bash
codex plugin marketplace add uncscode/particula --ref RELEASE_TAG
codex plugin add particula-helper@particula
```

Refresh and reinstall to pick up an updated plugin version:

```bash
codex plugin marketplace upgrade particula
codex plugin add particula-helper@particula
```

Remove only this plugin, optionally followed by its marketplace:

```bash
codex plugin remove particula-helper@particula
codex plugin marketplace remove particula
```

For local development, register your checkout instead of the GitHub source:

```bash
codex plugin marketplace add /absolute/path/to/particula
codex plugin add particula-helper@particula
```

Do not register a local checkout and the remote repository under the same
marketplace name simultaneously. Remove the old marketplace before switching.

## ChatGPT and desktop packages

In a desktop host that supports repo marketplaces, open its plugin browser,
select the `particula` marketplace, and install Particula Helper. Restart or
start a new chat after installing. Where custom plugin uploads are supported,
use `particula-helper-codex.zip` from a Particula release.

The ZIP uses the portable Agent Plugins manifest and includes a Codex
compatibility manifest. Repository distribution is separate from publication
in the public plugin directory; no public directory listing is implied.
Web/account upload availability depends on the host and account. This release
does not claim that the desktop upload path has been tested.

Skills can provide instructions anywhere they load, but simulations require
Python execution and filesystem access. A web-hosted interpreter is not your
local project environment. Never assume it can use your local `.venv`.

See [OpenAI plugin packaging and marketplace documentation](https://developers.openai.com/plugins/build/plugins)
and the [shared first-run guide](index.md).
