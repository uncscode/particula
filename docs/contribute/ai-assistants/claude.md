# Claude Code, Desktop, and Cowork

## Claude Code

Command syntax verified with Claude Code 2.1.319:

```bash
claude plugin marketplace add uncscode/particula
claude plugin install particula-helper@particula
claude plugin list
```

Start a new session in your project and invoke
`/particula-helper:particula-setup`, asking it to use your existing `.venv`.

Update or remove:

```bash
claude plugin marketplace update particula
claude plugin update particula-helper@particula
claude plugin uninstall particula-helper@particula
```

The repository marketplace follows the registered Git revision. For a fixed
release without relying on Git-ref syntax across CLI versions, download
`particula-helper-claude.zip` from that release, extract it into a dedicated
folder, then load the extracted folder:

```bash
claude --plugin-dir /absolute/path/to/extracted-particula-helper
```

For local development:

```bash
claude plugin validate /absolute/path/to/particula
claude plugin validate /absolute/path/to/particula/plugins/particula-helper
claude --plugin-dir /absolute/path/to/particula/plugins/particula-helper
```

## Desktop and Cowork

Use **Customize → Plugins** to add the `uncscode/particula` repository
marketplace, or upload the Claude ZIP using the custom-plugin flow. The same
skills and scripts are packaged for Claude Code and Cowork; no separate MCP
extension is required for these instructions.

Verify the plugin appears, grant the intended project access using the host's
normal controls, and ask it to identify its Python interpreter before running
the setup workflow. Cowork's execution environment may differ from your terminal;
access to a selected folder does not establish that its `.venv` can run there.
Desktop upload and Cowork execution remain unverified until tested in that app.

Normal Chat can use the skills for guidance where supported, but uploading this
package does not give Chat access to your local interpreter. See
[Claude's plugin guide](https://support.claude.com/en/articles/13837440-use-plugins-in-claude)
and the [marketplace documentation](https://code.claude.com/docs/en/plugin-marketplaces).

Continue with the [shared first-run guide](index.md).
