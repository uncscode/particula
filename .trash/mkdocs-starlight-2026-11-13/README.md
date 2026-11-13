# Retired MkDocs publishing implementation

Original repository paths are preserved below this directory. These files are
reference material, excluded from publication, discovery, and package builds.
The source revision and exact file list are recorded in inventory.json.

Use `npm --prefix website run build` for the complete Starlight build and link
verification; use `python scripts/check_docs_structure.py` for source checks.
Scientific results are refreshed explicitly with `scripts/docs_examples.py`.

There is no automatic deletion date. A later maintainer cleanup may remove this
archive. Roll back the complete migration commit/PR and republish its parent
revision; copying a single archived config is not a runnable rollback.

## Publication rollback

The previous published `gh-pages` revision is
`6597906ba8dd09bdccb9c3e502d8155c6668a1d4` (also recorded in `inventory.json`).
Keep this Git object reachable until cutover acceptance is complete.

If production verification fails after merge, first disable or revert the new
main deployment workflow so it cannot overwrite the restored publication. In a
separate checkout of the upstream `gh-pages` branch, restore the full tree from
the recorded publication revision, commit that restoration on top of the current
branch, and push it. This restores the actual published HTML/assets without
requiring the retired build stack or rewriting branch history. Verify the home,
search assets, legacy redirects, representative example and API links afterward.

For a source rollback, revert the complete migration PR and restore its tested
Python/Node or legacy environment as appropriate. Archived reference files alone
are not a runnable backup. Production verification and rollback are conditional
post-merge operations; neither is claimed by local preview validation.
