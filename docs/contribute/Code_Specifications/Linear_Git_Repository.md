# Linear Git Repository

Use a **rebase‑then‑fast‑forward** strategy to keep a linear history.  
Avoid the *“Update branch”* or *“Squash & merge”* buttons on GitHub—those create merge commits instead of a clean rebase.

A linear timeline makes it trivial to:

- read the progression of a feature from top to bottom,
- bisect for bugs (`git bisect`), and
- cherry‑pick or revert single commits.

In short, it keeps the repository **human–readable** and
**tool‑friendly**, so the extra rebase step pays off quickly.

**Rule of thumb**

1. **Rebase early, rebase often** – keep your PR branch current with `main`.
2. **Squash before review** if your PR has dozens of WIP commits.
3. **Always force‑push (`--force-with-lease`)** after a rebase or squash so GitHub updates the PR without adding merge commits.

---

## Rebase your feature branch onto `main`

_Why rebase?_  

Rebasing re‑plays your commits on top of the current
`main`, avoiding the merge commit that GitHub’s
*“Update branch”* button would create.  

Reviewers now see *only* the changes relevant to the feature, not a
noisy merge diff.

```bash
# Make sure you have the latest main
git fetch origin

# Switch to your feature branch
git switch my‑feature

# Rebase onto the tip of main with conflict checking
git rebase main
# ‑‐rebase merges your commits one‑by‑one
# If conflicts appear:
#   edit the files → git add <files> → git rebase --continue
#   or git rebase --abort to cancel the rebase

# To linearize history, use --force if you have already pushed the branch to GitHub
git rebase --force main

# Push the rebased branch to the PR
git push --force-with-lease   # safer than --force
```

> **Why CLI?**  
> GitHub’s *“Update branch”* option performs a **merge**, polluting history with an extra merge commit. Rebasing in the terminal keeps history linear and bisect‑friendly.

---

## Squash a long commit history

Why squash?  

A PR that contains dozens of “fix typo” or “WIP” commits is hard to
review and clutters the permanent history.  Squashing groups these
micro‑commits into logical units that tell a clear story.

Interactive rebase lets you collapse many tiny commits into one (or a few) logical commits before review.

```bash
# From your feature branch, start an interactive rebase
git rebase -i origin/main
# In the editor, mark commits you want to combine as 'squash' or 'fixup'
# Save & close → Git opens a second editor window for the new commit message
# Write a concise message, then save & close

# Push the squashed branch back to the PR
git push --force-with-lease
```
> **Tip:** Configure Git to auto‑rebase when you pull:
> ```bash
> git config --global pull.rebase true
> ```

---

## 3. Troubleshooting & quick reference

| Task | Command |
|------|---------|
| Soft‑reset all commits since `FIRST_COMMIT_HASH` into the index | `git reset --soft FIRST_COMMIT_HASH^` |
| Stage everything after a soft reset | `git add .` |
| Commit staged changes | `git commit -m "Concise, present‑tense message"` |
| Abort an in‑progress rebase | `git rebase --abort` |
| Continue after fixing conflicts | `git rebase --continue` |

---

