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
2. **Squash before review** if your PR has dozens of commits.
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

---

## Troubleshooting & Quick Reference

### Reset, Rebase & History Repair
| Task | Command |
|------|---------|
| Soft-reset all commits since `FIRST_COMMIT_HASH` into the index | `git reset --soft FIRST_COMMIT_HASH^` |
| Abort an in-progress rebase | `git rebase --abort` |
| Continue after fixing conflicts in a rebase | `git rebase --continue` |
| Amend the most recent commit | `git commit --amend` |
| Cherry-pick a specific commit onto HEAD | `git cherry-pick <commit>` |
| Revert (invert) a specific commit | `git revert <commit>` |

### Merge & Conflict Resolution
| Task | Command |
|------|---------|
| Abort an in-progress merge | `git merge --abort` |
| Continue (finish) a merge after resolving conflicts | `git commit` |

### Staging & Snapshot Creation
| Task | Command |
|------|---------|
| Stage everything after a reset | `git add .` |
| Show unstaged changes | `git diff` |
| Show staged changes | `git diff --cached` |
| Discard local changes in a file | `git restore <file>` |
| Commit staged changes (concise, present-tense message) | `git commit -m "message"` |

### Branch Management
| Task | Command |
|------|---------|
| Create a new branch and switch to it | `git switch -c <new-branch>` |
| Switch to an existing branch | `git switch <branch>` |
| Delete a local branch | `git branch -d <branch>` |
| Delete a remote branch | `git push origin --delete <branch>` |
| Pull with rebase (linear history) | `git pull --rebase` |
| Force-push safely (checks upstream) | `git push --force-with-lease` |
| Fetch all refs and prune deleted branches | `git fetch --all --prune` |

### Stash (Shelve) Work
| Task | Command |
|------|---------|
| Stash current changes | `git stash push -m "msg"` |
| List stashes | `git stash list` |
| Re-apply and drop latest stash | `git stash pop` |

### Clean-up & Inspection
| Task | Command |
|------|---------|
| Remove untracked files & dirs | `git clean -fd` |
| Compact one-line graph of history | `git log --oneline --graph --decorate --all` |


---

References:

- [Git: References](https://git-scm.com/book/en/v2)
- Firebase [Git in 100s](https://www.youtube.com/watch?v=hwP7WQkmECE), [Longer Video](https://www.youtube.com/watch?v=HkdAHXoRtos)
- ArjanCodes [Git Branches](https://www.youtube.com/watch?v=viAZQjs5lHk)