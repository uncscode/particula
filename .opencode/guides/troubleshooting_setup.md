# ADW Setup Troubleshooting

Practical fixes for common setup failures. Start with the [ADW Setup Guide](setup_guide.md)
for installation and the [Backend Configuration](backend_configuration.md) reference for
platform-specific variables and routing. Use this guide when validation fails, connectivity
is blocked, or credentials are rejected.

## Table of Contents
- [Environment Variable Issues](#environment-variable-issues)
  - [.env not loaded](#env-not-loaded)
  - [Environment variable format error](#environment-variable-format-error)
- [Authentication Failures](#authentication-failures)
  - [GitHub 401 Unauthorized](#github-401-unauthorized)
  - [GitLab token permission denied](#gitlab-token-permission-denied)
  - [Token expired or revoked](#token-expired-or-revoked)
- [API Connectivity Problems](#api-connectivity-problems)
  - [Timeout reaching anthropic.com](#timeout-reaching-anthropiccom)
  - [Timeout reaching api.github.com](#timeout-reaching-apigithubcom)
  - [SSL certificate errors](#ssl-certificate-errors)
  - [GitHub API rate limit 403](#github-api-rate-limit-403)
- [Repository Access Issues](#repository-access-issues)
  - [Repository not found (404)](#repository-not-found-404)
  - [Permission denied to repository](#permission-denied-to-repository)
  - [Fork not accessible or wrong target](#fork-not-accessible-or-wrong-target)
- [Worktree Directory Issues](#worktree-directory-issues)
  - [Plans --cwd path or missing plans/](#plans---cwd-path-or-missing-plans)
- [Git Configuration Errors](#git-configuration-errors)
  - [git user.email not configured](#git-useremail-not-configured)
  - [git user.name not configured](#git-username-not-configured)
  - [git not found in PATH](#git-not-found-in-path)
- [Platform-Specific Issues](#platform-specific-issues)
  - [GitHub Enterprise](#github-enterprise)
  - [Self-hosted GitLab](#self-hosted-gitlab)
  - [GitLab CI/CD token issues](#gitlab-cicd-token-issues)
- [Still Having Issues](#still-having-issues)

## Environment Variable Issues

### .env not loaded
**Symptoms**
```bash
$ adw setup validate
✗ Environment file not loaded (.env missing or unreadable)
```

**Possible Causes**
- `.env` not in repository root or has Windows CRLF line endings.
- File permissions prevent reading (e.g., 000).
- `direnv` not enabled (`direnv allow` missing) or shell not sourcing `.env`.
- WSL shell opened in a different path than the repo.

**Solution**
1. Confirm location and permissions:
   ```bash
   ls -l .env
   file .env     # ensure UTF-8, not UTF-16
   ```
2. Normalize line endings if copied from Windows:
   ```bash
   sed -i 's/\r$//' .env
   ```
3. Reload the file:
   ```bash
   set -a && source .env && set +a
   # or enable direnv
   direnv allow
   ```
   Trust boundary warning: `source .env` executes shell code from the file.
   Only run this against files you trust.
4. On WSL, verify you are inside the repo path (e.g., `/home/<user>/Code/Agent`).

**Verification**
```bash
adw setup validate
# Check if sensitive environment variables are set (without printing their values)
for var in GITHUB_PAT GITLAB_TOKEN; do
  val="${!var}"
  if [ -n "$val" ]; then
    echo "$var is set (length: ${#val})"
  else
    echo "$var is NOT set"
  fi
done
```

### Environment variable format error
**Symptoms**
```bash
$ adw setup validate
✗ Invalid environment variable format: unexpected quotes or trailing spaces
```

**Possible Causes**
- Quotes included in values (e.g., `GITHUB_PAT="ghp_..."`).
- Trailing spaces or hidden characters from copy/paste.
- Multiline private key paths not escaped.

**Solution**
1. Remove quotes and trailing spaces in `.env`:
   ```env
   GITHUB_PAT=ghp_...
   GITHUB_REPO_URL=https://github.com/owner/repo
   ```
2. Re-source the environment and re-run validation:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```

## Authentication Failures

### GitHub 401 Unauthorized
**Symptoms**
```bash
$ adw setup validate
✗ GitHub connectivity: 401 Unauthorized
```
```bash
$ gh auth status
X github.com: not logged in
```

**Possible Causes**
- PAT missing scopes (`repo`, `workflow`).
- Token not exported in the current shell.

**Solution**
1. Refresh GitHub auth (PAT example):
   ```bash
   gh auth login --hostname github.com --scopes repo,workflow --with-token < ~/.config/gh/token
   ```
   Or re-add to `.env`:
   ```env
   GITHUB_PAT=ghp_...
   ```
2. Reload env and validate:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```

**Verification**
```bash
gh auth status
curl -I https://api.github.com/user -H "Authorization: Bearer $GITHUB_PAT"
```

### GitHub App credentials removed
**Symptoms**
- Validation fails with missing PAT or unexpected GitHub auth errors.
- `.env` still contains `GITHUB_APP_*` variables from older setups.

**Solution**
1. Remove any `GITHUB_APP_*` entries from `.env` or CI secrets.
2. Set a GitHub PAT (or `GITHUB_TOKEN` in Actions):
   ```env
   GITHUB_PAT=ghp_...
   GITHUB_REPO_URL=https://github.com/<you>/<repo>
   ```
3. Reload the environment and re-run validation:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```

### GitLab token permission denied
**Symptoms**
```bash
$ adw setup validate
✗ GitLab connectivity: 403 Forbidden (permission denied)
```

**Possible Causes**
- Token missing scopes: `api`, `read_repository`, `write_repository`.
- Token tied to a user without project access.
- Self-hosted instance not specified with `ADW_PLATFORM_HINT=gitlab`.

**Solution**
1. Create a new Personal Access Token with required scopes.
2. Update `.env` and reload:
   ```env
   GITLAB_REPO_URL=https://gitlab.com/<group>/<project>
   GITLAB_TOKEN=glpat-...
   ADW_PLATFORM_HINT=gitlab
   ```
3. Re-run validation:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```

**Verification**
```bash
curl -I "$GITLAB_REPO_URL" -H "PRIVATE-TOKEN: $GITLAB_TOKEN"
``` 

### Token expired or revoked
**Symptoms**
- `401 Unauthorized` after previously working credentials.
- GitHub/GitLab UI shows token revoked or expired.

**Possible Causes**
- PAT or GitLab token exceeded expiry window.
- Credential rotation policy revoked previous tokens.

**Solution**
1. Revoke old tokens in the provider UI to avoid confusion.
2. Issue a new token with required scopes (`repo`, `workflow` or `api`, `read_repository`, `write_repository`).
3. Update `.env`, reload, and validate:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```
4. If using CI secrets, rotate the stored secret and redeploy the job environment.

**Verification**
```bash
gh auth status    # GitHub
curl -I "$GITLAB_REPO_URL" -H "PRIVATE-TOKEN: $GITLAB_TOKEN"  # GitLab
adw health
```

## API Connectivity Problems

### Timeout reaching anthropic.com
**Symptoms**
```text
requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='api.anthropic.com', ...)
```

**Possible Causes**
- Corporate proxy blocking outbound requests.
- Local firewall or VPN blocking Anthropic.
- Intermittent ISP/region routing issues.

**Solution**
1. Test reachability:
   ```bash
   curl -I https://api.anthropic.com
   ```
2. If behind a proxy, export settings (adjust NO_PROXY for local services):
   ```bash
   export HTTPS_PROXY=https://proxy.company.com:8443
   export NO_PROXY=localhost,127.0.0.1
   ```
3. Re-run health checks:
   ```bash
   adw health
   ```
4. If proxy performs TLS interception, add the corporate CA to your system trust and
   set `REQUESTS_CA_BUNDLE=/path/to/ca.pem`.

**Verification**
```bash
curl -I https://api.anthropic.com
adw setup validate
```

### Timeout reaching api.github.com
**Symptoms**
```text
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.github.com', ...)
```

**Possible Causes**
- Corporate proxy/firewall blocking GitHub.
- DNS resolution issues.
- VPN routing instability.

**Solution**
1. Basic connectivity check:
   ```bash
   curl -I https://api.github.com
   ```
2. If blocked, configure proxy variables:
   ```bash
   export HTTPS_PROXY=https://proxy.company.com:8443
   export NO_PROXY=localhost,127.0.0.1
   ```
3. Validate GitHub auth and rate limits:
   ```bash
   gh auth status
   curl -I https://api.github.com/rate_limit -H "Authorization: Bearer $GITHUB_PAT"
   ```
4. Retry validation after network changes:
   ```bash
   adw setup validate
   ```

**Verification**
```bash
curl -I https://api.github.com
adw health
```

### SSL certificate errors
**Symptoms**
```text
SSL: CERTIFICATE_VERIFY_FAILED
``` 

**Possible Causes**
- Corporate TLS interception without trusted root CA installed.
- Self-signed certificates on GitHub Enterprise or self-hosted GitLab.

**Solution**
1. Obtain the corporate or self-hosted CA certificate and place it locally.
2. Export the CA path for Python/requests:
   ```bash
   export REQUESTS_CA_BUNDLE=/path/to/corporate-ca.pem
   ```
3. For `gh`, add the cert to your OS trust store (varies by distro) or use
   `GH_HOST` pointing to the enterprise host after trust is configured.
4. Re-run validation:
   ```bash
   adw setup validate
   ```

**Verification**
```bash
curl -I https://api.github.com
curl -I "$GITLAB_REPO_URL" -H "PRIVATE-TOKEN: $GITLAB_TOKEN"
```

### GitHub API rate limit 403
**Symptoms**
```text
HTTP 403: rate limit exceeded
X-RateLimit-Remaining: 0
```

**Possible Causes**
- Using unauthenticated requests.
- Fine-grained PAT without sufficient quota for the org.
- Excessive polling from other tools.

**Solution**
1. Check limits:
   ```bash
   curl -I https://api.github.com/rate_limit -H "Authorization: Bearer $GITHUB_PAT"
   ```
2. Wait for reset or use a PAT with sufficient scopes/limits.
3. Reduce request volume (pause other automation hitting the same token).
4. Re-run health once the limit resets:
   ```bash
   adw health
   ```

**Verification**
```bash
curl -I https://api.github.com/rate_limit -H "Authorization: Bearer $GITHUB_PAT"
```

## Repository Access Issues

### Repository not found (404)
**Symptoms**
```text
Failed to fetch issue #123 from owner/repo: 404 Not Found
Repository detection method: gh_cli
```

**Possible Causes**
- `GITHUB_REPO_URL`/`GITLAB_REPO_URL` points to the wrong repo or is private.
- Detection using `gh` CLI is logged into a different org/account.
- Fork URL missing when targeting upstream.

**Solution**
1. Confirm URLs in `.env`:
   ```env
   GITHUB_REPO_URL=https://github.com/<you>/<fork>
   GITHUB_UPSTREAM_URL=https://github.com/<org>/<project>
   ```
2. Override detection temporarily if needed:
   ```bash
   export GH_REPO=<owner>/<repo>
   adw setup validate
   ```
3. Ensure `gh auth status` shows the intended account.
4. If errors mention `prefer_scope`, pass an explicit scope when retrying:
   ```bash
   adw platform create-issue --title "Bug" --body "Details" --prefer-scope upstream
   adw platform comment --issue-number 42 --body "Update" --prefer-scope upstream
   adw platform create-pr --title "Fix" --head my-branch --base main --prefer-scope upstream
   ```
   API/tooling calls can also provide `prefer_scope: "upstream"` in the
   `platform_operations` payload to force routing.

**Verification**
```bash
gh repo view
adw setup validate
```

### Permission denied to repository
**Symptoms**
```text
remote: Permission to owner/repo.git denied
fatal: unable to access 'https://github.com/owner/repo/': The requested URL returned error: 403
```

**Possible Causes**
- PAT lacks `repo` or `write_repository` scope.
- User not added to the org/project.
- Token belongs to a user without branch push rights.

**Solution**
1. Ensure correct scopes and reload environment.
2. Validate git remote access:
   ```bash
   git ls-remote https://github.com/owner/repo.git
   ```
3. If using upstream routing, confirm fork permissions and upstream visibility.
4. Retry setup validation:
   ```bash
   adw setup validate
   ```

**Verification**
```bash
git ls-remote "$GITHUB_REPO_URL"
adw health
```

### Fork not accessible or wrong target
**Symptoms**
- Workflows report fork unavailable when `--target upstream` is used.
- Validation shows missing upstream URL when targeting upstream.

**Possible Causes**
- `ADW_TARGET_REPO` set to `upstream` without `GITHUB_UPSTREAM_URL`/`GITLAB_UPSTREAM_URL`.
- Fork URL points to upstream, not the contributor fork.

**Solution**
1. Set fork and upstream explicitly:
   ```env
   GITHUB_REPO_URL=https://github.com/<you>/Agent
   GITHUB_UPSTREAM_URL=https://github.com/<org>/Agent
   ADW_TARGET_REPO=fork   # or upstream when ready
   ```
2. Re-run with the desired target (add `--adw-id <id>` and `--resume` only when continuing an
   existing workflow):
   ```bash
   adw workflow complete 123 --target upstream
   adw setup validate
   ```

**Verification**
```bash
adw health
adw setup validate
```

### OperationNotAvailable when writing issues/labels
**Symptoms**
- Workflows fail with `OperationNotAvailable` during issue updates, comments, or
  label changes when `ADW_TARGET_REPO=upstream`.

**Possible Causes**
- Upstream token lacks `issue:write`, `issue:comment`, or `label:write` permissions.
- Upstream client is configured read-only for write operations.

**Solution**
1. Grant the upstream token write permissions for issues/labels/comments.
2. Re-run the workflow after updating credentials.

**Verification**
```bash
adw setup validate
adw health
```

## Git Configuration Errors

### git user.email not configured
**Symptoms**
```text
*** Please tell me who you are.
fatal: unable to auto-detect email address
```

**Possible Causes**
- Global git config missing `user.email`.
- Different user profiles between WSL and Windows.

**Solution**
1. Set email globally:
   ```bash
   git config --global user.email "you@example.com"
   ```
2. For WSL, ensure git in WSL has its own config (separate from Windows git).

**Verification**
```bash
git config --get user.email
adw setup validate
```

### git user.name not configured
**Symptoms**
```text
fatal: unable to auto-detect user.name
```

**Possible Causes**
- Global git config missing `user.name`.

**Solution**
1. Set name globally:
   ```bash
   git config --global user.name "Your Name"
   ```
2. Re-run validation:
   ```bash
   adw setup validate
   ```

**Verification**
```bash
git config --get user.name
```

### git not found in PATH
**Symptoms**
```bash
$ git --version
bash: git: command not found
```

**Possible Causes**
- Git not installed.
- PATH not updated (common on fresh Windows/WSL shells).

**Solution**
1. Install git:
   ```bash
   sudo apt-get update && sudo apt-get install -y git   # Debian/Ubuntu
   ```
2. Confirm PATH includes git:
   ```bash
   which git
   echo $PATH
   ```
3. On Windows, prefer WSL git for consistency or add git to PATH and restart shell.

**Verification**
```bash
git --version
adw health
```

## Worktree Directory Issues

### Symptoms
- Workflow creation fails immediately after worktree add with missing `.opencode/` or `adw-docs/` paths
- Worktree exists but lacks ADW files (empty/partial copies) when directories are gitignored in the main repo
- Errors mention permission denied, symlink copy, or missing source directory during setup
- Logs show rollback triggered after copy failure

### Possible Causes
- `.opencode/` or `adw-docs/` gitignored in the main repo, so git worktree skipped them
- Local `.gitignore` inside `.opencode/` or `adw-docs/` excluded nested files, leading to partial copies
- Source directories missing entirely in the root repo (nothing to copy)
- Permissions prevent copying (e.g., locked files, restrictive ACLs)
- Symlinks in source directories that cannot be dereferenced or copied

### Solution
1. Inspect the worktree contents (replace `<adw_id>` with your workflow ID):
   ```bash
   ls -la trees/<adw_id>/.opencode
   ls -la trees/<adw_id>/adw-docs
   ```
2. If directories are missing or partial, re-run workspace creation so ADW copies from the main repo.
3. Ensure `.opencode/` exists in the root repo; failure to copy `.opencode/` is **fatal** and triggers rollback of the worktree. `adw-docs/` copy is **warn-only**; workflows continue without it but docs commands may fail.
4. If directories are gitignored, this is expected: ADW always copies them after worktree creation using `dirs_exist_ok=True` to merge missing files.
5. Fix permissions and rerun if you see permission errors:
   ```bash
   sudo chown -R $(whoami) .opencode adw-docs
   chmod -R u+rwX .opencode adw-docs
   ```
6. If symlinks block copy, replace problematic symlinks with real files in the source repo, then retry.

### Verification
- Re-run workspace creation and check logs for copy outcomes:
  - `.opencode/` copy failure ⇒ fatal + rollback of worktree creation
  - `adw-docs/` copy failure ⇒ warning only; workflow continues
- Confirm directories now exist in the worktree:
  ```bash
  ls -la trees/<adw_id>/.opencode | head
  ls -la trees/<adw_id>/adw-docs | head
  ```
- Validate permissions on the source before retrying:
  ```bash
  ls -la .opencode adw-docs
  ```
- Check setup/workspace logs for copy messages around worktree creation (between worktree add and checkout).

### Plans --cwd path or missing plans/
**Symptoms**
- `adw plans list-sections <plan-id> --json --cwd <path>` fails during maintenance reruns.
- `adw plans validate --cwd <path>` reports missing plan files or an invalid repo root.

**Possible Causes**
- `--cwd` points to a relative or stale path from a different shell/worktree.
- The target directory exists but does not contain `<path>/plans/`.

**Solution**
1. Verify the target is an absolute path and includes `plans/`:
   ```bash
   ls -la /absolute/path/to/repo/plans
   ```
2. Confirm section discovery from that same path:
   ```bash
   adw plans list-sections M25 --json --cwd /absolute/path/to/repo
   ```
3. Re-run validation against the same absolute path:
   ```bash
   adw plans validate --cwd /absolute/path/to/repo
   ```

**Verification**
- `list-sections --json` returns canonical maintenance section keys as a sorted JSON map.
- `validate --cwd` exits `0` with all-valid output.

## Platform-Specific Issues

### GitHub Enterprise
**Symptoms**
- `404` or `SSL` errors when using enterprise host.
- `gh auth status` points to `github.com` instead of the enterprise domain.

**Possible Causes**
- `GITHUB_REPO_URL` not set to enterprise host.
- Missing enterprise CA trust.

**Solution**
1. Set enterprise URLs and host:
   ```env
   GITHUB_REPO_URL=https://github.company.com/<org>/<repo>
   GH_HOST=github.company.com
   ```
2. Add enterprise CA to trust store and set `REQUESTS_CA_BUNDLE` if needed.
3. Re-authenticate `gh` against the enterprise host:
   ```bash
   gh auth login --hostname $GH_HOST --scopes repo,workflow
   ```
4. Re-run validation:
   ```bash
   adw setup validate
   ```

**Verification**
```bash
gh auth status
curl -I "$GITHUB_REPO_URL"
```

### Self-hosted GitLab
**Symptoms**
- 404/SSL errors against `https://gitlab.company.com`.
- Validation reports platform mismatch.

**Possible Causes**
- `ADW_PLATFORM_HINT` not set to `gitlab` for custom domains.
- Missing CA trust for self-signed certificates.

**Solution**
1. Set required variables:
   ```env
   ADW_PLATFORM_HINT=gitlab
   GITLAB_REPO_URL=https://gitlab.company.com/group/project
   GITLAB_TOKEN=glpat-...
   REQUESTS_CA_BUNDLE=/path/to/company-ca.pem  # if self-signed
   ```
2. Reload and validate:
   ```bash
   set -a && source .env && set +a
   adw setup validate
   ```

**Verification**
```bash
curl -I "$GITLAB_REPO_URL" -H "PRIVATE-TOKEN: $GITLAB_TOKEN"
```

### GitLab CI/CD token issues
**Symptoms**
- Pipelines fail with `insufficient_scope` or `401` when using CI variables.

**Possible Causes**
- CI token lacks `api` or repository scopes.
- Masked/Protected flags prevent access on branches.

**Solution**
1. Create a project/group token with `api`, `read_repository`, `write_repository`.
2. Store it as a masked, protected CI variable (e.g., `GITLAB_TOKEN`).
3. Ensure the variable is available to the branch/PR pipeline.
4. Re-run the pipeline step or `adw setup validate` locally with the same token.

**Verification**
```bash
curl -I "$GITLAB_REPO_URL" -H "PRIVATE-TOKEN: $GITLAB_TOKEN"
adw health
```

## Still Having Issues
- Re-run `adw setup validate` and `adw health` and capture full output.
- Include the exact error message, platform (GitHub/GitLab), and whether you use a fork.
- Attach key command outputs:
  - `gh auth status` or GitLab token scopes
  - `git config --get user.name && git config --get user.email`
  - `curl -I https://api.github.com` and `curl -I https://api.anthropic.com`
- Check related guides: [ADW Setup Guide](setup_guide.md),
  [Backend Configuration](backend_configuration.md), and
  [Authorization Troubleshooting](../docs/Examples/setup/authorization-troubleshooting.md).
- If unresolved, open a GitHub issue with the above details (omit secrets) and recent log
  snippets from `adw setup validate`.
