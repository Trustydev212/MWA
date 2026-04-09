# Release Checklist

MWA uses **tag-driven releases**: pushing a `v*.*.*` tag to `main`
triggers the [`.github/workflows/release.yml`](../.github/workflows/release.yml)
workflow which runs tests, builds sdist + wheel, smoke-installs the
wheel, validates PyPI metadata with `twine check`, and uploads to
PyPI using the `PYPI_API_TOKEN` repository secret.

## One-time setup

### 1. PyPI account + API token

If you haven't already:

1. Register at https://pypi.org/account/register/ (and enable 2FA).
2. Create an API token at https://pypi.org/manage/account/token/.
   - **First upload**: use an "Entire account" scope because the
     `mwa` project doesn't exist on PyPI yet.
   - **After first upload**: revoke the account-wide token and
     create a **project-scoped** token limited to `mwa`.  Narrower
     blast radius if the token ever leaks.

### 2. GitHub repository secret

Add the token to the repo:

1. Go to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Name: `PYPI_API_TOKEN`.
4. Value: the full `pypi-...` token string (don't trim whitespace).
5. Save.

GitHub auto-masks this value in workflow logs.

### 3. (Optional) Protected `pypi` environment

For belt-and-suspenders safety, add a protection rule so only you
can approve a release:

1. **Settings → Environments → New environment → `pypi`**.
2. Enable **Required reviewers** → add yourself.
3. Save.

The release workflow targets `environment: pypi`, so every publish
will wait for your manual approval in the Actions UI.

## Per-release checklist

### Pre-flight (on main, before tagging)

- [ ] All intended PRs are merged to `main`.
- [ ] `CHANGELOG.md` has a new section for the version you're
      about to release, dated and populated.  Move the relevant
      entries out of `[Unreleased]`.
- [ ] `pyproject.toml` version field matches the tag you're
      about to push.
- [ ] `git pull origin main` locally so your tag lands on the
      right commit.
- [ ] Local smoke: `python -m build && ls dist/` shows the expected
      wheel + sdist filenames.
- [ ] Local tests green: `pytest tests/ -q -m "not integration"`.

### Dry-run (optional but strongly recommended for the first few releases)

Validate the whole pipeline without actually publishing:

1. Go to **Actions → Release to PyPI → Run workflow**.
2. Leave `dry_run` checked (default `true`).
3. Click **Run workflow**.

The workflow will run tests + build + smoke-install + `twine check`
and stop.  Nothing is uploaded to PyPI.  If all jobs are green, you
can tag with confidence.

### Tag and push

```bash
# Cut the tag (annotated, with a short message)
git tag -a v0.1.0 -m "Release 0.1.0 — first alpha"

# Push the tag (this is what triggers the release workflow)
git push origin v0.1.0
```

### Watch the release

1. Open https://github.com/Trustydev212/MWA/actions — the **Release
   to PyPI** run should appear within seconds.
2. Three jobs run sequentially: `test` → `build` → `publish`.
3. If `environment: pypi` has protection, click the pending
   approval before `publish` runs.
4. When `publish` finishes green, PyPI indexing takes ~30 seconds.

### Verify the release

From a fresh venv:

```bash
python -m venv /tmp/verify-release
source /tmp/verify-release/bin/activate
pip install --upgrade pip
pip install mwa==0.1.0
python -c "
import mwa
from mwa.sdk import AgentRuntime, WorldAgent
print(f'mwa {mwa.__version__} — SDK importable')
"
deactivate && rm -rf /tmp/verify-release
```

If the install works and the SDK imports cleanly, the release is
shipped.  Announce it however you normally do.

## Rollback

If a release is broken and needs to come off PyPI **immediately**:

### Option 1 — Yank (preferred)

Yanking hides the release from `pip install mwa` (without a version
pin) but leaves it downloadable for users who pinned it.  Non-
destructive.

1. Go to https://pypi.org/manage/project/mwa/releases/.
2. Click the broken release → **Yank release** → provide a reason
   (will be shown to users who try to install a pinned version).

### Option 2 — Delete and re-release

**Only if yanking isn't enough** — e.g., you accidentally uploaded
secrets or broken metadata that would confuse future releases.
Deleting a release makes the version unusable forever (PyPI doesn't
allow re-uploading a deleted version under the same name).

1. PyPI UI → project → **Delete release**.
2. Bump the version in `pyproject.toml` (`0.1.0` → `0.1.1`).
3. Update `CHANGELOG.md`.
4. Commit, tag, push.

### Either way: delete the git tag

```bash
git tag -d v0.1.0                   # delete local
git push origin :refs/tags/v0.1.0   # delete remote
```

This prevents accidental re-triggering of the release workflow.

## Versioning rules (SemVer-ish for 0.x)

While MWA is pre-1.0:

- **0.x.Y** (patch) — backwards-compatible bug fixes, doc updates,
  CI changes.
- **0.X.y** (minor) — new features, may include breaking API changes
  if documented in CHANGELOG.  Everything between minors is fair
  game for 0.x.
- **X.0.0** (major) — reserved for 1.0 stable release when the
  public API is declared frozen.

Pre-releases use the `0.2.0-rc1`, `0.2.0-rc2` form.  The release
workflow's tag regex (`v*.*.*`) accepts these.

## Publishing a pre-release

Same flow, just a different tag shape:

```bash
git tag -a v0.2.0-rc1 -m "Release 0.2.0-rc1"
git push origin v0.2.0-rc1
```

PyPI recognises the `-rc1` suffix and marks the release as a
pre-release — it won't be installed by `pip install mwa` without
an explicit `--pre` flag.

## FAQ

### Why not use Trusted Publishing (OIDC)?

Short answer: we already set up `PYPI_API_TOKEN` and the token
path works fine.  Trusted Publishing is the modern best practice
and we may migrate to it later (it removes the token from the
equation entirely), but it adds steps during setup that aren't
worth the churn for a 1-person project.

To migrate later: replace the `TWINE_USERNAME`/`TWINE_PASSWORD`
env block in `release.yml` with `pypa/gh-action-pypi-publish` and
register a trusted publisher at
https://pypi.org/manage/account/publishing/.

### Why does the workflow run the full test suite before building?

Because a broken wheel is worse than a failed release.  The minute
we skip the test gate, someone will push a tag from a branch that
regressed `mwa.sdk` and `pip install mwa` will ship broken code to
every user.  The ~2 minute test run is cheap insurance.

### Can I publish from my local machine instead?

Yes, but only as a fallback.  Manual publish:

```bash
python -m build
twine upload dist/*
# username: __token__
# password: <PyPI token>
```

Prefer the workflow path — it's reproducible, leaves an audit
trail in Actions, and runs the smoke-install gate that `twine
upload` doesn't.

### A release failed halfway through.  What now?

Look at the Actions log.  The workflow's `concurrency` group is
`release` with `cancel-in-progress: false`, so you can re-run the
failed job in place after fixing whatever broke.  If the `publish`
job hit PyPI but errored out mid-upload, PyPI's idempotency will
reject a re-run of the same version — delete the tag, bump the
version, re-tag, re-push.
