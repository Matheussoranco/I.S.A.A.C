# Evidence Pack — `I.S.A.A.C #64` (`Validate release tags and pin PyPI publishing actions`)

- **Repository:** `Matheussoranco/I.S.A.A.C`
- **Issue:** `#64` (`Validate release tags and pin PyPI publishing actions`)
- **Trace:** `isaac-64`
- **Author:** Gemini 4 Neon (High reasoning effort)
- **Branch / Worktree:** `fix/64-release-tag-validation-pin-actions` (`C:\Users\mathe\Documents\Code\I.S.A.A.C-wt64`, base `origin/main` @ `8cdca15`)
- **Verification Status:** `PASS`
- **Review & Merge Readiness:** `PENDING_REVIEW` (awaiting independent cross-agent review from Claude Code and final integration assessment from Engineering Lead)

---

## 1. Acceptance Criteria Mapping (`#64`)

| Acceptance Criterion | Implementation | Verified By | Status |
|---|---|---|---|
| Pin actions used by the OIDC publishing job to reviewed full commit SHAs | Pinned `actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093` (`v4.3.0`) in both `test-wheel` and `publish-pypi`, and `pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086` (`release/v1` / `v1.14.3`) in `publish-pypi` (verified via GitHub API). Every `uses:` step in `.github/workflows/release.yml` is now pinned to a 40-hex commit SHA. | `test_release_workflow_pins_all_actions_to_full_commit_shas_and_gates_publish` | **PASS** |
| Validate version-tag push against package version and stop before publication if they differ | Added stdlib-only `scripts/validate_release_tag.py` verifying `refs/tags/v<version>` against both `pyproject.toml` (`[project].version`) and `wheelhouse/*.whl` `*.dist-info/METADATA` `Version:`. Wired as an early gate in `build-wheel` (before artifact upload) AND as a defense-in-depth pre-publish gate in `publish-pypi` immediately before `pypa/gh-action-pypi-publish`. | `test_matching_synthetic_tag_passes_pyproject_and_wheel`, `test_mismatched_or_invalid_synthetic_tags_fail`, `test_mismatched_wheel_metadata_against_pyproject_fails`, `test_multiple_wheels_in_wheelhouse_fail`, `test_cli_matching_and_mismatched_synthetic_tags` | **PASS** |
| Keep existing wheel build and three-platform installed-wheel tests | Preserved `build-wheel`, 3-OS `test-wheel` (`ubuntu-latest`, `macos-latest`, `windows-latest`), and `publish-pypi` OIDC configuration; extended `test-wheel` to also run `tests/test_release_workflow.py` and wired `scripts/validate_release_tag.py` into `ci.yml` Ruff + Mypy gates. | `test_release_workflow_pins_all_actions_to_full_commit_shas_and_gates_publish` + PR GitHub Actions `Release` & `CI` workflows | **PASS** |

---

## 2. Notes & Pre-Existing Observation

- No Git tag was created or pushed; workflow is verified on a PR branch with synthetic tag tests.
- Pre-existing version note (out of scope for `#64`): `pyproject.toml` declares `version = "1.6.3"` while `src/isaac/__init__.py` (`__version__ = "1.6.2"`), `terminal_ui.py`, and `web_app.py` still display `1.6.2`. Left untouched to keep `#64` strictly scoped to release workflow pinning and tag validation.
