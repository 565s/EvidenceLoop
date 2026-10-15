# Delivery verification

Verification date: 2026-10-15. Local environment: Windows, Python 3.12.7. All checks were offline with respect to model execution. No paid endpoint, weight download, training or paper experiment was run.

## Completed checks

- Existing suite: **273 passed, 4 skipped**. Skips require optional third-party installations. The old default pytest temporary directory had an access restriction, so verification used a fresh explicit `--basetemp`; no test was changed to address this environment issue.
- No test files were added or modified during this continuation: all **44 Python files** under the existing `tests/` tree match the initial continuation snapshot, including the frozen reference code.
- Ruff import/static checks and formatting checks cover `src` and `scripts`.
- **41 help routes** were invoked outside the project directory with credentials removed and network connections, client construction and model/plot imports blocked. Pipeline construction and help succeeded without creating output files.
- Canonical two-document example input validates. Exact matching on an explicitly formatted identical prediction export produces F1 = 1.0. This is a format check, not a model-quality claim.
- Wheel and source archive build successfully. Archive scans reject credential-like tokens, personal absolute paths, unexpected binaries, weights/logs/caches and files above 5 MiB.
- Original source-file hashes recorded in `docs/source_manifest.json` match the original project. Historical migration checks in the existing suite preserve source runs and validate resume/content mismatch behavior.

Independent wheel and editable installations both passed in a separate virtual environment, invoked from outside the checkout. The installed console entry, all modules, 41 help routes, lazy Pipeline construction, packaged prompts/relation maps and the baseline implementation fingerprint were checked. Existing local base dependencies were reused with `--no-deps`/`--no-index`; fresh dependency resolution across all extras was not performed. Model imports, client creation and network connections were blocked for the import/help checks.


## Git publication verification

The first GitHub run exposed a publication-only omission: the unanchored `data/` ignore rule excluded the 19 source files under `src/evidenceloop/data/`. Local tests originally saw these untracked files. Generated-directory ignore patterns are now anchored to the repository root, and the source release audit detects package files excluded by Git when a checkout and Git executable are available.

A separate export of the files eligible for the next commit was verified with the CI environment's empty API credential/endpoint variables: **273 passed, 4 skipped**, with no changes to the existing 44 test files. This also exposed and fixed empty-string versus absent-endpoint normalization in runtime/migrated checkpoint metadata. The fixes preserve inference and evaluation behavior. All 19 data-package files must be included in the next commit alongside these configuration/code changes. A fresh hosted GitHub run remains pending the user's push.


## Windows path compatibility

Windows CI also exposed a long-path versus 8.3 short-path mismatch in explicit efficiency-report run selection. Resolving an absolute path expanded its Windows alias, while automatic discovery preserved the supplied spelling. Explicit absolute run paths now keep the caller's spelling, matching discovery; relative paths retain their existing resolution.

The failure was reproduced locally with a real Windows 8.3 alias before applying the fix. The unchanged existing suite was then run from a publishable-file export with `TEMP`, `TMP` and `TMPDIR` pointed to that alias, and API settings empty as in CI: **273 passed, 4 skipped**. All 44 existing test files are unchanged. This verifies the local Windows path case; hosted CI must rerun after pushing the fix.

## Checks for maintainers

```sh
python -m pip install -e ".[dev]"
python -m pytest -q --basetemp .test_runtime_tmp/local
python -m ruff check src scripts
python -m ruff format --check src scripts
python scripts/audit_release.py
python scripts/build_release.py --output-dir ../evidenceloop-artifacts --no-isolation
```

The build output directory must be absent and outside the project. Omit `--no-isolation` if you want the build frontend to provision its own build environment. Dependencies and build tools must already be available for a fully offline build. The release auditor can also check original hashes with `--source-root ORIGINAL_PROJECT`; it reads the original files only.

Windows and Ubuntu CI is configured for Python 3.10 and 3.12. These hosted CI jobs have **not** been executed locally and are not claimed as a verified Linux result. CI installs packages, but its checks do not require API keys or model downloads.

## Remaining real-model validation

The reproduction guide contains commands for a small actual runtime check (`critic check`), a limited `run --limit 2`, separate GraphJudge/CS-FS scoring, LoRA training and third-party preflight/execution. These require the user's weights, adapters, dataset files, backend credentials and appropriate hardware/environments.

GPU BF16 loading/merging, real model output equivalence, third-party model execution, throughput, token usage, billing and paper table reproduction are **not validated by this offline delivery**. Keep that distinction when sharing the project. Unknown author, repository and custom adapter download information remain unfilled. The paper's additional unimplemented ablations are listed in `implementation-differences.md`.
