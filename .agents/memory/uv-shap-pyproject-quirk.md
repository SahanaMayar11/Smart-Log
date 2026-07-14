---
name: uv/shap pyproject quirk
description: Replit's auto-generated pyproject.toml can misroute a package to the wrong index via [tool.uv.sources], making `uv add`/installLanguagePackages fail even when the package installs fine with pip.
---

Symptom: `uv add <pkg>` (or the package-management tool) fails to resolve a
package (e.g. `shap`) even though it is genuinely available on PyPI and
`pip install <pkg>` works without issue.

Cause observed: the auto-generated `pyproject.toml` had a
`[tool.uv.sources]` override mapping the package to a narrow custom index
(e.g. a `pytorch-cpu` index) that has no wheels for it, combined with a
resolver quirk that splits markers across all future Python versions.

**Why:** uv's resolver takes `[tool.uv.sources]` overrides literally; a
stray/incorrect override silently breaks resolution with a confusing error
that looks like the package doesn't exist.

**How to apply:** If a package install fails via the standard tool but the
package is known to exist, check `pyproject.toml` for a bad
`[tool.uv.sources]` entry for that package and remove it; also pin
`requires-python` to a narrow range if the resolver is fanning out across
versions. As a fallback, `pip install` directly inside the project's
`.pythonlibs` venv works even when `uv add` keeps failing — use it to
unblock, then add the package/version to `requirements.txt` for the record.
