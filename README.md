# Alpha Research Ops Public Dashboard

This repo is the public deployment target for the Alpha Research Ops dashboard.

## Source Of Truth

- Canonical generator: `D:\projects\Alpha Research Ops Lab\alpha_research_ops\beginner_research_dashboard.py`
- Generated public bundle files:
  - `index.html`
  - `BEGINNER_RESEARCH_DASHBOARD_SUMMARY.json`
  - `DASHBOARD_BUNDLE_MANIFEST.json`
- Public runner, worker, tests, request/result JSON, and deployment config may be edited here.
- Do not directly patch dashboard UI behavior in `index.html`. Change the Alpha generator first, regenerate the bundle, then copy generated files here.

## Pre-Deploy Gate

Run in Alpha first:

```text
py -m py_compile alpha_research_ops\beginner_research_dashboard.py alpha_research_ops\cli.py
py -m pytest -q tests\test_beginner_research_dashboard.py
py -m pytest -q
py -m alpha_research_ops.cli beginner-dashboard-generate --repo-root . --output-dir artifacts/beginner_research_dashboard\<run_id>
py -m json.tool artifacts\beginner_research_dashboard\<run_id>\BEGINNER_RESEARCH_DASHBOARD_SUMMARY.json
py -m json.tool artifacts\beginner_research_dashboard\<run_id>\DASHBOARD_BUNDLE_MANIFEST.json
```

Copy only the generated bundle files into this public repo, then run:

```text
py -m py_compile scripts\external_strategy_batch.py
py -m pytest -q tests\test_external_strategy_batch.py
node --check workers\strategy-dispatch-worker.js
py scripts\verify_dashboard_bundle_sync.py --alpha-bundle "D:\projects\Alpha Research Ops Lab\artifacts\beginner_research_dashboard\<run_id>" --public-root .
```

The sync verifier must pass before public commit, push, or deploy. It fails if public HTML, summary JSON, or manifest values drift from the Alpha generated bundle.
