"""Verify public Alpha dashboard bundle is copied from the canonical generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


INDEX_HTML = "INDEX.html"
PUBLIC_INDEX_HTML = "index.html"
SUMMARY_JSON = "BEGINNER_RESEARCH_DASHBOARD_SUMMARY.json"
MANIFEST_JSON = "DASHBOARD_BUNDLE_MANIFEST.json"
CANONICAL_GENERATOR_MODULE = "alpha_research_ops.dashboard.build_dashboard_bundle"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Alpha generated dashboard bundle matches public repo files."
    )
    parser.add_argument("--alpha-bundle", required=True, type=Path)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument(
        "--alpha-focused-test-command",
        default="",
        help="Optional command that must pass before deploy verification succeeds.",
    )
    args = parser.parse_args(argv)

    errors = verify_bundle_sync(
        alpha_bundle=args.alpha_bundle,
        public_root=args.public_root,
        alpha_focused_test_command=args.alpha_focused_test_command,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("dashboard_bundle_sync: PASS")
    return 0


def verify_bundle_sync(
    *,
    alpha_bundle: Path,
    public_root: Path,
    alpha_focused_test_command: str = "",
) -> list[str]:
    alpha_bundle = alpha_bundle.resolve()
    public_root = public_root.resolve()
    errors: list[str] = []

    alpha_html = alpha_bundle / INDEX_HTML
    alpha_summary = alpha_bundle / SUMMARY_JSON
    alpha_manifest_path = alpha_bundle / MANIFEST_JSON
    public_html = public_root / PUBLIC_INDEX_HTML
    public_summary = public_root / SUMMARY_JSON
    public_manifest_path = public_root / MANIFEST_JSON

    required_paths = [
        ("Alpha generated HTML", alpha_html),
        ("Alpha generated summary JSON", alpha_summary),
        ("Alpha generated manifest", alpha_manifest_path),
        ("public HTML", public_html),
        ("public summary JSON", public_summary),
        ("public manifest", public_manifest_path),
    ]
    for label, path in required_paths:
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
    if errors:
        return errors

    alpha_manifest = _load_json(alpha_manifest_path, errors, "Alpha manifest")
    public_manifest = _load_json(public_manifest_path, errors, "public manifest")
    if errors:
        return errors

    _verify_policy(alpha_manifest, errors, "Alpha manifest")
    _verify_policy(public_manifest, errors, "public manifest")

    alpha_html_hash = _sha256(alpha_html)
    alpha_summary_hash = _sha256(alpha_summary)
    public_html_hash = _sha256(public_html)
    public_summary_hash = _sha256(public_summary)

    if alpha_html_hash != public_html_hash:
        errors.append(
            "public HTML differs from Alpha generated HTML "
            f"({public_html_hash} != {alpha_html_hash})"
        )
    if alpha_summary_hash != public_summary_hash:
        errors.append(
            "public summary JSON differs from Alpha generated summary JSON "
            f"({public_summary_hash} != {alpha_summary_hash})"
        )
    if _sha256(alpha_manifest_path) != _sha256(public_manifest_path):
        errors.append("public manifest differs from Alpha generated manifest")

    if alpha_manifest != public_manifest:
        errors.append("public manifest values do not match Alpha manifest values")

    expected_hashes = {
        "index_html_sha256": alpha_html_hash,
        "summary_json_sha256": alpha_summary_hash,
    }
    for key, expected in expected_hashes.items():
        if alpha_manifest.get(key) != expected:
            errors.append(f"Alpha manifest {key} does not match Alpha file hash")
        if public_manifest.get(key) != expected:
            errors.append(f"public manifest {key} does not match Alpha file hash")

    embedded_hashes = _embedded_hashes(alpha_html)
    for key, expected in embedded_hashes.items():
        if alpha_manifest.get(key) != expected:
            errors.append(f"Alpha manifest {key} does not match Alpha HTML")
        if public_manifest.get(key) != expected:
            errors.append(f"public manifest {key} does not match Alpha HTML")

    schema_versions = public_manifest.get("schema_versions")
    if not isinstance(schema_versions, dict):
        errors.append("public manifest schema_versions missing or invalid")
    else:
        for schema_name in [
            "alpha_research_strategy_batch",
            "alpha_research_strategy_review_patch",
            "alpha_research_strategy_batch_result",
        ]:
            if not schema_versions.get(schema_name):
                errors.append(f"public manifest schema version missing: {schema_name}")

    if alpha_focused_test_command:
        completed = subprocess.run(
            alpha_focused_test_command,
            shell=True,
            cwd=alpha_bundle,
            check=False,
        )
        if completed.returncode != 0:
            errors.append(
                "Alpha focused dashboard test command failed; deploy gate blocked"
            )

    return errors


def _verify_policy(
    manifest: dict[str, Any], errors: list[str], label: str
) -> None:
    if manifest.get("generator_repo") != "Alpha Research Ops Lab":
        errors.append(f"{label} generator_repo is not Alpha Research Ops Lab")
    if manifest.get("generator_module") != CANONICAL_GENERATOR_MODULE:
        errors.append(f"{label} generator_module is not canonical")
    policy = manifest.get("public_bundle_policy")
    if not isinstance(policy, dict):
        errors.append(f"{label} public_bundle_policy missing or invalid")
        return
    if policy.get("generated_from_alpha_canonical") is not True:
        errors.append(f"{label} generated_from_alpha_canonical is not true")
    if policy.get("manual_public_html_edit_allowed") is not False:
        errors.append(f"{label} manual_public_html_edit_allowed is not false")


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{label} is malformed JSON: line {exc.lineno}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embedded_hashes(path: Path) -> dict[str, str]:
    import re

    html = path.read_text(encoding="utf-8")
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.S | re.I))
    scripts = []
    for attrs, body in re.findall(
        r"<script([^>]*)>(.*?)</script>", html, flags=re.S | re.I
    ):
        if 'type="application/json"' not in attrs:
            scripts.append(body)
    js = "\n".join(scripts)
    return {
        "embedded_css_sha256": hashlib.sha256(css.encode("utf-8")).hexdigest(),
        "embedded_js_sha256": hashlib.sha256(js.encode("utf-8")).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
