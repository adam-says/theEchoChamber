#!/usr/bin/env python3
"""Reconstruct the recorded runtime-input comparison from its frozen manifest.

This is a new, explicit verifier for the recorded inline check. It does not
claim to be the original inline command or reproduce its execution timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "0c5efa23273fa1bdc13473d142681739c71094af92f32f521fdd21dd24e9572f"
EXPECTED_MAIN_REVISION = "c16c0eee0174c7c016d05a5ce617186a3c4c42d8"
EXPECTED_ISOLATED_REVISION = "40a77d924a8f4fae45498d86d3667fe1860ecbd9"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def newline_normalized(data: bytes) -> bytes:
    # The historical comparison allowed only CRLF-to-LF conversion.
    return data.replace(b"\r\n", b"\n")


def run(args: argparse.Namespace) -> dict:
    task_root = args.task_root.resolve(strict=True)
    main_root = args.main_root.resolve(strict=True)
    isolated_root = args.isolated_root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "manifest SHA-256 differs from the reviewed frozen manifest: "
            f"{manifest_sha}"
        )
    manifest = json.loads(manifest_bytes)
    expected_files = manifest.get("source_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("manifest has no source_sha256 file map")

    main_revision = git_head(main_root)
    isolated_revision = git_head(isolated_root)
    if main_revision != EXPECTED_MAIN_REVISION:
        raise ValueError(f"main HEAD mismatch: {main_revision}")
    if isolated_revision != EXPECTED_ISOLATED_REVISION:
        raise ValueError(f"isolated HEAD mismatch: {isolated_revision}")

    records = []
    errors = []
    for relative, expected_sha in sorted(expected_files.items()):
        if relative.replace("\\", "/").lower().startswith("docs/"):
            continue
        main_path = main_root / Path(relative)
        isolated_path = isolated_root / Path(relative)
        record = {
            "path": relative,
            "unit_test_input_sha256": expected_sha,
            "isolated_path": str(isolated_path),
            "main_path": str(main_path),
        }
        try:
            isolated_bytes = isolated_path.read_bytes()
            main_bytes = main_path.read_bytes()
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{relative}: {record['error']}")
            records.append(record)
            continue

        isolated_sha = sha256_bytes(isolated_bytes)
        main_sha = sha256_bytes(main_bytes)
        tested_unchanged = isolated_sha == expected_sha
        exact_equal = main_bytes == isolated_bytes
        normalized_equal = newline_normalized(main_bytes) == newline_normalized(
            isolated_bytes
        )
        line_endings_only = normalized_equal and not exact_equal
        record.update(
            {
                "isolated_sha256": isolated_sha,
                "main_sha256": main_sha,
                "tested_input_unchanged": tested_unchanged,
                "main_equivalent_to_tested_input": normalized_equal,
                "line_endings_only": line_endings_only,
            }
        )
        if not tested_unchanged:
            errors.append(f"{relative}: isolated input differs from manifest SHA")
        if not normalized_equal:
            errors.append(f"{relative}: main file differs beyond CRLF-to-LF")
        records.append(record)

    result = {
        "schema": "echo-chamber-runtime-input-proof-reconstruction-v1",
        "reconstruction_notice": (
            "This is a newly run reconstruction of the recorded inline verifier, "
            "using the same frozen manifest and comparison rule. It is not the "
            "original inline execution record."
        ),
        "task_root": str(task_root),
        "suite_manifest": str(manifest_path),
        "suite_manifest_sha256": manifest_sha,
        "manifest_scope": (
            "Manifest-listed unit-suite code/tests/config/model/UI inputs outside "
            "docs/. Documentation files are excluded."
        ),
        "main_commit": main_revision,
        "isolated_commit": isolated_revision,
        "source_inputs_checked": len(records),
        "all_inputs_verified": not errors and bool(records),
        "errors": errors,
        "line_endings_only_count": sum(r.get("line_endings_only", False) for r in records),
        "timing_acceptance_claim": False,
        "hardware_claim": False,
        "inputs": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--isolated-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_inputs_checked": result["source_inputs_checked"],
                "line_endings_only_count": result["line_endings_only_count"],
                "all_inputs_verified": result["all_inputs_verified"],
                "errors": len(result["errors"]),
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_inputs_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
