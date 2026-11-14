from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from .common import atomic_write_json, expand_path, load_config, sha256_file
except ImportError:  # direct script execution
    from common import atomic_write_json, expand_path, load_config, sha256_file


UNEXPANDED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
ENV_NAME = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def git_metadata(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "remote_origin": run("remote", "get-url", "origin"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _candidate_paths(spec: dict[str, Any], project_root: Path) -> list[Path]:
    values = spec.get("one_of", [spec.get("path")])
    return [expand_path(value, base=project_root) for value in values if isinstance(value, str)]


def check_resource(name: str, spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    kind = spec.get("kind", "file")
    required = bool(spec.get("required", True))
    raw_values = spec.get("one_of", [spec.get("path")])
    unresolved = [value for value in raw_values if isinstance(value, str) and UNEXPANDED_ENV.search(value)]
    environment_names = {
        match.group(1) or match.group(2)
        for value in raw_values
        if isinstance(value, str)
        for match in ENV_NAME.finditer(value)
    }
    missing_environment = sorted(
        variable for variable in environment_names if not os.environ.get(variable)
    )
    candidates = _candidate_paths(spec, project_root)
    result: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "required": required,
        "candidates": [str(path) for path in candidates],
        "status": "missing",
    }
    if unresolved:
        result["unexpanded_environment_variables"] = unresolved
    if missing_environment:
        # An empty variable expands to ``Path('.')`` and can otherwise make a
        # directory resource incorrectly pass by resolving to project_root.
        result["missing_or_empty_environment_variables"] = missing_environment
        return result

    if kind == "executable":
        executable = str(spec.get("path", name))
        resolved = shutil.which(os.path.expandvars(executable))
        if resolved:
            result.update(status="ready", resolved_path=resolved)
        return result

    predicate = {
        "file": Path.is_file,
        "directory": Path.is_dir,
        "git_repo": lambda path: path.is_dir() and (path / ".git").exists(),
    }.get(kind)
    if predicate is None:
        result.update(status="invalid_spec", error=f"unsupported resource kind: {kind}")
        return result
    for candidate in candidates:
        if predicate(candidate):
            result.update(status="ready", resolved_path=str(candidate))
            if kind == "file" and int(spec.get("expected_bytes", 0)) > 0:
                observed_bytes = candidate.stat().st_size
                expected_bytes = int(spec["expected_bytes"])
                result["observed_bytes"] = observed_bytes
                result["expected_bytes"] = expected_bytes
                if observed_bytes != expected_bytes:
                    result["status"] = "size_mismatch"
            expected_sha256 = str(spec.get("expected_sha256", "")).lower()
            if kind == "file" and (bool(spec.get("hash", False)) or expected_sha256):
                observed_sha256 = sha256_file(candidate)
                result["sha256"] = observed_sha256
                if expected_sha256:
                    result["expected_sha256"] = expected_sha256
                    if observed_sha256.lower() != expected_sha256:
                        result["status"] = "checksum_mismatch"
            if kind == "git_repo":
                result["git"] = git_metadata(candidate)
                expected_commit = spec.get("commit")
                if not expected_commit and spec.get("commit_env"):
                    expected_commit = os.environ.get(str(spec["commit_env"]))
                if not expected_commit:
                    result["status"] = "unpinned"
                    result["error"] = (
                        "a concrete commit must be frozen in the baseline config or its declared "
                        "commit_env"
                    )
                elif result["git"]["head"] != expected_commit:
                    result["status"] = "commit_mismatch"
                    result["expected_commit"] = expected_commit
            break
    return result


def audit_config(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(project_root).resolve()
    resources = config.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError(f"resources must be a mapping: {config_path}")
    checks = [check_resource(name, spec, root) for name, spec in resources.items()]
    blockers = [
        check["name"]
        for check in checks
        if check["required"] and check["status"] != "ready"
    ]
    return {
        "baseline": config.get("baseline", Path(config_path).stem),
        "config": str(Path(config_path).resolve()),
        "project_root": str(root),
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "resources": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external baseline code, weights and indexes")
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [audit_config(path, args.project_root) for path in args.config]
    result = {
        "status": "ready" if all(item["status"] == "ready" for item in reports) else "blocked",
        "baselines": reports,
    }
    atomic_write_json(args.output, result)
    print(f"{result['status']}: wrote {args.output}")
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
