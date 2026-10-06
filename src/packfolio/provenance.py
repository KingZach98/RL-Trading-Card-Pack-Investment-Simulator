"""Process and repository metadata shared by training and evaluation manifests.

Both `train.py` and `evaluate.py` record the git commit and installed
dependency set that produced their outputs. Keeping the lookups here means a
training manifest and an evaluation manifest always describe provenance the
same way.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re
import subprocess


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def git_metadata(repository_root: str | Path | None = None) -> tuple[str, bool]:
    """Return the current commit hash and whether the working tree is dirty."""
    root = Path(repository_root) if repository_root is not None else _REPOSITORY_ROOT
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return commit_result.stdout.strip(), bool(status_result.stdout.strip())


def installed_dependencies() -> list[dict[str, str]]:
    """Return the sorted, deduplicated set of installed package name/version pairs."""
    packages: dict[str, dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized_name = re.sub(r"[-_.]+", "-", name).casefold()
        package = {"name": name, "version": distribution.version}
        existing = packages.get(normalized_name)
        if existing is not None and existing["version"] != package["version"]:
            raise RuntimeError(f"multiple installed versions found for {name}")
        packages[normalized_name] = package
    return sorted(packages.values(), key=lambda package: package["name"].casefold())


__all__ = ["git_metadata", "installed_dependencies"]
