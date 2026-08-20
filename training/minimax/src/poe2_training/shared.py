"""Shared authenticated-report, hashing, and runtime provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class SharedArtifactError(ValueError):
    """Raised when a shared create-only artifact operation fails."""


_REPORT_FILENAMES = {"COMPLETE", "INCOMPLETE", "report.json"}


def _valid_attachment_name(filename: object) -> bool:
    return (isinstance(filename, str) and bool(filename) and
            Path(filename).name == filename and filename not in _REPORT_FILENAMES)


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    resolved = Path(path)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SharedArtifactError(f"could not hash {resolved}: {error}") from error
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize deterministic, human-readable JSON used by authenticated reports."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_write_bytes(path: Path, contents: bytes, *, replace: bool = False) -> None:
    """Durably write bytes either create-only or through an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    mode = "wb" if replace else "xb"
    try:
        with temporary.open(mode) as destination:
            destination.write(contents)
            destination.flush()
            os.fsync(destination.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # link(2) installs the final name atomically and refuses an existing target.
            os.link(temporary, path)
            temporary.unlink()
    except OSError:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def reserve_report_directory(directory: Path | str, schema: str, *,
                             error_type: type[ValueError] = SharedArtifactError) -> Path:
    """Reserve a create-only JSON-report directory and install INCOMPLETE."""
    root = Path(directory).resolve()
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        atomic_write_bytes(root / "INCOMPLETE", f"{schema}\n".encode())
    except OSError as error:
        raise error_type(f"could not reserve {root}: {error}") from error
    return root


def write_report_attachment(
    directory: Path | str,
    filename: str,
    contents: bytes,
    *,
    media_type: str,
    error_type: type[ValueError] = SharedArtifactError,
) -> dict[str, Any]:
    """Create one report-owned file and return its authentication metadata."""
    if not _valid_attachment_name(filename):
        raise error_type(f"unsafe report attachment name: {filename!r}")
    root = Path(directory).resolve()
    try:
        atomic_write_bytes(root / filename, contents)
    except OSError as error:
        raise error_type(f"could not write report attachment {root / filename}: {error}") from error
    return {
        "path": filename,
        "media_type": media_type,
        "bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def authenticate_report_attachments(
    directory: Path | str,
    report: dict[str, Any],
    *,
    error_type: type[ValueError] = SharedArtifactError,
) -> set[str]:
    """Authenticate every report attachment and return its exact filenames."""
    root = Path(directory).resolve()
    attachments = report.get("attachments", {})
    if not isinstance(attachments, dict):
        raise error_type(f"report attachments are malformed: {root}")
    paths: set[str] = set()
    for name, metadata in attachments.items():
        if (not isinstance(name, str) or not name or not isinstance(metadata, dict) or
                set(metadata) != {"path", "media_type", "bytes", "sha256"}):
            raise error_type(f"report attachment metadata is malformed: {root}")
        path = metadata["path"]
        digest = metadata["sha256"]
        if (not _valid_attachment_name(path) or path in paths or
                not isinstance(metadata["media_type"], str) or
                not isinstance(metadata["bytes"], int) or metadata["bytes"] < 0 or
                not isinstance(digest, str) or len(digest) != 64):
            raise error_type(f"report attachment metadata is malformed: {root}")
        try:
            bytes.fromhex(digest)
            size = (root / path).stat().st_size
            actual = sha256_file(root / path)
        except (OSError, ValueError, SharedArtifactError) as error:
            raise error_type(f"could not authenticate report attachment {root / path}: {error}") from error
        if size != metadata["bytes"] or actual != digest:
            raise error_type(f"report attachment differs from metadata: {root / path}")
        paths.add(path)
    return paths


def complete_json_report(directory: Path | str, schema: str,
                         report: dict[str, Any], *,
                         error_type: type[ValueError] = SharedArtifactError) -> str:
    """Commit report.json and COMPLETE last, preserving create-only semantics."""
    root = Path(directory).resolve()
    report_bytes = canonical_json(report)
    digest = hashlib.sha256(report_bytes).hexdigest()
    try:
        atomic_write_bytes(root / "report.json", report_bytes)
        atomic_write_bytes(
            root / "COMPLETE",
            f"{schema}\nreport_sha256={digest}\n".encode(),
        )
        (root / "INCOMPLETE").unlink()
    except OSError as error:
        raise error_type(f"could not complete {root}: {error}") from error
    return digest


def open_json_report(
    directory: Path | str,
    *,
    schema: str,
    schema_version: int = 1,
    error_type: type[ValueError] = SharedArtifactError,
) -> tuple[dict[str, Any], str]:
    """Authenticate a standard report.json/COMPLETE create-only artifact."""
    root = Path(directory).resolve()

    def fail(message: str, cause: Exception | None = None) -> None:
        error = error_type(message)
        if cause is None:
            raise error
        raise error from cause

    if not root.is_dir():
        fail(f"report directory does not exist: {root}")
    if (root / "INCOMPLETE").exists():
        fail(f"report is incomplete: {root}")
    complete = root / "COMPLETE"
    report_path = root / "report.json"
    if not complete.is_file() or not report_path.is_file():
        fail(f"report lacks COMPLETE or report.json: {root}")
    try:
        lines = complete.read_text(encoding="utf-8").splitlines()
        report = json.loads(report_path.read_bytes())
        actual_digest = sha256_file(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SharedArtifactError) as error:
        fail(f"could not read report {root}: {error}", error)
    if lines[:1] != [schema] or len(lines) != 2:
        fail(f"report COMPLETE marker is malformed: {root}")
    name, separator, expected_digest = lines[1].partition("=")
    if (name != "report_sha256" or not separator or len(expected_digest) != 64):
        fail(f"report COMPLETE digest is malformed: {root}")
    try:
        bytes.fromhex(expected_digest)
    except ValueError as error:
        fail(f"report COMPLETE digest is not hexadecimal: {root}", error)
    if actual_digest != expected_digest:
        fail(f"report digest differs from COMPLETE: {root}")
    if (not isinstance(report, dict) or report.get("schema") != schema or
            report.get("schema_version") != schema_version):
        fail(f"report schema is unsupported: {root}")
    authenticate_report_attachments(root, report, error_type=error_type)
    return report, actual_digest


def training_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def training_source_digest() -> str:
    """Digest the isolated training project sources and lock inputs."""
    root = training_project_root()
    fixed = (root / ".python-version", root / "pyproject.toml", root / "uv.lock")
    sources = sorted((root / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in (*fixed, *sources):
        relative = path.relative_to(root).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def find_repository_root(start: Path | None = None) -> Path | None:
    current = (start or training_project_root()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_provenance(repository: Path | None = None) -> dict[str, Any]:
    """Return full commit and dirty state for the enclosing repository."""
    root = repository or find_repository_root()
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SharedArtifactError(f"could not inspect Git provenance: {error}") from error
    if len(commit) != 40:
        raise SharedArtifactError("Git did not return a full commit ID")
    return {"commit": commit, "dirty": bool(status)}


def runtime_provenance(torch: Any, device: Any, *, float_dtype: str = "float64") -> dict[str, Any]:
    """Return the stable runtime fields embedded in experiment reports."""
    device_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                   else platform.processor() or platform.machine())
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_type": device.type,
        "device_name": device_name,
        "float_dtype": float_dtype,
    }
