"""Inspect release sources/archives and optionally verify original-source hashes."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "docs", "configs", "examples", "scripts", "tests", ".github")
SOURCE_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements.txt",
    "MANIFEST.in",
    "module_map.json",
    ".gitignore",
    ".env.example",
)
FORBIDDEN_PARTS = {
    ".env",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    "node_modules",
    ".venv",
    ".test_runtime_tmp",
}
FORBIDDEN_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors", ".log", ".pyc", ".pyo"}
PERSONAL_PATH = re.compile(
    r"(?i)(?:[a-z]:[/\\](?:users|documents and settings)[/\\][^/\\\s]+[/\\]|/(?:home|data)/[^/\s]+/models/)"
)
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b")


def inspect(name, data):
    path = Path(name)
    if any(part in FORBIDDEN_PARTS for part in path.parts) or path.suffix in FORBIDDEN_SUFFIXES:
        return [f"forbidden release file: {name}"]
    if len(data) > 5 * 1024 * 1024:
        return [f"file exceeds 5 MiB: {name}"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"unexpected binary file: {name}"]
    issues = []
    if PERSONAL_PATH.search(text):
        issues.append(f"personal absolute path: {name}")
    if SECRET.search(text):
        issues.append(f"possible credential: {name}")
    return issues


def ignored_package_files():
    """Catch package files omitted by Git before a source-only checkout loses them."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        return []
    candidates = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/evidenceloop").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    if not candidates:
        return []
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=ROOT,
        input="\0".join(candidates) + "\0",
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot verify Git ignore rules: " + result.stderr.strip())
    return [f"package file excluded by Git: {name}" for name in result.stdout.split("\0") if name]


def archive_entries(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    yield name, archive.read(name)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for entry in archive.getmembers():
                if entry.isfile():
                    yield entry.name, archive.extractfile(entry).read()
                elif entry.issym() or entry.islnk():
                    raise ValueError(f"archive links are not allowed: {entry.name}")
    else:
        raise ValueError(f"unsupported artifact: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", nargs="+", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="verify original files against docs/source_manifest.json; read only",
    )
    args = parser.parse_args()
    issues, count = [], 0
    if args.artifact:
        entries = (item for archive in args.artifact for item in archive_entries(archive))
    else:
        issues.extend(ignored_package_files())
        paths = [ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file()]
        paths += [
            p
            for folder in SOURCE_DIRS
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        ]
        paths += [p for p in ROOT.glob(".env*") if p.name != ".env.example"]
        entries = ((p.relative_to(ROOT).as_posix(), p.read_bytes()) for p in paths)
    for name, data in entries:
        count += 1
        issues.extend(inspect(name, data))
    if args.source_root:
        origin = args.source_root.resolve()
        for entry in json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8")):
            source = (origin / entry["old"]).resolve()
            if origin not in source.parents:
                issues.append("source manifest contains an unsafe path")
            elif (
                not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]
            ):
                issues.append(f"original source changed: {entry['old']}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(
        f"Release audit passed: {count} files"
        + ("; original source hashes match" if args.source_root else "")
    )


if __name__ == "__main__":
    main()
