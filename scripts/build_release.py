"""Build distributable artifacts into a fresh, explicitly selected directory."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-isolation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Use an output directory outside the project")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "build", "--outdir", str(output)]
    if args.no_isolation:
        command.append("--no-isolation")
    subprocess.run(command, cwd=root, check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/audit_release.py"),
            "--artifact",
            *map(str, sorted(output.iterdir())),
        ],
        check=True,
    )
    print(output)


if __name__ == "__main__":
    main()
