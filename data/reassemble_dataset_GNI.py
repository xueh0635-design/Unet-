#!/usr/bin/env python3
"""Reassemble dataset_GNI.mat from numbered 90 MiB parts and verify SHA-256."""

from pathlib import Path
import hashlib
import sys


def main() -> int:
    base = Path(__file__).resolve().parent / "dataset_GNI_parts"
    output = Path(__file__).resolve().parent / "dataset_GNI.mat"
    parts = sorted(base.glob("dataset_GNI.mat.part-*") )
    if not parts:
        print(f"No parts found in {base}", file=sys.stderr)
        return 1

    digest = hashlib.sha256()
    with output.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                while block := src.read(1024 * 1024):
                    dst.write(block)
                    digest.update(block)

    expected = (base / "dataset_GNI.mat.sha256").read_text().split()[0]
    actual = digest.hexdigest()
    if actual != expected:
        print(f"SHA-256 mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 2
    print(f"Reassembled {output} ({output.stat().st_size} bytes)")
    print(f"SHA-256 OK: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
