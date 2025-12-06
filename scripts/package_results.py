#!/usr/bin/env python3
"""Bundle workshop audio artefacts into results.zip."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
ARCHIVE_PATH = ROOT / "results.zip"


def main() -> None:
    wav_names = [
        "tacotron.wav",
        "xtts_v2_natural.wav",
        "xtts_v2_happy.wav",
        "xtts_v2_sad.wav",
    ]

    ARCHIVE_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in wav_names:
            wav_path = OUTPUT_DIR / name
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing expected audio file: {wav_path}")
            zf.write(wav_path, arcname=name)

    print(f"Created {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()

