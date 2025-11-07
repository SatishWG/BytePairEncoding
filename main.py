from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/AI4Bharat/sangraha-internet-archive-download.git"
WORKSPACE_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_DIR = WORKSPACE_ROOT / "third_party"
REPO_DIR = THIRD_PARTY_DIR / "sangraha-internet-archive-download"
OUTPUT_TEXT = WORKSPACE_ROOT / "input_marathi.txt"
DOWNLOAD_STAGING = WORKSPACE_ROOT / "_marathi_download"


def ensure_repo() -> None:
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    if REPO_DIR.exists():
        return

    subprocess.run(
        ["git", "clone", REPO_URL, str(REPO_DIR)],
        check=True,
    )


def run_download(language: str) -> None:
    script_path = REPO_DIR / "pipeline" / "single_machine_download.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Expected single_machine_download.py inside cloned repository"
        )

    DOWNLOAD_STAGING.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--language",
        language,
        "--output-folder",
        str(DOWNLOAD_STAGING),
    ]

    subprocess.run(cmd, check=True)


def collect_text() -> None:
    text_files = sorted(DOWNLOAD_STAGING.glob("*.txt"))
    if not text_files:
        raise RuntimeError(
            "No text files were downloaded. Inspect the staging directory for details."
        )

    with OUTPUT_TEXT.open("w", encoding="utf-8") as sink:
        for path in text_files:
            sink.write(path.read_text(encoding="utf-8"))
            sink.write("\n")


def main() -> None:
    ensure_repo()
    run_download("marathi")
    collect_text()


if __name__ == "__main__":
    main()
