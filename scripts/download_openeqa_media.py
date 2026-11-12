#!/usr/bin/env python3
"""Download the publicly mirrored OpenEQA episode histories (third-party mirror)."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1] / "03_SpatialBench"
TARGET = ROOT / "benchmarks" / "OpenEQA"
CACHE = ROOT / "benchmarks" / ".hf_cache" / "hub"
REPO = "AIGeeksGroup/OpenEQA"
REVISION = "9b127cf58e564fe2468bb140c10b20dd937a97f8"


def main() -> None:
    snapshot = Path(
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=CACHE,
            allow_patterns=["hm3d-v0/*.tar", "scannet-v0/*.tar", "README.md"],
            max_workers=4,
        )
    )
    link = TARGET / "media"
    if link.is_symlink() and link.resolve() != snapshot:
        link.unlink()
    if not link.exists():
        link.symlink_to(snapshot, target_is_directory=True)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "status": "complete",
        "note": "Third-party mirror of OpenEQA RGB episode histories; official annotations and judge remain in source/",
    }
    (TARGET / "media_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Complete OpenEQA media:", snapshot)


if __name__ == "__main__":
    main()
