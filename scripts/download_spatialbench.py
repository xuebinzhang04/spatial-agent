#!/usr/bin/env python3
"""Mirror lmms-eval datasets into the shared HF cache and link them by benchmark."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1] / "03_SpatialBench"
DATA_ROOT = ROOT / "benchmarks"
CACHE = DATA_ROOT / ".hf_cache"
DATASETS = {
    "Point-Bench": "PointArena/pointarena-data",
    "RefSpatial": "BAAI/RefSpatial-Bench",
    "Where2Place": "FlagEval/Where2Place",
    "BLINK": "BLINK-Benchmark/BLINK",
    "CV-Bench": "nyu-visionx/CV-Bench",
    "ERQA": "FlagEval/ERQA",
    "EmbSpatial": "FlagEval/EmbSpatial-Bench",
    "MindCube": "oscarqjh/MindCube_lmmseval",
    "SAT": "nv-njb/SAT",
    "RoboSpatial-Home": "chanhee-luke/RoboSpatial-Home",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--refresh", action="store_true", help="Resolve a new upstream revision instead of the recorded one")
    args = parser.parse_args()
    unknown = set(args.names) - DATASETS.keys()
    if unknown:
        parser.error(f"unknown datasets: {', '.join(sorted(unknown))}")
    os.environ["HF_HOME"] = str(CACHE)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name in args.names or DATASETS:
        repo = DATASETS[name]
        target = DATA_ROOT / name
        target.mkdir(exist_ok=True)
        manifest_path = target / "download_manifest.json"
        recorded = json.loads(manifest_path.read_text()) if manifest_path.exists() and not args.refresh else {}
        revision = recorded.get("revision") or api.dataset_info(repo).sha
        print(f"Downloading {name}: {repo}@{revision}", flush=True)
        snapshot = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, cache_dir=CACHE / "hub", max_workers=4))
        link = target / "dataset"
        if link.is_symlink() and link.resolve() != snapshot:
            link.unlink()
        if not link.exists():
            link.symlink_to(snapshot, target_is_directory=True)
        manifest = {"repo": repo, "revision": revision, "snapshot": str(snapshot), "status": "complete"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Complete {name}: {snapshot}", flush=True)
    for alias, task in (("RoboSpatial-Poi", "robo_spatial_context"), ("RoboSpatial-VQ", "robo_spatial_configuration,robo_spatial_compatibility")):
        target = DATA_ROOT / alias
        target.mkdir(exist_ok=True)
        link = target / "dataset"
        shared_link = Path("../RoboSpatial-Home/dataset")
        if link.is_symlink() and link.resolve() != (target / shared_link).resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(shared_link, target_is_directory=True)
        (target / "README.md").write_text(
            f"# {alias}\n\nThe dataset is shared with `../RoboSpatial-Home/dataset`; no bytes are duplicated.\n"
            f"lmms-eval tasks: `{task}`.\n"
        )
        shared = json.loads((DATA_ROOT / "RoboSpatial-Home" / "download_manifest.json").read_text())
        (target / "download_manifest.json").write_text(
            json.dumps({"repo": shared["repo"], "revision": shared["revision"], "shared_with": "RoboSpatial-Home/dataset", "tasks": task.split(","), "status": "complete"}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
