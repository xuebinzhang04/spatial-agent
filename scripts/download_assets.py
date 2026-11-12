"""Download pinned official assets; rerunning resumes interrupted downloads."""
import concurrent.futures
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from huggingface_hub import HfApi, snapshot_download

ASSETS = [
    ("Qwen/Qwen3.8-27B", "model", "01_Models/Qwen3.8-27B", None),
    ("SimWorld-AI/SimWorld", "dataset", "02_Simulator/SimWorld", ["Base/Linux.zip"]),
    ("SimWorld-AI/DeliveryBench", "dataset", "04_AgentTask/SpatialAgentBench/DeliveryBench/assets/SimWorld-AI-DeliveryBench", ["Source/Linux/*"]),
    ("nyu-visionx/VSI-Bench", "dataset", "03_SpatialBench/benchmarks/VSI-Bench", None),
    ("MLL-Lab/MindCube", "dataset", "03_SpatialBench/benchmarks/MindCube", None),
    ("MLL-Lab/tos-data", "dataset", "03_SpatialBench/benchmarks/Theory-of-Space", None),
]

PINNED_REVISIONS = {
    "nyu-visionx/VSI-Bench": "bdcadb3fea447621a828a24911801faba3587c12",
    "MLL-Lab/MindCube": "9c941b46a6bd65b6914669ef7a579948fc9c8467",
    "MLL-Lab/tos-data": "923e0572597d7a01f8a25587fb003303928cd7ac",
}

def download(asset):
    repo, kind, directory, patterns = asset
    target = ROOT / directory
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "download_manifest.json"
    if manifest.exists():
        revision = json.loads(manifest.read_text())["revision"]
    else:
        revision = PINNED_REVISIONS.get(repo) or HfApi().repo_info(repo, repo_type=kind).sha
    record = dict(repo_id=repo, repo_type=kind, revision=revision,
                  allow_patterns=patterns, status="downloading")
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"START {repo} revision={revision} target={target}", flush=True)
    snapshot_download(repo, repo_type=kind, revision=revision,
                      local_dir=target, allow_patterns=patterns, max_workers=4)
    record["status"] = "complete"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"COMPLETE {repo}", flush=True)

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, a): a[0] for a in ASSETS}
        failed = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append(futures[future])
                print(f"FAILED {futures[future]}: {exc}", flush=True)
        if failed:
            raise SystemExit(1)
