"""Render an offline HTML view of real episode artifacts (no invented scores)."""
import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
root = args.run_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
esc = lambda value: html.escape(str(value))
parts = ["<!doctype html><meta charset='utf-8'><title>DeliveryBench test</title>",
         "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f6f8fb;color:#182337}pre{white-space:pre-wrap;background:white;padding:16px;border:1px solid #dde3eb}img{max-width:100%;max-height:500px}a{color:#235ab2}figure{background:white;padding:16px}</style>",
         f"<h1>DeliveryBench · {esc(root.name)}</h1>",
         "<p>Single-map full lifecycle result; inspect validation and scheduler status. This is not a multi-map aggregate.</p>" if "-full-" in root.name else "<p>Integration smoke test. Partial or failed episodes are not full benchmark scores.</p>"]
for filename in ("result_validation.json", "job_status.json", "episode_status.json", "experiment_config.json", "models.json"):
    path = root / filename
    if path.exists():
        parts += [f"<h2>{esc(filename)}</h2><pre>{esc(path.read_text())}</pre>"]
reports = sorted((root / "reports").glob("run_report_agent*.json"))
if not reports:
    parts.append("<p>No episode report yet. See job status and service logs.</p>")
for path in reports:
    data = json.loads(path.read_text())
    parts.append(f"<h2>{esc(path.name)}</h2>")
    for key in ("meta", "money", "orders"):
        if key in data:
            parts.append(f"<h3>{esc(key)}</h3><pre>{esc(json.dumps(data[key], ensure_ascii=False, indent=2))}</pre>")
parts.append("<h2>Artifacts</h2><ul>")
for path in sorted(root.rglob("*")):
    if path.is_file() and path.suffix in {".log", ".csv", ".json", ".txt", ".sh"}:
        rel = path.relative_to(root).as_posix()
        parts.append(f"<li><a href='{quote(rel)}'>{esc(rel)}</a></li>")
parts.append("</ul><h2>Observation images</h2>")
images = sorted((root / "trajectories").rglob("*.png"))
images += sorted((root / "trajectories").rglob("*.jpg"))
for path in images[:30]:
    rel = path.relative_to(root).as_posix()
    parts.append(f"<figure><img loading='lazy' src='{quote(rel)}'><figcaption>{esc(rel)}</figcaption></figure>")
if not images:
    parts.append("<p>No captured observations yet.</p>")
(root / "demo.html").write_text("\n".join(parts), encoding="utf-8")
print(root / "demo.html")
