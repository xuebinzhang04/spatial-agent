#!/usr/bin/env python3
"""Collect complete lmms-eval results using the 13-column SpatialBench mapping."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "03_SpatialBench" / "eval_configs" / "eval_matrix.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path, help="One or more lmms-eval *_results.json files")
    parser.add_argument("--openeqa-metrics", type=Path, help="Official OpenEQA *-metrics.json (1-5 LLM-Match scores)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    task_values = {}
    for path in args.results:
        report = json.loads(path.read_text())
        if report.get("model_name") not in ("Qwen3.8-27B", None):
            parser.error(f"unexpected model in {path}: {report.get('model_name')}")
        counts = report.get("n-samples", {})
        for task, metrics in report.get("results", {}).items():
            components = report.get("group_subtasks", {}).get(task) or [task]
            if not isinstance(components, list):
                components = [task]
            if any(
                component not in counts
                or counts[component].get("original") is None
                or counts[component].get("effective") != counts[component].get("original")
                for component in components
            ):
                continue  # smoke tests and partial runs are never reportable scores
            if task in task_values:
                parser.error(f"duplicate complete task {task}; provide one result per task")
            task_values[task] = metrics

    matrix = json.loads(MATRIX.read_text())
    benchmark_scores = {}
    pending = []
    for benchmark in matrix["benchmarks"]:
        if benchmark["name"] == "OpenEQA":
            if args.openeqa_metrics is None:
                pending.append("OpenEQA: requires official LLM-Match result")
            else:
                official = json.loads((ROOT / "03_SpatialBench" / "benchmarks" / "OpenEQA" / "source" / "data" / "open-eqa-v0.json").read_text())
                expected = {row["question_id"] for row in official}
                scores = json.loads(args.openeqa_metrics.read_text())
                if set(scores) != expected:
                    parser.error(f"OpenEQA judge file is incomplete: {len(scores)} scores for {len(expected)} questions")
                benchmark_scores["OpenEQA"] = sum((min(max(float(score), 1.0), 5.0) - 1.0) / 4.0 for score in scores.values()) / len(scores)
            continue
        task_scores = []
        for task in benchmark["tasks"]:
            metrics = task_values.get(task)
            metric_key = benchmark["metric"] + "," + benchmark.get("filter", "none")
            if metrics is None or metric_key not in metrics:
                pending.append(f"{benchmark['name']}: missing complete {task}/{metric_key}")
                break
            value = metrics[metric_key]
            if isinstance(value, dict):
                value = value.get("overall")
            if not isinstance(value, (int, float)):
                pending.append(f"{benchmark['name']}: nonnumeric {task}/{metric_key}")
                break
            task_scores.append(float(value))
        else:
            # RoboSpatial VQA uses the equal-weight mean of its two categories.
            benchmark_scores[benchmark["name"]] = sum(task_scores) / len(task_scores)
    output = {"model": matrix["model"], "scores": benchmark_scores, "pending": pending, "overall_avg": None}
    if len(benchmark_scores) == len(matrix["benchmarks"]):
        output["overall_avg"] = sum(benchmark_scores.values()) / len(benchmark_scores)
    payload = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
