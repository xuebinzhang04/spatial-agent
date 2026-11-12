#!/usr/bin/env python3
"""Convert lmms-eval OpenEQA sample JSONL into official judge input JSON."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path, help="*_samples_openeqa_predictions.jsonl")
    parser.add_argument("output", type=Path, help="Official OpenEQA results JSON")
    args = parser.parse_args()
    rows = []
    with args.samples.open() as handle:
        for line in handle:
            sample = json.loads(line)
            record = sample["openeqa_response_rate"]
            rows.append({"question_id": record["question_id"], "answer": record["answer"]})
    if len({row["question_id"] for row in rows}) != len(rows):
        raise SystemExit("duplicate question_id in sample log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
