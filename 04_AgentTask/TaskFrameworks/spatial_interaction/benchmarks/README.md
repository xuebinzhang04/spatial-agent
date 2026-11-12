# Default Stage 1 splits

Each JSONL contains four public episode recipes, one for each supported task. Answers and simulator state are kept in separate evaluator artifacts.

| Split | Seeds | iTHOR layout in these four episodes |
| --- | --- | --- |
| Development | 42–45 | FloorPlan1 |
| Validation | 1000042–1000045 | FloorPlan2 |
| Held-out test | 2000042–2000045 | FloorPlan3 |

Shapes and all their distractors are assigned to splits by the hash of their rotation-equivalence class. The native scene snapshots are privileged backend assets under `spatial_interaction/environments/dynamic_memory/scene_states/`; they are not part of observations. The generator supports the separate living-room layouts FloorPlan201/202/203 in larger splits.

`validation.json` is the acceptance report for these exact recipes. `artifact_root` points to oracle/random trajectories, frames, GIFs, debug pages, the full test log, run metadata, and hashes of source code plus snapshots. These are simulator validation results, not VLM reasoning measurements.

Regenerate and validate a fresh split with `python run.py generate --count 4 --split development --seed 42 --output NEW_PATH/development.jsonl` using the THOR Python runtime and rendering settings documented in the parent README. Generation validates oracle solvability by default. Output from `--recipes-only` is explicitly unvalidated.
