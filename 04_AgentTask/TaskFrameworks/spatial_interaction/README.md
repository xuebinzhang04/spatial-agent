# Spatial Interaction Learning — Stage 1

> This is a shared multi-task framework. Independent, task-named entrypoints now live in [CustomAgentTasks](../../CustomAgentTasks/README.md). Use this directory for framework-wide maintenance and regression.

Two working interactive simulator backends share `reset(seed=None, options=None) -> (observation, info)`, `step(action) -> (observation, reward, terminated, truncated, info)`, `render() -> uint8 RGB`, and `close()`. This implements simulator construction and validation; no RL training is included.

Eight task entrypoints live directly under CustomAgentTasks. This backend reuses installed NumPy/Pillow, the cached AI2-THOR 5.0.0 runtime and the existing Qwen inference service. Gymnasium call signatures are supported without requiring its base class or spaces. The experimental graph backend has been retired to the existing archive.

## Native observation update

New native RGB has no object-number overlays. Public `view:STEP:X:Y` references resolve through private visible-pixel masks cached for observed frames; `view_step` is public. Submission resolves these references to evaluator IDs internally. Private IDs are rejected by default (invalid action, one step consumed); `OracleAgent`, construction validators and replay of oracle traces opt in with `env.privileged_references = True`. Native task instructions replace numbered targets with object types; generation selects only objects with a unique public description (`public_description`). Historical native acceptance does not validate this revised protocol; new scene-level and policy checks are required in an allocation. See the [current demo protocol](../../Demos/spatial_agent/README.md#native-visual-observation-protocol).

## Run without installing anything

All commands below start from the task directory. Use a fresh output directory for every rollout/validation run.

```bash
cd /gpfs/scrubbed/xuebiz/SpatialAgent/04_AgentTask/TaskFrameworks/spatial_interaction
export SPATIAL_ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export SPATIAL_PY=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
export SPATIAL_THOR_PY="$SPATIAL_ROOT/02_Simulator/AI2THOR/runtime/python/bin/python"
export SPATIAL_OUT="$SPATIAL_ROOT/05_Experiment/results/my-stage1-run"
```

The lightweight renderer works on CPU with no display server:

```bash
"$SPATIAL_PY" run.py run --task shape_identification --seed 42 --policy oracle --debug --output "$SPATIAL_OUT/shape"
"$SPATIAL_PY" run.py run --task pose_alignment --seed 42 --policy oracle --debug --output "$SPATIAL_OUT/pose"
```

For the **verified software Vulkan path on this host**, set:

```bash
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2
export XDG_CONFIG_HOME="$SPATIAL_OUT/unity-config"
export XDG_CACHE_HOME="$SPATIAL_ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
"$SPATIAL_THOR_PY" run.py run --task missing_object --scene FloorPlan1 --seed 42 --policy oracle --debug --output "$SPATIAL_OUT/missing"
"$SPATIAL_THOR_PY" run.py run --task moved_object --scene FloorPlan1 --seed 43 --policy oracle --debug --output "$SPATIAL_OUT/moved"
```

This uses a cached CloudRendering build, FIFO IPC, and Unity `-force-device-index 0`. The default Unity selection skips the CPU Vulkan device and crashes on this host. No X server is needed. GPU inference/simulation belongs in a Slurm allocation following `SpatialAgent/workspace.md` and `05_Experiment/JOB_RUNBOOK.md`. For GPU THOR, unset `SPATIAL_THOR_SOFTWARE` and `VK_ICD_FILENAMES`; GPU execution has not been verified by this implementation session. No Slurm submission was made: the sandbox denies scheduler sockets and has no working NVIDIA driver.

Run all unit and **real Unity** integration tests, oracle episodes, random rollouts, and action/frame replay in a single command:

```bash
"$SPATIAL_THOR_PY" validate.py --thor --splits --output "$SPATIAL_ROOT/05_Experiment/results/my-stage1-validation"
```

The validator creates `run_metadata.json` before running, saves concrete episode recipes and source hashes, and retains partial/failed results. `--splits` validates four task types in each of development, validation, and test. Without `--thor`, only the shape backend is executed and Unity tests are explicitly skipped.

Standalone tests:

```bash
"$SPATIAL_PY" -m unittest discover -s tests -v
SPATIAL_TEST_THOR=1 "$SPATIAL_THOR_PY" -m unittest discover -s tests -v
```

## Generate, reload, evaluate, replay

```bash
"$SPATIAL_THOR_PY" run.py generate --count 4 --split development --seed 42 --output "$SPATIAL_OUT/benchmark/development.jsonl"
"$SPATIAL_THOR_PY" run.py generate --count 4 --split validation --seed 42 --output "$SPATIAL_OUT/benchmark/validation.jsonl"
"$SPATIAL_THOR_PY" run.py generate --count 4 --split test --seed 42 --output "$SPATIAL_OUT/benchmark/test.jsonl"
"$SPATIAL_THOR_PY" run.py run --episodes "$SPATIAL_OUT/benchmark/development.jsonl" --index 1 --policy random --output "$SPATIAL_OUT/random"
"$SPATIAL_THOR_PY" run.py replay "$SPATIAL_OUT/moved" --verify
"$SPATIAL_PY" run.py replay "$SPATIAL_OUT/shape" --verify
"$SPATIAL_PY" run.py evaluate "$SPATIAL_OUT"
```

`generate` validates every episode with the real selected backend and an oracle before writing the JSONL and its validation certificate. To export *unvalidated recipes* explicitly, use `--recipes-only`. Shape-only generation works in the main environment with `--environment shape3d`. Prevalidated default recipes are in `benchmarks/{development,validation,test}.jsonl`; their certificates and artifact paths are recorded in `benchmarks/validation.json`. An empty recipe `instruction` selects the backend default prompt; the resolved instruction is present in every observation and trajectory.

Every trajectory contains `episode.json`, `trajectory.jsonl`, a PNG at reset and every step, `report.json`, and `replay.gif`. With `--debug`, it additionally saves `privileged.jsonl`, `ground_truth.json`, initial/changed screenshots for THOR, and an offline `debug.html` evaluator view. Open the GIF or HTML directly. Debug artifacts are never supplied to a policy.

Shape replay compares exact frame hashes. THOR replay compares actions, observations, rewards, termination, and frames. For debug traces it also checks native object bindings, poses, container membership, interaction flags, camera pose, and task history (pose tolerance 0.0001). RGB mean absolute error must be <= 0.5; at most 0.1% of pixels may differ by more than eight channel levels. Fresh Unity instances exhibited isolated raster edge differences at identical physical state; maximum pixel error and outlier counts are reported. `strict_pixels=True` still requires exact hashes. Canonical snapshots remove native initial physics drift. Cross-device pixel identity is not promised.

## Observation and action contract

```python
from spatial_interaction import EpisodeConfig, make_env

env = make_env(EpisodeConfig(task_type="pose_alignment", seed=42))
try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(
        {"action": "OBJECT_ROTATE_X_POS", "memory": "I rotated around world X."})
finally:
    env.close()
```

Observations contain exactly `rgb`, `instruction`, `phase`, `action_history`, and `memory`. Memory is optional agent-authored text supplied through an action. The model adapter explicitly serializes these public fields; it never receives the environment object, evaluator `info`, full geometry, native metadata, or answers. The VLM can use its notes between phases. Saved frames provide the researcher's observation history.

Actions are JSON dictionaries (or JSON strings). Unexpected keys, missing/wrong fields, duplicates in answers, unknown candidates/IDs, actions in the wrong phase, and native failures are explicitly rejected. Invalid actions consume one step and incur the usual small action cost; the episode continues. Steps after episode end require a reset. `reset(options=...)` rejects unsupported options; all supported generation parameters belong in `EpisodeConfig.task_config`.

Dynamic actions: `MoveAhead`, `MoveBack`, `RotateLeft`, `RotateRight` (30 degrees), `LookUp`, `LookDown`, `OpenObject`, `CloseObject`, `PickupObject`; object interactions require `object_id`. `MEMORIZE` advances to detection, or the exploration budget advances it automatically. `SUBMIT` terminates:

```json
{"action":"SUBMIT","answer":{"changes":[{"type":"moved","object_id":"object_058","previous_location":"object_019","current_location":"object_018"}]}}
```

IDs are assigned from the complete initial scene before choosing an intervention. Only visible segmentation detections are labeled in RGB; compact label `058` means `object_058`. Receptacles use the same stable IDs. `previous_location` and `current_location` are receptacle IDs, or `floor` when appropriate. The numeric labels and connector lines reduce annotation overlap without exposing hidden objects, positions, or the chosen target.

Shape camera actions: `CAMERA_LEFT/RIGHT/FORWARD/BACKWARD`, `CAMERA_YAW_LEFT/RIGHT`, `CAMERA_PITCH_UP/DOWN`. B1 submits `{"action":"SUBMIT_SHAPE","candidate_id":"candidate_2"}`. B2 also supports `OBJECT_ROTATE_{X,Y,Z}_{POS,NEG}` and `{"action":"SUBMIT_POSE"}`. Object rotations compose about **world axes**, independently of camera pose. A fixed reference panel communicates the desired pose; candidate panels show three rendered reference views each. These are task stimuli, not geometry/answer metadata.

## Backend details and generation controls

**Dynamic memory.** Reset restores a canonical scene snapshot with `SetObjectPoses(placeStationary=True)`, seeds Unity and NumPy, and pauses physics. The generator selects a reachable view and a visible, pickupable, non-container target from actual metadata and image detections. It preflights an intervention, checks unchanged distractor poses, restores the scene, and repeats the verified intervention at the phase boundary. Relocation uses native receptacle spawn points and collision-checked `PlaceObjectAtPoint`, requires >= 0.4 m displacement and a different semantic receptacle, and validates a reachable visible viewpoint. Invalid generation fails explicitly rather than producing a labeled episode.

Missing-object semantics are **native deactivation**, not object destruction: `DisableObject` calls `GameObject.SetActive(false)`, disabling rendering, colliders, and interactions. THOR retains the inactive registry entry. The evaluator records the inactive set separately, verifies invisibility/noninteractability and no segmentation detection, and excludes inactive entries from the accessible task scene. The agent cannot call `EnableObject`. Destructive `RemoveFromScene` was tested on this cached build and hangs after a native metadata `NullReferenceException`; its failed diagnostic log is retained. No visual hiding workaround is presented as physical deletion.

L1 uses a directly visible target with a minimum projected bounding-box area. L2 uses a small target and starts 120 degrees away from its verified view, requiring viewpoint adjustment. L3 relocates one target to a different open receptacle. L4/L5 are explicitly unsupported rather than silently downgraded. Existing objects provide unchanged distractors; the scene controls their count and occlusion. Open/close and pickup use native visibility/capability checks. Held items are dropped before intervention; systematic tasks involving agent rearrangement/pickup are not yet calibrated.

Dynamic `task_config`: `image_size` (64..512), `exploration_budget`, `min_target_pixels`, `max_target_pixels`, `max_target_distance`, `initial_pose` (native camera pose), `reset_camera` (default true), `scene_state_path` (privileged canonical snapshot). Size is currently a projected-area proxy; arbitrary distractor counts and controlled container occlusion are deferred. Six supplied kitchen/living-room snapshots match the installed build. To introduce another scene, capture and verify a privileged snapshot first:

```bash
"$SPATIAL_THOR_PY" capture_scenes.py --scenes FloorPlan4 --runtime-dir "$SPATIAL_OUT/capture-runtime"
```

The capture tool refuses to overwrite existing snapshots and checks native restoration. The default snapshots are benchmark assets separate from public observations.

**Shape3D.** Connected polycubes are generated procedurally. A NumPy perspective ray/voxel renderer supplies depth-correct hidden-surface removal, directional face shading, and optional physical foreground occlusion. It needs no GL context, mesh package, or simulator process. The camera can translate/yaw/pitch; the target uses exact integer rotation matrices. A breadth-first oracle finds a solution in the proper cube rotation group. Pose verification minimizes rotation error over the geometry's rotational symmetry group; translation-normalized geometry distinguishes rotations from reflections.

Shape `task_config`: `voxels` (4..30; default `5+difficulty`), `symmetry` (`asymmetric`, `symmetric`, `any`), `axes` (nonempty subset of `XYZ`), `initial_rotation` (index 0..23 in `geometry.ROTATIONS`), `camera_position`, `camera_yaw`, `camera_pitch`, `occlusion` (0..1), `candidates` (2..6), `image_size` (64..512), `split`. Higher levels increase voxel count/candidates; L4/L5 add occlusion by default. Restricted-axis reference poses are generated through allowed rotations and checked against the step budget. Distractors are unique under proper rotations; when feasible, one shares an orthographic front silhouette with the target. Identical perspective appearance is not claimed.

Splits use separate iTHOR layouts (FloorPlan1/201, 2/202, 3/203) and seed ranges. Every shape, including distractors, is assigned by a hash of its full rotation-equivalence class, preventing geometry leakage across splits. The default four-episode splits use the kitchen scene from each pair; larger splits include living rooms. No ProcTHOR package/dataset or existing 3D shape wrapper was found; neither is required for this MVP.

## Reward, policies, integration, and limitations

Each action costs `0.1/max_steps`; total cost is capped at 0.1 over an episode. Dynamic terminal score is changed-object F1. Type, previous/current receptacle accuracy, full-answer success, step count, and efficiency are reported separately. A wrong location can retain F1 reward but fails full-answer success, as specified by the initial reward. Shape/pose terminal score is binary correctness. Timeouts score zero minus action costs. Pure evaluators work without a model.

`OracleAgent` explicitly uses privileged evaluator state to validate mechanical solvability. Its scores must not be interpreted as observation-only reasoning performance. `RandomAgent` sees only observations and action names; it exercises navigation/termination and excludes targeted object interactions. The validator records both policies separately.

`VLMAdapter` connects to the existing server configured by `scripts/serve_qwen.sh` through `QWEN_BASE_URL` (default `http://127.0.0.1:8000/v1`) and `QWEN_MODEL` (default `Qwen3.8-27B`). The model stays resident in the external service. No per-step model load occurs.

**Harness (2026-11-09).** Each model call is one *turn* returning a JSON list of up to `max_actions` actions (default 3, VAGEN-style). `runner.run_turn` executes them in order, stops at the first failed action, phase change or episode end, and presents only the image after the last executed action; earlier frames of the turn are saved but marked `withheld` in `trajectory.jsonl`, and native environments drop them from `view:` resolution and pre-change evidence (`withhold_view`). Input modes:
- `history='rollout'` (default, Theory of Space-style): a growing multi-turn chat — system rules, then per turn the execution feedback, current state, allowed actions and that turn's image, with the model's raw replies kept as assistant messages. The instruction is re-sent only when the phase changes. `image_window=K` keeps images only in the last K observation messages.
- `history='memory'`: the earlier stateless single message (current image, action history, notebook, last-turn results); the belief-bottleneck alternative.

CLI: `run.py run --policy vlm --history rollout|memory --max-actions N [--image-window K]`; the settings are recorded in `report.json` (`harness`, `turns`, `environment_steps`). Scripted oracle/random policies still act one step per turn. The legacy `VLMAdapter.act` (one action, memory input) remains for older scripts. The demo `Session.infer` path has not been migrated. To run against an already running service:

```bash
QWEN_BASE_URL=http://127.0.0.1:8000/v1 QWEN_MODEL=Qwen3.8-27B \
  "$SPATIAL_PY" run.py run --task shape_identification --policy vlm --output "$SPATIAL_OUT/vlm-shape"
```

The adapter's payload filtering, image encoding, repeated interactions, and action parsing are tested with an injected inference transport. **Live Qwen vision inference was not run in this session**: no GPU allocation was available, and a live endpoint request failed with `URLError: [Errno 1] Operation not permitted` under sandbox socket restrictions. Transport errors preserve partial traces and error records. THOR tasks must run in the THOR Python runtime; the HTTP client uses the standard library and needs no additional SDK in that runtime.

Current limits: simple discrete geometry/materials, rendered candidate views rather than meshes as agent input, one hidden dynamic change, no calibrated L4/L5 memory tasks, no full RL loop, no observer-only scripted reasoning claim, and no verified GPU throughput. Stage 2 should add observation-only VLM baselines, calibrated navigation/occlusion and multi-change curricula, learned persistent memory, and RL with held-out-layout/shape evaluation. Native destructive removal should remain optional until the build's metadata bug is addressed.

See `FILES.txt` for the full created-file inventory and `benchmarks/validation.json` for the executed default-split validation results.
