# Executed Stage 1 validation

Authoritative run: `/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/results/stage1-final-20261019`.

The final source and six canonical scene snapshots match all 32 recorded hashes. No GPU job, model download, simulator reinstall, or training was performed.

| Check | Result |
| --- | --- |
| Unit + real THOR integration tests | 28 passed, 0 failed, 0 skipped |
| Oracle episodes | 12/12 successful |
| Deterministic action/frame replay | 12/12 verified |
| Random rollouts | 12/12 terminated or truncated without crashes |
| Default splits | Development / validation / held-out test, 4 episodes each |

| Task | Oracle successes |
| --- | --- |
| missing_object | 3/3 |
| moved_object | 3/3 |
| shape_identification | 3/3 |
| pose_alignment | 3/3 |

Native replay maximum pixel error: 3 levels/channel. Shape replays are exact. Native object poses are restored from canonical stationary snapshots; native RGB is checked with the explicitly bounded raster tolerance.

Metrics and precise episode IDs/seeds are in [benchmarks/validation.json](benchmarks/validation.json). Complete commands are in [README.md](README.md). The run contains `tests.log`, `run_metadata.json`, `source_hashes.json`, public recipes, and per-episode oracle/random artifacts.

Example evaluator pages: `episodes/development_0000/oracle/debug.html` (missing), `development_0001/oracle/debug.html` (moved), `development_0002/oracle/debug.html` (shape), and `development_0003/oracle/debug.html` (pose), relative to the authoritative run. Each directory includes a GIF and per-step PNGs.

Live VLM inference is unverified. A real request to the configured local endpoint failed with `URLError: [Errno 1] Operation not permitted` under sandbox socket restrictions. The observation-only adapter passes injected-transport tests; partial trace and transport failure are preserved under `05_Experiment/results/stage1-local-20261019/vlm-transport-probe/`.

Earlier diagnostic failures (Unity CPU-device selection, grid-snapping incompatibility, destructive removal, and initial physics drift) are retained under `stage1-local-20261019`. The provisional split run is retained under `stage1-validated-20261019`; the final run above is authoritative.

Remaining limits: native deactivation rather than destruction for missing objects; dynamic L4/L5 deferred; advanced pickup/rearrangement and difficulty calibration deferred; no verified GPU rendering or live model benchmark. Stage 2 needs observation-only VLM baselines, calibrated curricula with containers/multiple changes, persistent learned memory, and RL with held-out evaluations.
