# Shared task support

Eight independent native/geometry task entrypoints live in [CustomAgentTasks](../CustomAgentTasks/README.md).

- `spatial_interaction/`: AI2-THOR dynamic-memory and native manipulation/navigation backends, Shape3D renderer, episode schema, model adapter and regression tools.
- `task_support.py`: binds each task's `task.json` to the shared spatial implementation and enforces task identity.
- `validate_layout.py` and `tests/`: check task boundaries, generation, replay and CLI entrypoints. Native execution requires an allocated simulator runtime.

The experimental Interaction Lab graph backend and its harness were retired to `archive/agent-interaction/legacy-demos/symbolic/`. Historical manifests retain their original paths; active task APIs and the demo no longer accept this backend.
