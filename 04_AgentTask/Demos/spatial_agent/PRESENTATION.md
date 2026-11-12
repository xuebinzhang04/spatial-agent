# SpatialAgent research presentation

The presentation explains the research through concrete interaction examples. The main entry is the [academic project page](site/index.html), using Academic Project Page Template / Nerfies layout with the record player embedded in its examples section. Supplementary media are a [13-page PDF](presentation/research-presentation.pdf), [eight-task overview](presentation/overview.png), task filmstrips, and MP4s. The reusable components are the publication-page template, observation/action/evaluation records, and media renderer, maintained independently of meeting dates.

## Research story and speaking order

1. **Scope:** learn spatial reasoning through active interaction and RL. Ask whether controlled, verifiable tasks can provide a scalable learning signal that generalizes.
2. **Construction:** Environment -> Task -> Agent -> Verifier. Explain where state transitions, instance generation, execution, and correctness live. RL and difficulty calibration are next steps.
3. **Examples:** show a goal and a real action's consequence before explaining its verifier. Group examples into active perception, spatial memory, transformation, prerequisites, relation grounding, and replanning.
4. **Diagnostic failure:** the model can identify a missing object but report its previous location incorrectly. The strict objective prevents partial credit from being mistaken for success.
5. **Progress and next steps:** distinguish runnable tasks, Oracle feasibility, and actual VLM attempts from unmeasured RL and transfer. End with controlled experiments that could resolve the open questions.

For a short talk, use slides 1–2, shape identification, dependency doors, dynamic replanning, model failure, and evidence/next steps. The other task examples remain available for discussion. Start each video near the relevant event using the steps in [examples.json](presentation/examples.json); full sequences are included for inspection. At 2 steps/s, source step `k` appears at `k / 2` seconds. Filmstrips show selected steps, not the duration between them.

## Interaction examples

| Research question | Filmstrip | Video | Interpretation boundary |
| --- | --- | --- | --- |
| What disappeared, and where was it? | [missing_object](presentation/03-missing_object.png) | [MP4](presentation/missing_object.mp4) | Privileged reference; no visual inference is demonstrated by this scripted tour. |
| Did the object move, or the observer? | [moved_object](presentation/04-moved_object.png) | [MP4](presentation/moved_object.mp4) | Old and new locations are both required; a single frame may not expose the hidden change. |
| Which view resolves shape ambiguity? | [shape_identification](presentation/05-shape_identification.png) | [MP4](presentation/shape_identification.mp4) | Camera movement changes evidence; successful privileged selection does not establish view-selection learning. |
| What changes the view versus the pose? | [pose_alignment](presentation/06-pose_alignment.png) | [MP4](presentation/pose_alignment.mp4) | Camera actions and object-axis rotations have different effects. |
| Can memory support ordered revisits? | [memory_maze](presentation/07-memory_maze.png) | [MP4](presentation/memory_maze.mp4) | This selected Oracle trace starts in recall; no exploration/MEMORIZE or causal memory benefit is shown. |
| Can prerequisites compose into a valid plan? | [dependency_doors](presentation/08-dependency_doors.png) | [MP4](presentation/dependency_doors.mp4) | Native container access, single-hand transport, and closure; not key/lock puzzles. |
| Can a language relation become a physical state? | [relational_rearrangement](presentation/09-relational_rearrangement.png) | [MP4](presentation/relational_rearrangement.mp4) | World-frame relation plus protected-object preservation. |
| What should change when the world changes? | [dynamic_replanning](presentation/10-dynamic_replanning.png) | [MP4](presentation/dynamic_replanning.mp4) | The checkpoint triggers a physical obstruction; the Oracle executes a detour. Learned replanning is untested. |
| Why is partial progress insufficient? | [Qwen diagnostic](presentation/11-model-failure.png) | [MP4](presentation/missing_object-model-failure.mp4) | Reward 0.925 but previous-location accuracy 0 and strict failure. This is one recorded attempt. |

## What the references contribute

[DeliveryBench](https://deliverybench.github.io/) connects decisions to constraints. [VLM-3R](https://vlm-3r.github.io/) connects video, selected frames, questions, and answers. The [PHAI research page](https://phai-lab.github.io/research.html) and [publications](https://phai-lab.github.io/publication.html) lead to relevant examples of mechanism comparisons, trace roles, physical diagnostics, and robustness tests. The [reference review](../../../survey/2026-10-agent-interaction/INTERACTION_PRESENTATION.md) records 11 entries and their project implications. These inform our research presentation and proposed experiments; their media and performance claims are not reused as our evidence.

## Reproduce the artifacts

From the SpatialAgent root:

```bash
module load conda
conda activate spatialagent
python 04_AgentTask/Demos/spatial_agent/present.py
```

No browser, local server, model load, or simulator launch is needed. The exporter uses Pillow, the existing public `site/data/catalog.json`, original lossless observations, and CPU ffmpeg from the active environment. `examples.json` beside the exporter defines research captions and important action types. `--site`, `--output`, `--record`, and `--ffmpeg` select other inputs or paths. A new timestamped experiment record is created by default; failed exports retain their manifest.

The PDF contains raster slides. PNGs can be inserted into a slide deck, and the corresponding MP4 played alongside them. Videos contain all recorded steps in order at two steps per second, followed by a two-second terminal hold. This is a discrete interaction replay, not the original wall-clock speed. The policy label is always visible. Each image is the complete source observation, including existing task overlays; captions sit outside it. Native 128-pixel frames are enlarged without inventing detail.

## Evidence

The original native/geometry media subset contains 8 task families, 20 episodes, and 875 frames. The active catalog additionally imports native delivery records; graph prototype records are archived and excluded. The current project page reports its actual totals. The presentation selects 8 Oracle/reference episodes and 1 Qwen3.8-27B failure. Oracle uses privileged state. All 4 imported model smoke episodes fail strict success; budgets and instances are not matched across policies. Two pending task definitions are not presented as completed 3D task examples. Structural difficulty labels are not calibrated policy difficulty. RL training and benchmark transfer have not been demonstrated.

The [exported example manifest](presentation/examples.json) identifies each source trajectory, selected steps, policy, final metrics, and annotation. Export metadata is written under `05_Experiment/results/interaction-presentation-<timestamp>/`; the first export uses [interaction-presentation-20261025](../../../05_Experiment/results/interaction-presentation-20261025/run_metadata.json). All task media use the project's own recorded observations.

The maintained presentation now uses plain academic figure formatting: black text on white, task names, observations, actions, and outcome captions. The earlier branded layout is superseded. Its media is preserved in the first export's `presentation-before-layout-correction.zip`; current media is recorded under [research-presentation-media-20261025](../../../05_Experiment/results/research-presentation-media-20261025/run_metadata.json).

The unified page now provides selector-driven simulator/task definitions and an Online Demo section. Delivery is a task on the dedicated SimWorld gym_citynav build. Its [recorded delivery clip](presentation/deliverybench-ue-345795.mp4) shows the first-person stream from the existing manual/script cutoff, while the common player retains all three named streams. This is not a new benchmark rollout.
