# 2026-11-12 — Research and agent handoff workflow

## Purpose and current boundary

The immediate project goal is a working environment and a research demo. The
user's long-term scope is interactive spatial learning for VLM agents through RL,
but the present evidence is primarily task construction, native admission,
recorded examples, and limited model execution. The existing research protocol
already states the information-gap test; this note makes the next work cycle and
demo acceptance check small enough to use with two independent agents.

At inspection, `SpatialAgent/` had no root Git metadata. `tmux` had a `codex`
session and Claude sessions, but the displayed panes were rooted at the parent
directory or another project. A review of first-party files therefore needs
named paths and before/after copies, not an assumed `main` diff or PR. No tmux
session was driven or interrupted for this change.

The root was subsequently initialized as a source-only Git checkout for the
user's collaboration remote. The paragraph above records the state when this
workflow was first drafted; new first-party changes should use tracked diffs
once the initial commit exists.

## Collaboration repository access check

The intended remote is `https://github.com/xuebinzhang04/spatial-agent`.
After the user made it public on 2026-11-12, Codex's GitHub connection read
its repository metadata successfully: public, default branch setting `main`,
size 0, and no commit yet. Codex's anonymous HTTP check returned 200 and
`git ls-remote --symref ... HEAD` exited 0 with no refs. An independent
read-only Claude Code invocation obtained the same HTTP 200 and empty successful
`ls-remote` result. Thus both agents can reach the repository for anonymous
reads; neither has yet read a committed project file or tested writes. No
GitHub API token is needed for this read test. The repository still needs an
initial commit before branch diffs, reviews, and PRs can be exercised.

## One-case research cycle

Use a single case as the unit of progress. A case handoff should fit in one note:

| Field | Required content |
| --- | --- |
| Question | One capability claim that the case could test |
| Gap and action | What the initial observation cannot settle; which executable action can change the evidence |
| Control | A shortcut or matched counterfactual that should fail or differ |
| Frozen inputs | Task/scene/seed, recipe hash, observation protocol, policy/model, budget, evaluator |
| Evidence | Public trajectory, private verifier result, logs, exact command and run ID |
| Interpretation | Observed behavior, plausible cause, next single change; label unknowns |

Generation admission, oracle feasibility, model behavior, and learning outcomes
remain separate evidence levels. A case advances to the demo only after the
audience can see the question, the new evidence produced by an action, and the
strict outcome without reading raw implementation details.

## First proposed iteration

Start with the CPU Shape3D `shape_identification` showcase case. Its current
generator checks an identical initial target image for a counterfactual pair and
a reachable revealing view. This makes it a low-cost way to test the concrete
question: **does the displayed action reveal information that the initial view
cannot provide?** Use the current frozen showcase recording first; inspect its
public frames, candidate panel, action sequence, verifier, and provenance. The
existing recorded generator audit supports case construction, not VLM ability.

If the presentation is unclear, revise only the explanation or playback for this
case, then ask an independent reviewer to check whether the initial ambiguity,
revealing action, and outcome are visible and accurately labelled. A later model
comparison should freeze the same case and budget, compare initial-view-only and
interactive policies, and use an allocated model service only after the normal
budget/scheduler checks. Do not use a privileged oracle clip as a model baseline.

The first case sheet is already fillable from the existing artifacts:

| Field | Current evidence |
| --- | --- |
| Frozen inputs | `showcase_shape_identification_seed2007`, Shape3D, difficulty 2, 224 px, 60-step budget, required initial pixel ambiguity; `05_Experiment/results/showcase-cases-20261111/cases/shape_identification/episode.json` |
| Gap and action | The generator's recorded witness says the initial target pixels have a wrong-candidate twin; six `CAMERA_LEFT`, two `CAMERA_FORWARD`, and two `CAMERA_YAW_RIGHT` actions reach a distinguishing view. The first and tenth recorded frames visibly differ; this alone does not prove a model selected the path. |
| Control | An initial-view-only choice should be tested against the pixel twin under the same candidate panel. No model control result is recorded here. |
| Outcome | The privileged oracle submits `candidate_1` at step 11, terminates successfully, and records 10 camera actions. This is a feasibility and presentation witness. |
| Evidence | `trajectory.jsonl`, `ground_truth.json`, `report.json`, and `frame_0000.png` through `frame_0011.png` in that case directory. |
| Open presentation check | Make the initial ambiguity and the first useful new view legible in the page; have the reviewer identify those frames from the public recording and confirm the clip is labelled oracle. |

Next, apply the same case sheet to one native memory task. The 2026-11-11 audit
identifies historical weak memory examples and current native admission evidence;
it does not yet show a measured memory advantage. Keep this as a second question
until the cheap presentation check is resolved.

## Demo acceptance check

For each featured case, check these items against the actual recording:

1. A viewer can state the task question and what is initially unknown.
2. The replay marks the exact action and subsequent observation that add evidence.
3. The public view contains no private identifiers or evaluator answer.
4. Policy identity, scene/seed, case source, observation protocol, and success or
   failure are visible or one click away.
5. The presentation says whether the clip is an oracle witness, scripted tour,
   uncorrected model run, or generation audit.
6. Any claim of model improvement uses matched runs and metrics; a visually
   cleaner clip alone supports only a presentation improvement.

## Minimal two-agent pass

Codex owns the research queue: it proposes the next testable question from the
scope, literature, failed cases, and current artifacts; ranks it by information
value, cost, and demo relevance; and writes the brief without waiting for another
user idea. Claude Code owns implementation. Codex performs independent evidence
review and updates the next research brief. The user may redirect either role.

1. **Codex:** choose one case and one deliverable, then hand Claude Code the
   question, control, frozen inputs, acceptance check, and evidence paths.
2. **Claude Code:** save before copies of the few
   first-party files to be edited, or work in a real Git checkout. Implement and
   run only the relevant local checks. Write a short handoff with the six case
   fields above, changed paths, `diff -u` output or Git diff, commands/results,
   and remaining uncertainty.
3. **Codex reviewer:** read only the handoff and affected source/evidence. Return only
   actionable findings with location, trigger, impact, and smallest fix. Check
   public/private boundaries and whether the demo claim matches the run type.
4. **Claude Code and Codex:** the implementer addresses findings and reruns the
   affected check; Codex records the final evidence and selects the next case.

Reusable reviewer request:

> Read `AGENTS.md`, the case handoff, the listed before/after diff, and its run
> evidence. Do not modify files. Check the stated acceptance criteria, public
> observation boundary, case/protocol consistency, and whether the demo claim
> follows from the trajectory. For each finding, give path and line, trigger,
> impact, and a minimal fix. Say when evidence is insufficient.

## Change and verification

This workflow pass adds `AGENTS.md`, links it from `CLAUDE.md`, and records this
case-oriented process. It does not change task code, simulator state, recorded
experiments, or the demo page. Verification is a consistency check against
`workspace.md`, `00_Scope/`, the shared research protocol, current expectations,
the demo README, and the 2026-11-11 audit record. No GPU job or model inference
was run.
