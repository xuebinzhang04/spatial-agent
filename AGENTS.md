# SpatialAgent working agreement

This workspace root tracks first-party source in Git. Runtime assets, results,
and some external components stay outside that repository; other components are
separate Git checkouts. Before reporting a diff or PR, verify that the affected
files are tracked in the intended checkout.

## Start with the relevant source of truth

- Read `workspace.md` for ownership, code standards, and experiment records.
- Read `00_Scope/` for the user's research intent; do not edit it.
- For task or demo work, read `survey/2026-10-agent-interaction/DESIGN.md`,
  `04_AgentTask/CustomAgentTasks/expectations.json`, and the relevant task or demo README.
- Before executing an experiment, read `05_Experiment/RECORD_CONTRACT.md` and
  `05_Experiment/JOB_RUNBOOK.md`. Check live account and scheduler state before a paid job.

## Current research loop

Codex is the research lead by default. It should actively inspect relevant
literature and project evidence, propose and rank testable research questions,
choose the next low-cost discriminating case or control, define the acceptance
and demo evidence, and review completed work. Do not wait for the user to supply
each next idea. Keep proposed claims within `00_Scope/` and mark hypotheses as
hypotheses. Claude Code is the primary implementation owner: it turns a scoped
research brief into task, harness, analysis, or demo changes and reports tests
and artifacts. The user can redirect priorities or roles at any time.

Work on one research claim and one frozen example at a time:

1. State the ability, the initial information or control gap, and the interaction
   expected to close it. Identify the matching shortcut or negative control.
2. Freeze the case recipe, observation contract, policy, model, budget, and evaluator.
3. Check generation and native execution separately from policy behavior. A
   privileged oracle is a feasibility witness, not evidence of model ability.
4. Inspect actual observations, actions, feedback, memory, and terminal outcome.
   Record observed facts separately from explanations and proposed fixes.
5. Change one cause at a time, rerun the same case, and record the comparison.
6. Promote a demo example only when its provenance, public observation, action
   consequence, verifier result, and limitations are legible to a viewer.

Keep current findings in a dated note under `archive/agent-interaction/`; keep
run artifacts and metadata under `05_Experiment/results/<run-id>/`. Do not rewrite
historical runs or claim RL improvement or transfer from a generated case or demo.

## Claude Code and Codex handoff

For each change, Codex hands Claude Code a brief with the question, expected
behavior, frozen inputs, control, deliverable, and acceptance check. Claude Code
implements and hands back changed paths, a diff or before/after copies, exact
test commands and results, evidence paths, and unresolved risks. Codex reviews
those inputs without editing the implementation and names a file location,
trigger, impact, and smallest credible fix for each finding. Claude Code resolves
findings; Codex checks the resulting evidence and chooses the next question.
Keep the agents from editing the same files concurrently.

The collaboration remote is `https://github.com/xuebinzhang04/spatial-agent`.
Use tracked first-party diffs and commits for handoff; record paths to ignored
runtime evidence separately. For any untracked first-party file, add it to the
source repository deliberately or retain before/after copies and use `diff -u`.
Do not claim that local commits or unversioned files were pushed until the
remote branch has been verified.

Review task and demo changes for three project-specific failures:

- Private simulator IDs, future state, or verifier data enter policy-visible
  observations, prompts, feedback, or public action references.
- A generated recipe, privileged replay, or successful rendering is presented as
  an uncorrected model run or as evidence of learning.
- A demo example changes its case, budget, observation protocol, or evaluator
  without making the comparison and provenance explicit.
