# 2026-11-12 — Shape3D evidence card review

## Research question and fixed evidence

The presentation question is whether a viewer can identify the initial shape
ambiguity, the camera path that changes the public evidence, and the outcome
without mistaking a privileged oracle replay for model behavior. The frozen case
is `showcase_shape_identification_seed2007` in
`05_Experiment/results/showcase-cases-20261111/cases/shape_identification/`.
Its public trajectory SHA-256 is
`91515028c248137f815d745b7d1f4226cdda2b70b14224b166b652700b4c8df7`;
the separate privileged trace SHA-256 is
`06a85355d3018f1478a625198ff62c02acde292692d51300140aab52e4b58626`.
The task, 224 px observation protocol, 60-step budget, evaluator, and recorded
policy are unchanged. The generator's initial-pixel twin and revealing-view
checks are construction evidence. An initial-view-only model control has not
been run.

## Handoff and review

Codex scoped a display-only change to the shape card. Claude Code implemented
a side-by-side public step-0 and step-10 comparison, the ten camera actions,
and an explicit privileged-oracle label in
`04_AgentTask/Demos/spatial_agent/{build.py,style.css,test_demo.py,README.md}`.
The first independent review found that the page claimed the displayed path
was generator-verified while the code checked only its action count. Claude
Code then added a build-time exact comparison with the step-0
`ambiguity_witness.actions` in `privileged.jsonl`. The private witness is used
for this assertion and is not copied into the page or catalog. The caption now
states the narrower generator criterion: the target and one initial twin
candidate differ in at least 10% of foreground pixels at the witness view.

The reviewed diff changes no case recipe, budget, observation, evaluator, or
recorded run. The card remains a presentation of a privileged feasibility
witness, not a VLM run or learning result. The final source review found no
remaining actionable issue in the changed logic.

## Verification and limits

- `git diff --check`: passed.
- `/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python -m unittest
  test_demo.DemoTests.test_shape_evidence_rejects_same_length_different_path`
  from the demo directory: 1 test passed; a same-length reordered path and a
  missing witness are rejected.
- The pre-fix complete demo suite passed 27 tests. This does **not** validate
  the final change. The final full build and page-level suite remain pending;
  the generated site in this worktree predates the exact-path fix.
- On this check, `sinfo -s` showed no idle H200 or MIG node, `squeue --me` was
  empty, and `hyakusage -a video` showed $0 active credits. No paid Slurm job
  was submitted. Browser layout has not been visually inspected.

Next, rebuild the site in an allowed allocation when account and scheduler
capacity permit, run `test_demo.py` against that build, inspect the rendered
comparison at desktop and mobile widths, and keep the test artifacts with a
run manifest. Only then promote the display change as fully verified. A later
model comparison must use the same frozen case and an explicit initial-view
control; the oracle clip is not that control.
