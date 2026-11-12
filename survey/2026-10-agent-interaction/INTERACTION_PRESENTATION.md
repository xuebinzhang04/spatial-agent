# Presenting spatial interaction as research evidence

Reviewed on 2026-10-25. This note follows the user's correction: the deliverable is a research presentation with concrete interaction examples. Its core is the research question and evidence carried by each example. The previous homepage implementation is supplementary infrastructure.

## Presentation unit

Use one research question, a short objective, a recorded observation/action sequence, a deterministic outcome, and one interpretation per example. A useful sequence is **observe -> choose an action -> observe its consequence -> verify**. Include the actual policy, scene, seed, and selected steps. The viewer should understand what capability the task exercises before reading its implementation.

Separate three concepts: a person interacting with a demo, a model interacting with an environment, and a model answering questions about a fixed video. An interactive project page does not by itself establish an interactive learning task.

## References and presentation mechanisms

The following descriptions summarize official project pages or abstracts. Presentation lessons and SpatialAgent proposals are our interpretation, rather than findings attributed to these papers. This is a focused review of relevant work from the supplied publication list, not a systematic review of all the lab's publications. No published performance number is imported as our result.

| Reference | Research content and visible presentation structure | Lesson for our examples |
| --- | --- | --- |
| [DeliveryBench](https://deliverybench.github.io/) | Delivery couples navigation with deadlines, resources, expenses, physical conditions, and social interactions. Its page connects the environment overview to concrete constraints and planning-style cases. | Explain a decision through its consequences and competing constraints. Show which constraint the current action satisfies or violates. |
| [PHAI research](https://phai-lab.github.io/research.html) | Groups work into research themes, including spatial intelligence and robot learning, with a video entry for each theme and linked projects. | Lead with a small number of capability questions and short examples, then explain the shared method. |
| [VLM-3R](https://vlm-3r.github.io/) | Couples monocular video with spatial representations. Its VSTI-Bench examples expose a video, numbered frames, a temporal question, and an answer reveal. | Pair the visual sequence with a specific question. For our tasks, the agent must also choose actions that change its evidence. |
| [SpatialStack](https://spatial-stack.github.io/) | Hierarchical geometry-language fusion is motivated through ROI feature comparisons across encoder depths, followed by an architecture diagram. | A figure should test a specific mechanism. For pose tasks, contrast camera motion with object rotation rather than showing an unexplained rotating object. |
| [VeriWorld](https://shengyun-peng.github.io/papers/veriworld) | The page describes executable spatial tasks with deterministic verification and matched visual, structured, and combined inputs. It also describes seed-based generation and correctness/solvability specifications. Some resources are marked “Coming soon.” | Keep the objective fixed when diagnosing observation or harness effects. Verification and task generation are close related work, not sufficient novelty claims on their own. |
| [CrashTwin](https://crash-twin.github.io/) | Organizes examples around physical failure modes and connects visual sequences to consistency, conservation, and identity diagnostics. Its gallery distinguishes reference, baseline, and post-trained results. | Label the actual failure criterion. A plausible-looking sequence is not evidence of task success. |
| [EgoHOI](https://egohoi.github.io/) | Presents first-person action-driven hand-object synthesis and comparisons starting from a shared initial image, without privileged future object states. | State what information is available to the policy. Never imply a privileged Oracle trajectory was inferred from public RGB. |
| [NavTrust](https://navtrust.github.io/), [paper](https://arxiv.org/abs/2603.19229) | Tests navigation under RGB/depth corruption and instruction changes, with trajectory demonstrations and robustness metrics. | Plan matched clean/perturbed examples with the same task and budget; do not confuse different seeds or budgets with controlled comparisons. |
| [PACER](https://pacer-corl.github.io/) | Uses trace roles such as success, near miss, wrong location, failure, and correction to explain process-based weighting for supervised VLA adaptation. Some numerical fields on the retrieved page are blank. | Preserve partial progress and exact failure reasons. A near miss can motivate learning while still failing the strict task predicate. This is an SFT reference, not evidence that our RL loop is implemented. |
| [RuleSmith](https://adonis-galaxy.github.io/RuleSmith-website/) | Connects adjustable game rules, executable LLM self-play, measured outcomes, and Bayesian optimization; includes replay-oriented explanation. | Calibrate task parameters using actual policy behavior. Game balance is a different objective from spatial task difficulty. |
| [Large Spatial Model](https://largespatialmodel.github.io/) | Connects sparse unposed views to a semantic 3D field and novel-view outputs. | Make the observation-to-spatial-state relationship visible. Reconstruction quality and interaction policy learning remain different claims. |

The [PHAI publication list](https://phai-lab.github.io/publication.html) was used to discover related projects. Their official pages and linked abstracts were inspected. The PHAI theme MP4 URLs did not load through the available browser tool, so this note does not claim to have watched those videos. No external media is reused in our artifacts.

## SpatialAgent research narrative

**Motivation.** Spatial competence requires selecting observations, maintaining spatial memory, executing valid transformations, and reacting when an action changes the world. A final answer alone does not reveal which of these abilities failed.

**Research question.** Can verifiable, compositional interaction tasks supply a scalable learning signal for VLM spatial reasoning, and can that learning transfer beyond the training scenes and task combinations?

**Current construction.** A simulator supplies transitions and observations; task generators select instances and objectives; a harness connects policy actions to execution; evaluators check outcomes. This week's concrete evidence is task execution and recorded rollouts. A shared renderer turns those records into presentation examples.

**Hypothesis, not a result.** Training on interactions that isolate information gathering, memory, transformations, prerequisites, and recovery may improve generalization. The current recordings do not establish an RL gain or transfer.

## Eight concrete example stories

| Task | What the audience should see | What is verified | Question for learning |
| --- | --- | --- | --- |
| missing_object | Room inspection, MEMORIZE, later observations, submitted object and prior location | Change identification and prior-location correctness | Does useful spatial memory improve the final report? |
| moved_object | Inspection before and after a hidden relocation, then a report | Object identity, old location, new location | Can the policy distinguish object motion from viewpoint change? |
| shape_identification | Chosen camera moves with a fixed candidate panel, then a selection | Candidate identity | Which view is informative, rather than merely different? |
| pose_alignment | A camera move followed by object-axis rotations | Symmetry-aware orientation error | Can the policy distinguish observation transforms from world transforms? |
| memory_maze | Physical navigation and ordered landmark marks | Ordered visits, visibility/proximity, terminal location | Does exploration memory reduce later navigation cost? |
| dependency_doors | Opening a real container, pickup, transport, placement, closure | Valid physical prerequisites, delivery order, closed sources, empty hand | Can simple dependencies compose into longer valid plans? |
| relational_rearrangement | Pickup and placement relative to a protected anchor | All requested world-frame relations, preserved objects, empty hand | Do relations survive a change of camera orientation? |
| dynamic_replanning | Checkpoint-triggered scene change and subsequent detour | Physical blocked-edge witness, reachable alternate route, terminal target | Does the policy update its plan from new evidence? |

The selected native `memory_maze` Oracle recording starts in recall and does not contain a MEMORIZE action. Present it as ordered navigation, not proof of learned memory or a complete explore/recall demonstration. Missing/moved objects may be occluded in an individual frame; do not draw an invented bounding box or claim that the hidden change is directly visible in every comparison.

## A failure that explains why training is needed

The recorded Qwen3.8-27B `missing_object` case identifies the change but has previous-location accuracy 0 and strict success false, despite reward 0.925. Show this as a specific diagnostic example. It motivates checking what was remembered and where the information was lost; it does not by itself identify a causal failure mechanism.

Across the four imported Qwen smoke episodes, strict success is 0/4. They are small development cases. Oracle trajectories use privileged state and different budgets, so success examples and model failures are not a fair policy leaderboard.

## Next experiments

1. Freeze scene, seed, instance, budget, action API, and harness when comparing policies or observation conditions. Use task-relevant paired ablations: memory retained/cleared, container open/closed, no event/event, or camera motion/object rotation.
2. Validate generated initial states, physical feasibility, evaluators, and failure cases. Oracle execution supplies a feasibility witness, not visual-policy competence. Private state remains evaluator-only.
3. Calibrate structural levels using repeated policy rollouts, pass rates, action cost, and subgoal failures. Current L1/L2/L3 labels describe construction parameters, not measured policy difficulty. Use the existing TMAX reference for generation/solve/filter organization.
4. Run RL against the existing task API only after reward and reset checks; keep partial reward separate from strict success. Compare to an unchanged base policy with equal rollout budgets.
5. Hold out scenes and parameter combinations. Later evaluate spatial benchmark transfer using a fixed protocol. Neither RL nor transfer has been established by the current demo.

## Artifacts

The maintained [presentation exporter](../../04_AgentTask/Demos/spatial_agent/present.py) consumes the existing public episode catalog. [Research slides](../../04_AgentTask/Demos/spatial_agent/presentation/research-presentation.pdf), task filmstrips, and MP4s live with that exporter. They accompany the academic project page, which embeds the replay inside its research narrative. Research prose and captions are maintained in English.

## DeliveryBench project-page template: source inspection

The user clarified the intended format further: the conventional academic GitHub Pages project-page template. Inspecting the actual source establishes the following reference, beyond removing commercial decoration or changing colors.

- DeliveryBench explicitly credits [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), derived from [Nerfies](https://nerfies.github.io/). The template repository includes a modernized main version and an `original-version` branch; they should not be assumed identical.
- Its [actual HTML](https://github.com/DeliveryBench/deliverybench.github.io/blob/main/index.html), [CSS](https://github.com/DeliveryBench/deliverybench.github.io/blob/main/static/css/index.css), and [JavaScript](https://github.com/DeliveryBench/deliverybench.github.io/blob/main/static/js/index.js) were read. The page is static HTML/CSS/JS: Bulma handles the layout, Inter is the font, and small scripts support controls. GitHub Pages is the hosting layer, not the name of the visual template.
- The reading order is publication title and subtitle -> author/affiliation block -> compact resource links -> teaser video -> abstract -> benchmark overview figure and caption -> features -> results -> analysis -> citation and attribution. The body is organized as a research argument with figures, not a software dashboard.
- DeliveryBench aligns its main video and several body sections to centered wrappers of 70% width, uses centered section headings, and places explanatory paragraphs with figures. This is a source observation, not a universal responsive-width rule to copy blindly.
- The base template provides optional video/image carousels and sliders. Native HTML video and small JavaScript extensions can carry interaction examples inside the research sections; they do not require a separate app shell.
- SpatialAgent should reuse that publication-page hierarchy and place task examples inside it: research title -> brief scope -> interaction teaser -> motivation -> environment/task overview -> task videos or frame-based interaction examples -> generation/verification -> current evidence and next experiments. Only real authors, released resources, and actual findings should be populated. A dashboard-first layout and a PDF-only replacement both miss this intended format.
- Reusing template code requires retaining its attribution and applicable license. A future adaptation should preserve those notices, remove sample author/resource/analytics metadata, and keep local assets for easy startup.

The subsequent implementation now adapts the upstream `original-version` publication layout, with its unchanged Bulma 0.9.1 and template CSS vendored locally. The existing record player is embedded in Interaction Examples, after the teaser, abstract, and overview. Attribution, licensing, and exact source URLs are recorded in [static/TEMPLATE.md](../../04_AgentTask/Demos/spatial_agent/static/TEMPLATE.md). Local media and styles are copied into the self-contained build. This is a local template migration; no hosted publication is implied.
