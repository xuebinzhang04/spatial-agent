# DeliveryBench interaction cases

Environment: SimWorld's dedicated DeliveryBench `gym_citynav` UE build. Task layer: the existing native DeliveryBench order, movement, bag, charging and settlement APIs. This directory adds fixed case construction, public task instructions and strict evaluation. It does not modify the upstream benchmark or implement another simulator.

## Initial families

| Family | Required behavior | Controlled variables | Admission evidence |
| --- | --- | --- | --- |
| `pickup_delivery` | Accept → travel → wait for preparation → pick up → pack → travel → deliver | Start pose, restaurant/building doors, menu item, deadline, delivery method | Native order transitions, bag evidence, actual UE proximity; remote pickup and premature delivery must fail |
| `batch_schedule` | Accept the batch and schedule preparation, pickups and deliveries | 2–3 orders, preparation times, distances, case-relative deadlines | A graph-feasible serial witness and a different serial order that misses a deadline; native timing must then validate feasibility |
| `recharge_delivery` | Reach a station, restore sufficient charge and complete the order | Initial battery, graph distances, charge rate, target charge, transport restrictions | Station reachable with initial battery; direct delivery exceeds initial battery. Actual battery increase at any public charging station, no depletion, and completed delivery |
| `visual_handoff` | At the building, inspect and approach the customer for direct handoff | Coarse destination, hidden customer position, arrival view, primitive movement | **Candidate only:** requires verified customer visibility, an informative later viewpoint and UE/Python position synchronization before admission |

L1–L3 are structural profiles: longer trips; more orders and temporal dependencies; reduced charge margin. They are not measured policy difficulty. `MOVE` invokes the native planner, so these cases do not establish low-level route-finding ability. Visual handoff is the intended first-person search setting and is explicitly withheld from positive admission until its required controls exist.

## Case construction and verification

Generation imports the native `Map`, graph and food/mechanics data. Endpoints use native door/POI nodes, not building centers or invented graph locations. Roads are stored in meters; UE/API coordinates and graph distances are centimeters. Disconnected graph endpoints are rejected. Native shortest-path distance is retained, while tie-dependent internal route identities are excluded from the case hash.

The generator samples a fixed start, fixed orders and resource conditions. The policy receives `public/` instructions, API examples, food summaries, budgets and relevant map POIs. Full `cases/` recipes also contain private native-binding centers, graph witnesses and shortcut checks. These private certificates and the research preview never enter policy input. Public order IDs are transaction handles; object-identity overlays are not added to first-person RGB.

Native execution installs the exact orders with a seeded `FixedOrderPool`, suppressing automatic replenishment. It reuses native `Order`, food handling and settlement. This avoids a source issue in the current Gym reset: it seeds global RNGs but omits `seed` when constructing `OrderManager`, whose local order RNG otherwise uses a fixed fallback. Upstream files stay unchanged; configuration copies and output are isolated under each run.

The native `MapObserver` must run even without a display: its timer synchronizes UE poses into the courier state and charges movement resource costs. The wrapper enables that observer with Qt offscreen and seeds its initial polling baseline to the fixture start. Disabling the observer leaves logical coordinates stale while the UE actor moves. The Slurm launcher checks Qt before starting UE and selects the environment's FreeType for the Python process to avoid a diagnosed HarfBuzz symbol conflict.

The evaluator checks the recorded initial conditions, accept/pickup/packing/delivery order, native actor proximity, case-relative deadlines, required delivery method, allowed transport and position agreement. Gym's constant-zero reward and agent-provided success flags are not success evidence. Native records include separate UE and Python positions, resources, order state and raw snapshots. Native wall-clock timing is measured; deterministic recipes do not imply bitwise deterministic UE replay.

`STEP_FORWARD` currently calls UE movement without directly updating courier coordinates in its handler. Visual-handoff admission must establish whether the rest of the native stack synchronizes those coordinates before relying on delivery proximity. The admission worker currently supports the first three families and fails explicitly for visual-handoff candidates.

## Commands

From the SpatialAgent root:

```bash
module load conda
conda activate spatialagent
python 04_AgentTask/CustomAgentTasks/delivery_cases/run.py generate \
  --output 05_Experiment/results/delivery-cases-20261026 \
  --maps small-city-11 --levels 1 2 3 --per-level 3
```

The output directory must be fresh. This generates 36 recipes: 27 ordinary graph-screened candidates and 9 visual-handoff candidates. Graph screening is not UE admission.

```bash
python 04_AgentTask/CustomAgentTasks/delivery_cases/present.py \
  --batch 05_Experiment/results/delivery-cases-20261026
python -m unittest discover -s 04_AgentTask/CustomAgentTasks/delivery_cases/tests -v
```

After checking the current account and resource availability, submit the first native admission:

```bash
sbatch scripts/experiments/delivery_case_admission.slurm
```

The job selects the first L1 pickup/delivery recipe, launches its matching UE map on the existing UnrealCV port 9000 and runs a privileged mechanical witness, followed by negative probes using a separate accepted-and-ready control-order fixture. The remote-pickup check is therefore not confounded by food preparation. Four underlying API calls are reserved for admission controls outside the public task action budget. It does not start a website or call a model. Artifacts go to `05_Experiment/results/delivery-case-JOB_ID/`: `episode/validation.json`, `episode/native-trace.jsonl`, public observations, raw snapshots, isolated checkout, source snapshot and simulator logs. A failed witness remains a failed case for diagnosis; it is not corrected into a VLM result or published as a successful demo.

## Files

| File | Responsibility |
| --- | --- |
| `cases.py` | Native-map recipes, graph witnesses, public input boundary and evaluator |
| `native.py` | Fixed-order fixture, real API execution, physical state and image recording |
| `run.py` | Generate a batch or admit one native case, with experiment records |
| `present.py` | Research-only graph construction examples; not UE images |
| `tests/` | Seed reproducibility, native parser compatibility, shortcut and verifier checks |

The active batch and experiment records describe actual progress. No new native rollout or VLM performance is implied by generation alone. The current eight-task homepage remains separate until a real DeliveryBench case passes admission and has recorded media.

## Recorded development evidence

The L1 pickup/delivery case `pickup_delivery-830181257353` passed native admission in job 389653, including remote-pickup and delivery-without-pickup negative controls. Qwen3.8-27B then completed the same case in job 389762 with seven actual decisions and strict success: 240.69 simulated seconds against a 340.27-second deadline. See the [model run](../../../05_Experiment/results/delivery-qwen-389762/episode/README.md). This single workflow-control case does not establish visual search or a benchmark success rate. The shared research page imports its exact public request text, images, responses and resulting observations.

To run the configured local model in a fresh allocation, use:

```bash
sbatch --export=ALL,DELIVERY_MODEL_RUN=1 scripts/experiments/delivery_case_admission.slurm
```

The script checks Qt, starts vLLM, confirms the served model identity, starts UE and performs at most 16 actual model calls. It excludes loopback from HTTP proxies and applies the diagnosed FreeType override only to Python. No new website port is opened. Requests, raw responses, snapshots, native state and evaluation are retained under `05_Experiment/results/delivery-qwen-JOB_ID/`.
