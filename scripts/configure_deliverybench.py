"""Configure one isolated-output episode. Run sequentially, not concurrently."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "04_AgentTask/SpatialAgentBench/DeliveryBench"
parser = argparse.ArgumentParser()
parser.add_argument("--map", default="small-city-11")
parser.add_argument("--run-id", default="qwen3.8-smoke")
args = parser.parse_args()
if not (REPO / "maps" / args.map).is_dir():
    parser.error("Unknown map")
output = ROOT / "05_Experiment/results" / args.run_id
(output / "reports").mkdir(parents=True, exist_ok=True)
config = {
    "gym_env": {"base_dir": str(REPO), "ue_ip": "127.0.0.1", "ue_port": 9000,
                "map_name": args.map, "simulator_package": "DeliveryBench/Linux/gym_citynav"},
    "lifecycle": {"export_path": str(output / "reports")},
    "trajectory_output_dir": str(output / "trajectories"),
    "agent_count": 1, "multi_agent": False,
}
model = {"provider": "openai", "url": "http://127.0.0.1:8000/v1",
         "model": "Qwen3.8-27B", "temperature": 1.0, "top_p": 0.95,
         "max_tokens": 16384}
models = {"agents": {"1": model}, "default": model}
for name, data in [("experiment_config.json", config), ("models.json", models)]:
    content = json.dumps(data, indent=2) + "\n"
    (REPO / "vlm_delivery/input" / name).write_text(content)
    (output / name).write_text(content)
(output / "game_mechanics_config.json").write_bytes(
    (REPO / "vlm_delivery/input/game_mechanics_config.json").read_bytes())
print(output)
