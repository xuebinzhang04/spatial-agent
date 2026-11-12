"""Small DeepSeek Harness agent loop with a replaceable simulator adapter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


ACTIONS = ("north", "south", "east", "west")


@dataclass
class Transition:
    observation: str
    reward: float
    done: bool


class Environment(Protocol):
    """Implement this interface to connect a simulator later."""

    def reset(self) -> str: ...

    def step(self, action: str) -> Transition: ...


class GridWorld:
    """A deterministic 3x3 environment, with a wall in the centre."""

    def __init__(self) -> None:
        self.position = (0, 0)
        self.goal = (2, 2)
        self.wall = (1, 1)

    def reset(self) -> str:
        self.position = (0, 0)
        return self.observe()

    def observe(self) -> str:
        return (
            f"3x3 grid; position={self.position}; goal={self.goal}; "
            f"wall={self.wall}; valid actions={','.join(ACTIONS)}. "
            "Coordinates are (x,y); east increases x, south increases y."
        )

    def step(self, action: str) -> Transition:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action!r}")
        dx, dy = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
        }[action]
        x, y = self.position
        next_position = (x + dx, y + dy)
        if (0 <= next_position[0] < 3 and 0 <= next_position[1] < 3
                and next_position != self.wall):
            self.position = next_position
        done = self.position == self.goal
        return Transition(self.observe(), 1.0 if done else -0.01, done)


def parse_action(response: str) -> str:
    """Only accept a single JSON action; never execute model text as code."""
    text = response.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"action"}:
        raise ValueError(f"Expected {{\"action\": ...}}, got {response!r}")
    action = value["action"]
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action!r}")
    return action


def prompt(observation: str) -> str:
    return (
        "You control a tiny grid simulator. Reach the goal in as few steps as possible. "
        "The centre cell is blocked. Respond ONLY with one JSON object, "
        'for example {"action":"east"}. No prose or markdown.\n'
        f"Current observation: {observation}"
    )


def mock_action(environment: GridWorld) -> str:
    """Deterministic offline policy for checking the environment/trace loop."""
    x, y = environment.position
    return "east" if x < 2 else "south"


def run_demo(
    mode: str,
    max_steps: int,
    output: Path | None,
    environment: Environment | None = None,
) -> dict:
    environment = environment or GridWorld()
    observation = environment.reset()
    trace: list[dict] = []
    session_id = f"grid-{uuid.uuid4().hex}"

    def loop(choose_action=None) -> None:
        nonlocal observation
        for index in range(max_steps):
            if choose_action is None:
                if not isinstance(environment, GridWorld):
                    raise ValueError("Mock policy only supports GridWorld")
                action = mock_action(environment)
            else:
                action = parse_action(choose_action(observation))
            transition = environment.step(action)
            trace.append({
                "step": index + 1,
                "action": action,
                **asdict(transition),
            })
            observation = transition.observation
            print(f"step {index + 1}: {action:5s} "
                  f"reward={transition.reward} done={transition.done}")
            if transition.done:
                break

    if mode == "mock":
        loop()
    elif mode == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is missing; use --mode mock for the offline demo")
        from deepseek_harness import DeepSeekHarness

        # Isolate the SDK home and its tool workspace for this short demo.
        with tempfile.TemporaryDirectory(prefix="spatialagent-dsh-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            with DeepSeekHarness(
                provider="deepseek-official",
                model=os.environ.get("DSH_MODEL", "deepseek-flash"),
                profile="sdk-minimal",
                cwd=str(workspace),
                dsh_home=str(root / "home"),
                max_tokens=512,
            ) as harness:
                def choose_deepseek(current_observation: str) -> str:
                    result = harness.run(prompt(current_observation), session_id=session_id)
                    if not result.final_response:
                        raise RuntimeError(
                            f"Harness returned no final response; finish_reason={result.finish_reason}"
                        )
                    return result.final_response

                loop(choose_deepseek)
    elif mode == "qwen":
        from deepseek_harness import DeepSeekHarness

        patch = Path(__file__).with_name("qwen.patch.yml")
        with tempfile.TemporaryDirectory(prefix="spatialagent-qwen-dsh-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            with DeepSeekHarness(
                provider="qwen-local",
                model=os.environ.get("QWEN_MODEL", "Qwen3.8-27B"),
                reasoning_effort="low",
                profile="sdk-minimal",
                patches=(str(patch),),
                cwd=str(workspace),
                dsh_home=str(root / "home"),
                env={
                    "QWEN_API_KEY": "local-unused",
                    "QWEN_BASE_URL": os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
                },
                max_tokens=1024,
            ) as harness:
                def choose_qwen(current_observation: str) -> str:
                    result = harness.run(prompt(current_observation), session_id=session_id)
                    if not result.final_response:
                        raise RuntimeError(
                            f"Harness returned no final response; finish_reason={result.finish_reason}"
                        )
                    return result.final_response

                loop(choose_qwen)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    report = {
        "mode": mode,
        "model": (
            os.environ.get("DSH_MODEL", "deepseek-flash") if mode == "deepseek" else
            os.environ.get("QWEN_MODEL", "Qwen3.8-27B") if mode == "qwen" else None
        ),
        "session_id": session_id if mode in ("deepseek", "qwen") else None,
        "success": bool(trace and trace[-1]["done"]),
        "steps": trace,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"trace: {output}")
    print(f"success={report['success']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "deepseek", "qwen"), default="mock")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON trajectory path")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    run_demo(args.mode, args.max_steps, args.output)


if __name__ == "__main__":
    main()
