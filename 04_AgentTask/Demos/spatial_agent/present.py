"""Export research slides, interaction filmstrips, and videos from public records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH, HEIGHT = 1600, 900
INK, MUTED, ACCENT = "#111111", "#555555", "#111111"
FONTS = Path("/usr/share/fonts/cantarell")


def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ("Cantarell-Bold.otf" if bold else "Cantarell-Regular.otf")), size)


def write(image, value, x, y, width, size=25, color=INK, bold=False):
    draw = ImageDraw.Draw(image)
    face = font(size, bold)
    for paragraph in str(value).splitlines():
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=face) > width and line:
                draw.text((x, y), line, font=face, fill=color)
                y += int(size * 1.32)
                line = word
            else:
                line = candidate
        draw.text((x, y), line, font=face, fill=color)
        y += int(size * 1.32)
    return y


def page(title, subtitle):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)
    write(image, title, 55, 65, 1490, 40, bold=True)
    write(image, subtitle, 55, 135, 1490, 24, MUTED)
    draw.line((55, 846, 1545, 846), fill="#dce3e7", width=2)
    write(image, "SpatialAgent · Recorded interaction examples", 55, 864, 1490, 18, MUTED)
    return image


def block(image, label, value, x, y, width, size=25):
    y = write(image, label.upper(), x, y, width, 19, ACCENT, True)
    return write(image, value, x, y + 9, width, size) + 24


def paste_observation(image, source, box):
    with Image.open(source) as original:
        rgb = original.convert("RGB")
        # Preserve the complete observation, including any existing task overlays.
        resample = Image.Resampling.NEAREST if min(rgb.size) <= 128 else Image.Resampling.LANCZOS
        fitted = ImageOps.contain(rgb, (box[2] - box[0], box[3] - box[1]), method=resample)
        image.paste(fitted, (box[0] + (box[2] - box[0] - fitted.width) // 2,
                            box[1] + (box[3] - box[1] - fitted.height) // 2))


def action(frame):
    value = frame["action"]
    if not value:
        return "Initial observation"
    if isinstance(value, str):
        return value
    extras = " ".join(f"{key}={value[key]}" for key in ("object_id", "slot_id", "candidate_id") if key in value)
    return value.get("action", "Action") + (" " + extras if extras else "")


def select_steps(episode, annotation):
    frames = episode["frames"]
    selected = {0, len(frames) - 1}
    first_events = set()
    for index, frame in enumerate(frames[1:], 1):
        kind = (frame["action"] or {}).get("action")
        changed_phase = frame["observation"].get("phase") != frames[index - 1]["observation"].get("phase")
        if changed_phase or kind in annotation["events"] and kind not in first_events:
            selected.add(index)
            selected.add(index - 1)
            first_events.add(kind)
    # Filmstrips highlight a single causal transition with its real preceding view.
    if episode["task"] == "dynamic_replanning":
        event = next(i for i in sorted(selected) if frames[i]["observation"].get("phase") == "after_change")
    elif episode["task"] == "dependency_doors":
        event = next(i for i in sorted(selected) if (frames[i]["action"] or {}).get("action") == "OpenObject")
    elif episode["task"] == "relational_rearrangement":
        event = next(i for i in sorted(selected) if (frames[i]["action"] or {}).get("action") == "PLACE")
    elif episode["task"] == "pose_alignment":
        event = next(i for i in sorted(selected) if (frames[i]["action"] or {}).get("action", "").startswith("OBJECT_ROTATE"))
    elif episode["task"] in ("missing_object", "moved_object"):
        event = next(i for i in sorted(selected) if (frames[i]["action"] or {}).get("action") == "MEMORIZE")
    elif episode["task"] == "memory_maze":
        event = next(i for i in sorted(selected) if (frames[i]["action"] or {}).get("action") == "MARK")
    else:
        event = 1
    return [max(0, event - 1), event, len(frames) - 1], sorted(selected)


def filmstrip(episode, annotation, site, steps, failure=False):
    subtitle = f"{episode['agent']} | {episode['scene']} | seed {episode['seed']} | L{episode['difficulty']} | selected steps, not continuous time"
    image = page(episode["task"].replace("_", " ").capitalize(), subtitle)
    write(image, annotation["goal"], 55, 188, 1490, 27)
    for column, index in enumerate(steps):
        frame = episode["frames"][index]
        x = 55 + column * 500
        paste_observation(image, site / frame["image"], (x + 6, 253, x + 474, 612))
        write(image, f"STEP {frame['step']:02d} / {frame['observation'].get('phase', 'interaction')}", x, 628, 480, 20, ACCENT, True)
        write(image, action(frame), x, 660, 480, 23, bold=True)
    outcome = "STRICT SUCCESS" if episode["result"]["success"] else "STRICT FAILURE"
    if failure:
        caption = "Change identified; prior-location accuracy = 0. Reward = 0.925; strict success = false."
    else:
        caption = annotation["verification"]
    write(image, outcome + " / " + caption, 55, 730, 1490, 24, ACCENT if episode["result"]["success"] else "#a03c28", True)
    write(image, annotation["lesson"], 55, 793, 1490, 21, MUTED)
    return image


def movie_frame(episode, annotation, site, index):
    frame = episode["frames"][index]
    heading = "Delivery on SimWorld (gym_citynav build)" if episode["task"]=="deliverybench" else episode["task"].replace("_", " ").capitalize()
    image = page(heading, f"{episode['agent']} | {episode['scene']} | seed {episode['seed']} | discrete-step replay, 2 steps/s")
    paste_observation(image, site / frame["image"], (55, 198, 1025, 806))
    y = block(image, "Task objective", annotation["goal"], 1060, 195, 480, 25)
    y = block(image, f"Executed action / step {frame['step']}", action(frame), 1060, y, 480, 26)
    y = block(image, "Observation phase", frame["observation"].get("phase", "interaction"), 1060, y, 480)
    feedback = frame["observation"].get("feedback")
    if feedback:
        y = block(image, "Environment feedback", feedback, 1060, y, 480, 23)
    if index == len(episode["frames"]) - 1:
        status = "RECORDING CUTOFF / NOT EVALUATED" if episode["result"]["success"] is None else "STRICT SUCCESS" if episode["result"]["success"] else "STRICT FAILURE"
        block(image, "Terminal evaluator result", status, 1060, y, 480, 25)
    else:
        write(image, "Outcome is evaluated at termination.", 1060, max(y, 724), 480, 22, MUTED)
    return image


def export_video(episode, annotation, site, path, ffmpeg):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "1280x720", "-framerate", "2", "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "2", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-r", "12", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(len(episode["frames"])):
            frame = movie_frame(episode, annotation, site, index).resize((1280, 720), Image.Resampling.LANCZOS)
            process.stdin.write(frame.tobytes())
        # Hold the terminal result for two additional seconds.
        for _ in range(4):
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait() != 0:
            raise RuntimeError(error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=HERE / "site")
    parser.add_argument("--output", type=Path, default=HERE / "presentation")
    parser.add_argument("--record", type=Path, default=ROOT / "05_Experiment/results" / ("interaction-presentation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--ffmpeg", default=str(Path(sys.executable).parent / "ffmpeg"))
    parser.add_argument("--delivery-only", action="store_true", help="Export the existing native Delivery recording without changing the task presentation")
    args = parser.parse_args()
    args.site, args.output, args.record = args.site.resolve(), args.output.resolve(), args.record.resolve()
    catalog_path = args.site / "data/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    annotations = json.loads((HERE / "examples.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    args.record.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    sources = {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__).resolve(), HERE / "examples.json", catalog_path)}
    metadata = {
        "run_id": args.record.name, "project_id": "interaction-presentation-" + uuid.uuid4().hex,
        "project_name": "Spatial interaction research presentation", "task_name": "Recorded interaction examples", "task_id": "interaction-presentation-v1",
        "purpose": "Explain spatial task mechanisms, verifiers, and limitations with recorded interaction examples",
        "dataset": {"name": "Existing public SpatialAgent episode catalog", "version": catalog["schema"], "split": "Development recordings", "tasks": list(annotations), "materials": [str(catalog_path.relative_to(ROOT))]},
        "experiment_type": "Media export from existing evidence; no new evaluation or training",
        "model": {"name": "Recorded Oracle and Qwen3.8-27B", "checkpoint": "Not loaded", "precision": "Not applicable"},
        "environment": {"name": "Pillow and CPU ffmpeg", "version": "Source frames unchanged"},
        "source": {"commit": "Not recorded: no workspace Git metadata", "entrypoint": str(Path(__file__).resolve().relative_to(ROOT)), "config": str((HERE / 'examples.json').relative_to(ROOT)), "sha256": sources},
        "parameters": {"movie_size": [1280, 720], "slides_size": [WIDTH, HEIGHT], "replay_steps_per_second": 2, "ffmpeg": args.ffmpeg, "video_threads": 2},
        "slurm": {"job_id": "Not applicable: light local media export", "partition": "Not applicable", "resources": "CPU rendering and encoding, two encoder threads, no GPU", "state": "RUNNING", "created_at": started, "started_at": started, "ended_at": "Not recorded", "exit_code": "Not recorded", "failure_reason": "Not recorded"},
        "artifacts": {"result_dir": str(args.record.relative_to(ROOT)), "report": str((args.output / 'research-presentation.pdf').relative_to(ROOT)), "primary_metrics": "Not applicable: no model evaluation", "trajectory": str(catalog_path.relative_to(ROOT)), "logs": []},
        "notes": "Oracle is privileged; model smoke is not a controlled policy comparison. Structural levels are uncalibrated. No external media copied; no RL or transfer claim."
    }
    manifest = args.record / "run_metadata.json"

    def save_metadata():
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    slides, examples = [], []
    try:
        if args.delivery_only:
            episode = next(e for e in catalog["episodes"] if e["task"] == "deliverybench")
            annotation = {"goal": "Delivery task on SimWorld's dedicated gym_citynav build: inspect orders and execute native courier actions. This short manual/script record is not a completed benchmark."}
            video = args.output / (episode["id"] + ".mp4")
            export_video(episode, annotation, args.site, video, args.ffmpeg)
            metadata["dataset"]["tasks"] = ["deliverybench"]
            metadata["artifacts"]["report"] = str(video.relative_to(ROOT))
            metadata["artifacts"]["sha256"] = {video.name: digest(video)}
            metadata["slurm"].update(state="COMPLETED", ended_at=datetime.now(timezone.utc).isoformat(), exit_code="0", failure_reason="Not applicable")
            print(f"Exported native delivery recording: {video}", flush=True)
            return
        image = page("Learning spatial reasoning through interaction", "Research scope -> concrete examples -> current evidence -> next experiments")
        y = block(image, "Research question", "Can verifiable, compositional interaction tasks supply a scalable learning signal for VLM spatial reasoning?", 55, 210, 880, 34)
        y = block(image, "Training unit", "Observation -> chosen action -> changed observation -> verified outcome", 55, y + 10, 880, 30)
        block(image, "Current stage", "Eight task families with recordings. Environment and task construction are demonstrated; RL gains and transfer remain open.", 55, y + 10, 880, 28)
        block(image, "What to look for", "What must the agent know?\nWhat does the action change?\nWhat makes an outcome correct?", 1030, 220, 480, 29)
        block(image, "Presentation format", "Real observations and actions, one mechanism per example, explicit policy provenance.", 1030, 490, 480, 27)
        slides.append(("01-research-question", image))

        image = page("A reusable task construction and learning loop", "Construction and recorded execution exist; policy difficulty calibration and RL are next")
        stages = [("Environment", "Simulator transitions, visible observations, physical validity"),
                  ("Task", "Sample instances, compose objectives, validate feasibility"),
                  ("Agent", "Harness: observation, memory, structured action, execution"),
                  ("Verifier", "Check final predicates, retain failures, audit trajectories")]
        for i, (label, text) in enumerate(stages):
            x = 55 + i * 380
            block(image, f"{i + 1}. {label}", text, x, 220, 340, 28)
        block(image, "Implemented", "Native AI2-THOR and procedural geometry tasks; seeded cases; public logs; Oracle feasibility and replay checks; four actual VLM smoke recordings.", 55, 485, 700, 28)
        block(image, "Proposed research", "Repeated rollouts to calibrate difficulty -> RL with verified rewards -> held-out scenes and task combinations -> fixed benchmark transfer evaluation.", 835, 485, 700, 28)
        write(image, "TMAX informs task generation. Related-work overlap must be addressed before claiming novelty.", 55, 760, 1480, 25, MUTED)
        slides.append(("02-construction-loop", image))

        for number, (task, annotation) in enumerate(annotations.items(), 3):
            episode = next(e for e in catalog["episodes"] if e["task"] == task and e["agent"] == "oracle")
            steps, events = select_steps(episode, annotation)
            image = filmstrip(episode, annotation, args.site, steps)
            slides.append((f"{number:02d}-{task}", image))
            video = args.output / (task + ".mp4")
            export_video(episode, annotation, args.site, video, args.ffmpeg)
            examples.append({"task": task, "episode": episode["id"], "agent": episode["agent"], "source": episode["provenance"],
                             "filmstrip_steps": steps, "event_steps": events, "frame_count": len(episode["frames"]), "video": video.name,
                             "outcome": episode["result"], "annotation": annotation})
            print(f"Exported {task}: {len(episode['frames'])} source frames", flush=True)

        episode = next(e for e in catalog["episodes"] if e["task"] == "missing_object" and e["agent"] != "oracle")
        annotation = dict(annotations["missing_object"], question="Partial progress can still be a strict failure", lesson="A diagnostic case: high reward does not establish correct spatial memory. No matched-policy comparison is implied.")
        steps, events = select_steps(episode, annotation)
        slides.append(("11-model-failure", filmstrip(episode, annotation, args.site, steps, failure=True)))
        video = args.output / "missing_object-model-failure.mp4"
        export_video(episode, annotation, args.site, video, args.ffmpeg)
        examples.append({"task": episode["task"], "episode": episode["id"], "agent": episode["agent"], "source": episode["provenance"],
                         "filmstrip_steps": steps, "event_steps": events, "frame_count": len(episode["frames"]), "video": video.name,
                         "outcome": episode["result"], "annotation": annotation})

        image = page("What this week's evidence establishes", "Task feasibility, valid interaction, and diagnostic model attempts")
        block(image, "Recorded task coverage", f"The shared catalog contains {len(catalog['episodes'])} records across {catalog['stats']['recorded_tasks']} task entries. This presentation selects eight native/geometry task examples and one model failure; legacy and delivery records remain available in the project page.", 55, 215, 700, 30)
        block(image, "Actual model evidence", "Qwen3.8-27B: 4 development smoke episodes, 0 strict successes. Missing-object identification partly succeeds but the previous location is wrong.", 835, 215, 700, 30)
        block(image, "Limits", "Oracle uses private state and may have a different budget. Native examples have low-resolution source RGB. Structural levels are not policy-calibrated. No RL or benchmark transfer result yet.", 55, 490, 700, 28)
        block(image, "Next decisive experiments", "Matched budgets and instances; paired observation and memory ablations; difficulty curves from repeated rollouts; then RL and held-out transfer evaluation.", 835, 490, 700, 28)
        slides.append(("12-evidence-and-next-steps", image))

        image = page("References that shape the examples", "Research mechanisms and evidence organization, rather than project-page styling")
        refs = [
            ("DeliveryBench", "Constraints and consequential planning decisions", "deliverybench.github.io"),
            ("VLM-3R", "Video + numbered frames + question + answer", "vlm-3r.github.io"),
            ("VeriWorld", "Executable tasks and matched diagnostic input conditions", "shengyun-peng.github.io/papers/veriworld"),
            ("CrashTwin / EgoHOI", "Physical failure criteria and action-driven consequences", "crash-twin.github.io | egohoi.github.io"),
            ("PACER / NavTrust", "Trace roles, partial progress, and controlled robustness tests", "pacer-corl.github.io | navtrust.github.io"),
            ("SpatialStack / RuleSmith / LSM", "Mechanism comparisons, calibration, spatial representation", "spatial-stack.github.io | adonis-galaxy.github.io/RuleSmith-website | largespatialmodel.github.io")]
        for i, (name, lesson, url) in enumerate(refs):
            y = 200 + i * 99
            write(image, name, 55, y, 430, 26, ACCENT, True)
            write(image, lesson, 490, y, 1050, 25)
            write(image, url, 490, y + 36, 1050, 19, MUTED)
        slides.append(("13-references", image))

        for name, image in slides:
            image.save(args.output / (name + ".png"))
        slides[0][1].save(args.output / "research-presentation.pdf", save_all=True,
                          append_images=[image for _, image in slides[1:]], resolution=120, title="SpatialAgent: learning through interaction")
        # A compact overview shows task diversity without any browser or server.
        overview = Image.new("RGB", (1600, 1000), "white")
        write(overview, "Eight spatial interaction tasks / recorded Oracle examples", 30, 16, 1540, 32, bold=True)
        for i, item in enumerate(examples[:8]):
            episode = next(e for e in catalog["episodes"] if e["id"] == item["episode"])
            x, y = 30 + (i % 4) * 395, 85 + (i // 4) * 435
            index = item["filmstrip_steps"][1]
            paste_observation(overview, args.site / episode["frames"][index]["image"], (x, y, x + 365, y + 285))
            write(overview, item["task"].replace("_", " "), x, y + 294, 365, 24, ACCENT, True)
            write(overview, action(episode["frames"][index]), x, y + 332, 365, 20)
            write(overview, f"Oracle | step {index} | strict success", x, y + 390, 365, 17, MUTED)
        overview.save(args.output / "overview.png")
        (args.output / "examples.json").write_text(json.dumps({"note": "Recorded steps; discrete replay at 2 steps/s, not original wall-clock speed. No private observations or external media.", "examples": examples}, indent=2) + "\n")
        sources.update({str(p.relative_to(ROOT)): digest(p) for p in (HERE / "examples.json", HERE / "present.py")})
        metadata["artifacts"]["sha256"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
        metadata["slurm"].update(state="COMPLETED", ended_at=datetime.now(timezone.utc).isoformat(), exit_code="0", failure_reason="Not applicable")
        metadata["parameters"].update(slides=len(slides), videos=len(examples))
        print(f"Completed: {len(slides)} slides, {len(examples)} videos; {args.output}", flush=True)
    except Exception as error:
        metadata["slurm"].update(state="FAILED", ended_at=datetime.now(timezone.utc).isoformat(), exit_code="1", failure_reason=str(error))
        raise
    finally:
        save_metadata()


if __name__ == "__main__":
    main()
