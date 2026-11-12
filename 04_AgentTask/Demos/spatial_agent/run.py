"""Serve the research page, recordings, online environments, and model inference."""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import threading
import time

from session import DeliveryAdapter, ObservedSession, ROOT, Session, jsonable

HERE = Path(__file__).resolve().parent


def stop_previous(port):
    """Replace only a same-user server running this exact project entrypoint."""
    result=subprocess.run(['ss','-H','-ltnp',f'sport = :{port}'],capture_output=True,text=True,check=True)
    if not result.stdout.strip():return
    pids={int(p) for p in re.findall(r'pid=(\d+)',result.stdout)}
    if not pids:raise RuntimeError(f'Port {port} is occupied by an unidentified service; no process was stopped')
    for pid in pids:
        proc=Path('/proc')/str(pid)
        try:
            command=(proc/'cmdline').read_bytes().decode().split('\0')
            cwd=(proc/'cwd').resolve()
            owned=proc.stat().st_uid==os.getuid()
            ours=any(Path(arg).name=='run.py' and (cwd/arg).resolve()==Path(__file__).resolve() for arg in command)
        except FileNotFoundError:continue
        if not owned or not ours:raise RuntimeError(f'Port {port} belongs to another service (PID {pid}); no process was stopped')
    for pid in pids:
        print(f'Restarting SpatialAgent on port {port} (previous PID {pid})',flush=True)
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for _ in range(100):
        state=subprocess.run(['ss','-H','-ltn',f'sport = :{port}'],capture_output=True,text=True,check=True)
        if not state.stdout.strip():return
        time.sleep(.1)
    raise RuntimeError(f'Previous server is still closing port {port}; no alternate port was opened')


def handler(session, stop):
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            if not self.path.startswith('/api/'):
                immutable=self.path.startswith(('/assets/','/static/media/'))
                self.send_header('Cache-Control','public, max-age=31536000, immutable' if immutable else 'no-store')
            super().end_headers()

        def send_head(self):
            path=Path(self.translate_path(self.path))
            self.range_remaining=None
            if path.suffix.lower()!='.mp4' or not path.is_file():return super().send_head()
            size=path.stat().st_size;start,end=0,size-1
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                if not match or not any(match.groups()):return self.invalid_range(size)
                a,b=match.groups()
                if a:start=int(a);end=min(int(b),end) if b else end
                else:start=max(0,size-int(b))
                if start>end or start>=size:return self.invalid_range(size)
            file=path.open('rb');file.seek(start)
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type','video/mp4')
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1))
            if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers();self.range_remaining=end-start+1
            return file

        def invalid_range(self,size):
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
            self.send_header('Content-Length','0');self.end_headers()

        def copyfile(self,source,output):
            if getattr(self,'range_remaining',None) is None:return super().copyfile(source,output)
            try:
                while self.range_remaining>0:
                    data=source.read(min(65536,self.range_remaining))
                    if not data:break
                    output.write(data);self.range_remaining-=len(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def reply(self, value, code=200):
            data = json.dumps(value, default=jsonable).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/api/health':
                self.reply({'service':'spatialagent-demo','workspace':str(ROOT),'pid':os.getpid(),
                            'node':os.uname().nodename,'port':self.server.server_port})
            elif self.path == "/api/capabilities":
                self.reply(session.capabilities())
            elif self.path == "/api/state":
                self.perform("state", {})
            elif self.path.startswith("/api/"):
                self.reply({"error": "Unknown API route"}, 404)
            else:
                super().do_GET()

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.reply({"error": "Cross-origin request rejected"}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply({"error": "JSON required"}, 415)
                return
            if self.path not in ("/api/reset", "/api/action", "/api/infer", "/api/stop", "/api/model"):
                self.reply({"error": "Unknown API route"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32768:
                    raise ValueError("Invalid JSON body size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Expected an object")
            except (ValueError, TypeError) as error:
                self.reply({"error": str(error)}, 400)
                return
            self.perform(self.path.rsplit("/", 1)[-1], value)

        def perform(self, operation, value):
            if isinstance(session, ObservedSession) and operation not in ('state', 'stop'):
                self.reply({'error':'Observation-only viewer; inference and environment control belong to the worker'}, 409)
                return
            if operation == "stop":
                self.reply({"status": "stopping"})
                stop()
                return
            if not session.lock.acquire(blocking=False):
                self.reply({"error": "An environment/model operation is already running"}, 409)
                return
            try:
                if operation in ("action", "infer"):
                    for field in ("environment", "task"):
                        expected = value.get("expected_" + field)
                        if expected is not None and (not session.last or session.last.get(field) != expected):
                            raise ValueError("Selected simulator/task does not match the active session; reset or read its state")
                if operation == "state":
                    result = session.state()
                elif operation == "reset":
                    result = session.reset(value)
                elif operation == "action":
                    result = session.step(value.get("action", value.get("command")))
                elif operation == "infer":
                    if type(value.get("execute", False)) is not bool:
                        raise ValueError("execute must be a boolean")
                    if value.get("model", session.model) != session.model:
                        raise ValueError("This server has no endpoint for the selected model")
                    result = session.infer(value.get("execute", False))
                elif operation == "model":
                    result = session.connect_model(value.get('model'), value.get('url'))
                else:
                    result = {"status": "stopping"}
                    stop()
                self.reply(result)
            except Exception as error:
                session.log("api_errors.jsonl", {"operation": operation, "error": str(error)})
                self.reply({"error": f"{type(error).__name__}: {error}"}, 400)
            finally:
                session.lock.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--environment", choices=("shape3d", "native", "deliverybench"))
    parser.add_argument("--task", default="shape_identification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--map", default="small-city-11")
    parser.add_argument("--model-url", help="Existing model service URL ending in /v1")
    parser.add_argument("--model", help="Actual model ID served by the endpoint")
    parser.add_argument("--observe-run", type=Path, help="Display a worker run without loading a model or environment")
    parser.add_argument("--scripted-tour", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "05_Experiment/results" / ("demo-session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    args = parser.parse_args()
    if args.observe_run and (args.environment or args.model or args.model_url or args.scripted_tour):
        parser.error('--observe-run is independent of environment/model execution arguments')
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("Use a loopback host and SSH forwarding for simulator/model control")
    site = HERE / "site"
    stop_previous(args.port)
    session = ObservedSession(args.observe_run) if args.observe_run else Session(args.run_dir, args.model_url, args.model)
    stopping = threading.Event()
    server = None
    try:
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        signal.signal(signal.SIGINT, lambda *_: stopping.set())
        if args.environment == "deliverybench":
            session.adapter = DeliveryAdapter(session.run, args.map, args.seed)
            session.last = session.adapter.snapshot()
            session.metadata["environment"]["name"] = "SimWorld dedicated gym_citynav build"
            session.metadata["environment"]["id"] = "simworld_delivery"
            session.metadata["task_name"] = "Delivery (DeliveryBench)"
            session.metadata["dataset"]["tasks"] = [args.map]
            session.metadata["parameters"].update(map=args.map, seed=args.seed)
            session.save_metadata()
            session.publish()
            session.log('session.jsonl', {'event':'reset','state':session.last})
        elif args.environment:
            session.reset({key: getattr(args, key) for key in ("environment", "task", "seed", "difficulty", "budget", "scene")})
        server = ThreadingHTTPServer((args.host, args.port), partial(handler(session, stopping.set), directory=str(site)))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"READY http://{args.host}:{server.server_port} on {os.uname().nodename}", flush=True)
        print(f'VS Code Remote SSH: fixed local port {args.port}; use the browser address opened by Ports forwarding.',flush=True)
        if args.scripted_tour and isinstance(session.adapter, DeliveryAdapter):
            def tour():
                try:
                    with session.lock:
                        for command in ('VIEW_ORDERS()', 'TURN_AROUND(angle=60, direction="left")', 'STEP_FORWARD()'):
                            session.step(command)
                        env = session.adapter.env
                        orders = env._invoker.call(lambda: env.om.list_orders())
                        if orders.get("ok") and orders["result"]:
                            session.step(f"ACCEPT_ORDER({int(orders['result'][0].id)})")
                    (session.run / "tour_complete.txt").write_text(datetime.now(timezone.utc).isoformat())
                except Exception as error:
                    (session.run / "tour_failure.txt").write_text(str(error))
            threading.Thread(target=tour, daemon=True).start()
        if isinstance(session.adapter, DeliveryAdapter):
            from PyQt5.QtCore import QTimer
            timer = QTimer()
            timer.timeout.connect(lambda: session.adapter.env._app.quit() if stopping.is_set() else None)
            timer.start(250)
            session.adapter.env.run_qt_loop()
        else:
            while not stopping.wait(.25):
                pass
    except Exception as error:
        if not isinstance(session, ObservedSession):
            session.metadata["slurm"].update(state="FAILED", exit_code=1, failure_reason=str(error))
        raise
    finally:
        if server:
            server.shutdown()
            server.server_close()
        session.close()


if __name__ == "__main__":
    main()
