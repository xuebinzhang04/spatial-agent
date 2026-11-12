import argparse
import os
import socket
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("--pids", nargs="*", type=int, default=[])
args = parser.parse_args()
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    for pid in args.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"Service process {pid} exited; inspect service logs")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            assert response.status == 200
        with socket.create_connection(("127.0.0.1", 9000), timeout=2):
            pass
        print("vLLM and UnrealCV ready", flush=True)
        break
    except (OSError, AssertionError):
        time.sleep(5)
else:
    raise SystemExit("Services did not become ready; inspect vllm.log and simulator.log")
