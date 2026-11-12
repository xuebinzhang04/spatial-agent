#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$ROOT/05_Experiment/queue_inspector"
LOG_DIR="$ROOT/05_Experiment/logs"
PYTHON="$ROOT/../conda/envs/spatialagent/bin/python"
PID_FILE="$APP_DIR/queue-inspector-$(hostname -s).pid"
LOG_FILE="$LOG_DIR/queue-inspector.log"
PORT="${QUEUE_INSPECTOR_PORT:-18766}"
SSH_HOST="${QUEUE_INSPECTOR_SSH_HOST:-tillicum.hyak.uw.edu}"

is_running() {
  local existing_pid process_args
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]]; then
      process_args="$(ps -ww -p "$existing_pid" -o args= 2>/dev/null || true)"
      if [[ "$process_args" == *"-m uvicorn app:app --app-dir $APP_DIR --host 127.0.0.1 --port $PORT"* ]]; then
        kill -0 "$existing_pid" 2>/dev/null && return 0
      fi
    fi
  fi
  # Recover a server whose shared/legacy PID file was lost or overwritten.
  existing_pid="$(ps -ww -u "$(id -u)" -o pid=,args= | awk -v target="-m uvicorn app:app --app-dir $APP_DIR --host 127.0.0.1 --port $PORT" '$2 ~ /(^|\/)python[0-9.]*$/ && index($0, target) {print $1; exit}')"
  if [[ -n "$existing_pid" ]]; then
    echo "$existing_pid" > "$PID_FILE"
    return 0
  fi
  return 1
}

case "${1:-status}" in
  restart)
    bash "$0" stop
    bash "$0" start
    ;;
  start)
    if is_running; then
      echo "Queue Inspector is already running (PID $(cat "$PID_FILE")), http://127.0.0.1:$PORT"
      exit 0
    fi
    mkdir -p "$LOG_DIR"
    export SPATIAL_ROOT="$ROOT"
    nohup "$PYTHON" -m uvicorn app:app --app-dir "$APP_DIR" --host 127.0.0.1 --port "$PORT" \
      >> "$LOG_FILE" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    sleep 1
    if ! is_running; then
      rm -f "$PID_FILE"
      echo "Queue Inspector failed to start; inspect $LOG_FILE" >&2
      exit 1
    fi
    echo "Queue Inspector is running at http://127.0.0.1:$PORT (PID $(cat "$PID_FILE"))"
    echo "On your workstation terminal, open the tunnel with: ssh -N -L $PORT:127.0.0.1:$PORT $USER@$SSH_HOST"
    echo "Log: $LOG_FILE"
    ;;
  stop)
    if is_running; then
      stopped_pid="$(cat "$PID_FILE")"
      kill "$stopped_pid"
      for ((attempt = 0; attempt < 50; attempt++)); do
        if ! kill -0 "$stopped_pid" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
      if kill -0 "$stopped_pid" 2>/dev/null; then
        echo "Queue Inspector is still stopping (PID $stopped_pid); retry shortly." >&2
        exit 1
      fi
      rm -f "$PID_FILE"
      echo "Queue Inspector stopped."
    else
      rm -f "$PID_FILE"
      echo "Queue Inspector is not running."
    fi
    ;;
  status)
    if is_running; then
      echo "Queue Inspector is running at http://127.0.0.1:$PORT (PID $(cat "$PID_FILE"))"
    else
      echo "Queue Inspector is not running. Start it with: bash scripts/experiments/queue_inspector.sh start"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
