#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cvc-bats-auto"
CONFIG_PATH="$CONFIG_DIR/config.json"
STATE_PATH="${XDG_STATE_HOME:-$HOME/.local/state}/cvc-bats-auto/state.json"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$SERVICE_DIR/cvc-bats-auto.service"
DEFAULT_WATCH_DIR="$HOME/Nagrania nietoperzy"
PRESET_PATH="$REPO_DIR/presets/default.json"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    echo "Run this installer as your normal desktop user, without sudo." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
    exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe is required. Install FFmpeg first (for Debian/Ubuntu: sudo apt install ffmpeg)." >&2
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemd is required by the Linux installer." >&2
    exit 1
fi

if [[ ! -f "$PRESET_PATH" ]]; then
    echo "Missing required preset: $PRESET_PATH" >&2
    exit 1
fi

if [[ -f "$CONFIG_PATH" ]]; then
    CURRENT_WATCH_DIR="$(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("watch_directory", ""))
except (OSError, ValueError):
    print("")
PY
)"
    if [[ -n "$CURRENT_WATCH_DIR" ]]; then
        DEFAULT_WATCH_DIR="$CURRENT_WATCH_DIR"
    fi
fi

if [[ $# -ge 1 ]]; then
    WATCH_INPUT="$1"
elif [[ -t 0 ]]; then
    read -r -e -p "Folder nagrań nietoperzy [$DEFAULT_WATCH_DIR]: " WATCH_INPUT
else
    WATCH_INPUT=""
fi
WATCH_INPUT="${WATCH_INPUT:-$DEFAULT_WATCH_DIR}"
WATCH_DIR="$(python3 - "$WATCH_INPUT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
WATCH_CHANGED=false
if [[ -n "${CURRENT_WATCH_DIR:-}" && "$CURRENT_WATCH_DIR" != "$WATCH_DIR" ]]; then
    WATCH_CHANGED=true
fi

mkdir -p -- "$WATCH_DIR" "$CONFIG_DIR" "$(dirname -- "$STATE_PATH")" "$SERVICE_DIR"

if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
    if ! python3 -m venv "$REPO_DIR/.venv"; then
        echo "Could not create the virtual environment. Install python3-venv and run the installer again." >&2
        exit 1
    fi
fi
"$REPO_DIR/.venv/bin/python" -m pip install --upgrade pip
"$REPO_DIR/.venv/bin/python" -m pip install -e "$REPO_DIR"

python3 - "$CONFIG_PATH" "$REPO_DIR" "$WATCH_DIR" "$PRESET_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path, repo_dir, watch_dir, preset_path = map(Path, sys.argv[1:])
config = {
    "repo_directory": str(repo_dir.resolve()),
    "watch_directory": str(watch_dir.resolve()),
    "preset_path": str(preset_path.resolve()),
    "output_directory_name": "output-default",
    "video_extensions": ".mp4,.avi,.mov,.mkv",
    "scan_interval_seconds": 60,
    "stable_for_seconds": 180,
    "retry_after_seconds": 3600,
}
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

python3 - "$SERVICE_PATH" "$REPO_DIR" "$CONFIG_PATH" "$STATE_PATH" <<'PY'
import sys
from pathlib import Path

service_path, repo_dir, config_path, state_path = map(Path, sys.argv[1:])

def quote(value: Path) -> str:
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'

python_path = repo_dir / ".venv" / "bin" / "python"
content = f"""[Unit]
Description=CVC Bats automatic sequential video analysis
After=default.target

[Service]
Type=simple
WorkingDirectory={quote(repo_dir)}
ExecStart={quote(python_path)} -m cvc_bats_auto --config {quote(config_path)} --state {quote(state_path)}
Restart=on-failure
RestartSec=15
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
service_path.write_text(content, encoding="utf-8")
PY

if [[ "$WATCH_CHANGED" == true ]]; then
    "$REPO_DIR/.venv/bin/python" -m cvc_bats_auto \
        --config "$CONFIG_PATH" --state "$STATE_PATH" --force-initialize
elif [[ ! -f "$STATE_PATH" ]]; then
    "$REPO_DIR/.venv/bin/python" -m cvc_bats_auto \
        --config "$CONFIG_PATH" --state "$STATE_PATH" --initialize
else
    echo "Existing automation state retained: $STATE_PATH"
fi

systemctl --user daemon-reload
systemctl --user enable --now cvc-bats-auto.service

if command -v loginctl >/dev/null 2>&1; then
    if ! loginctl enable-linger "$USER" >/dev/null 2>&1; then
        echo "Warning: user lingering could not be enabled. The service will start when you log in."
    fi
fi

echo
echo "CVC Bats Automat is ready."
echo "Watched folder: $WATCH_DIR"
echo "Preset: $PRESET_PATH"
echo "Output folder beside each source folder: output-default"
echo "Status: systemctl --user status cvc-bats-auto.service"
echo "Log: journalctl --user -u cvc-bats-auto.service -f"
