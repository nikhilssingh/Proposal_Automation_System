# backend/agent_status_tracker.py
import json, time, threading
from datetime import datetime
from pathlib import Path
from backend.path_utils import LOG_DIR

# ---------- Log / status file paths ----------
STATUS_FILE  = LOG_DIR / "agent_status.json"
LOG_FILE     = LOG_DIR / "agent_log.txt"
PIPELINE_START_FILE = LOG_DIR / "pipeline_start.txt"
PIPELINE_END_FILE   = LOG_DIR / "pipeline_end.txt"

# ---------- Agent list ----------
AGENTS = [
    "RFP Analyzer",
    "Context Retriever",
    "Table Summarizer",
    "Proposal Generator",
    "Strategy Optimizer",
    "Compliance Checker",
    "Scorer",
]

# ---------- Timing helpers ----------
def mark_pipeline_start() -> None:
    print(">>> mark_pipeline_start fired – writing file", flush=True)  # DEBUG
    PIPELINE_START_FILE.write_text(datetime.now().isoformat())

def mark_pipeline_end() -> None:
    print(">>> mark_pipeline_end fired – writing file", flush=True)    # DEBUG
    PIPELINE_END_FILE.write_text(datetime.now().isoformat())

def get_pipeline_start() -> str:
    return PIPELINE_START_FILE.read_text().strip() if PIPELINE_START_FILE.exists() else ""

def get_pipeline_end() -> str:
    return PIPELINE_END_FILE.read_text().strip() if PIPELINE_END_FILE.exists() else ""

# ---------- Status handling ----------
UPDATE_COOLDOWN = 1          # minimum seconds between writes
_status_lock     = threading.Lock()
_last_update_time = 0

def reset_pipeline_timers() -> None:
    """Delete start/end timestamp files."""
    if PIPELINE_START_FILE.exists():
        PIPELINE_START_FILE.unlink()
    if PIPELINE_END_FILE.exists():
        PIPELINE_END_FILE.unlink()

def reset_status() -> None:
    """Initialise all agents to ⏳ Pending and clear audit log."""
    status = {agent: {"state": "🕓 Waiting", "timestamp": None} for agent in AGENTS}

    with _status_lock:
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        LOG_FILE.write_text("")

def update_status(agent: str, new_status: str, *, force: bool = False) -> None:
    """
    Thread-safe status update.
    Set force=True to bypass the UPDATE_COOLDOWN debounce.
    """
    global _last_update_time
    now = time.time()

    if not force and now - _last_update_time < UPDATE_COOLDOWN:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _status_lock:
        if not STATUS_FILE.exists():
            reset_status()

        data = json.loads(STATUS_FILE.read_text())
        if data.get(agent, {}).get("state") != new_status:
            data[agent] = {"state": new_status, "timestamp": timestamp}
            STATUS_FILE.write_text(json.dumps(data, indent=2))
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(f"[{timestamp}] {agent}: {new_status}\n")

        _last_update_time = now

def get_status() -> dict:
    with _status_lock:
        if not STATUS_FILE.exists():
            reset_status()
        return json.loads(STATUS_FILE.read_text())

def get_log() -> str:
    with _status_lock:
        if not LOG_FILE.exists():
            return "📭 No logs yet. Upload an RFP to begin tracking."
        return LOG_FILE.read_text().strip() or "📭 No logs yet. Upload an RFP to begin tracking."
