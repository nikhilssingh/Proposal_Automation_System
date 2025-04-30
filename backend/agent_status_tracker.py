import json
import time
from pathlib import Path
import threading
from datetime import datetime

STATUS_FILE = Path("logs/agent_status.json")
LOG_FILE = Path("logs/agent_log.txt")
UPDATE_COOLDOWN = 2  # seconds
_status_lock = threading.Lock()
_last_update_time = 0
STATUS_FILE.parent.mkdir(exist_ok=True, parents=True)

AGENTS = [
    "RFP Analyzer",
    "Context Retriever",
    "Table Summarizer",
    "Proposal Generator",
    "Strategy Optimizer",
    "Compliance Checker",
    "Scorer"
]

def reset_status():
    """Reset all agent statuses and clear audit log."""
    status = {
        agent: {"state": "⏳ Pending", "timestamp": None}
        for agent in AGENTS
    }
    with _status_lock:
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        LOG_FILE.parent.mkdir(exist_ok=True, parents=True)
        LOG_FILE.write_text("")

_status_lock = threading.Lock()
_last_update_time = 0
UPDATE_COOLDOWN = 1  # Minimum 1 second between updates

def update_status(agent: str, new_status: str):
    """Thread-safe status update with rate limiting"""
    global _last_update_time
    current_time = time.time()
    
    if current_time - _last_update_time < UPDATE_COOLDOWN:
        return
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with _status_lock:
        if not STATUS_FILE.exists():
            reset_status()
            
        data = json.loads(STATUS_FILE.read_text())
        if data.get(agent, {}).get("state") != new_status:
            data[agent] = {"state": new_status, "timestamp": timestamp}
            STATUS_FILE.write_text(json.dumps(data, indent=2))
            
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {agent}: {new_status}\n")
            _last_update_time = current_time

def get_status():
    """Return the current status of all agents."""
    with _status_lock:
        if not STATUS_FILE.exists():
            reset_status()
        return json.loads(STATUS_FILE.read_text())

def get_log():
    """Return the full execution log as text."""
    with _status_lock:
        if not LOG_FILE.exists():
            return "📭 No logs yet. Upload an RFP to begin tracking."
        return LOG_FILE.read_text().strip() or "📭 No logs yet. Upload an RFP to begin tracking."