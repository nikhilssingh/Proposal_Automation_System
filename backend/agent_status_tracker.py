# backend/agent_status_tracker.py

import json
import time
from pathlib import Path

STATUS_FILE = Path("logs/agent_status.json")
LOG_FILE = Path("logs/agent_log.txt")
STATUS_FILE.parent.mkdir(exist_ok=True)

AGENTS = [
    "RFP Analyzer",
    "Context Retriever",
    "Proposal Generator",
    "Strategy Optimizer",
    "Compliance Checker",
    "Scorer"
]

def reset_status():
    status = {
        agent: {"state": "⏳ Pending", "timestamp": None}
        for agent in AGENTS
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))

def update_status(agent: str, new_status: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if not STATUS_FILE.exists():
        reset_status()
    data = json.loads(STATUS_FILE.read_text())
    data[agent] = {"state": new_status, "timestamp": timestamp}
    STATUS_FILE.write_text(json.dumps(data, indent=2))
    with LOG_FILE.open("a") as f:
        f.write(f"[{timestamp}] {agent}: {new_status}\n")

def get_status():
    if not STATUS_FILE.exists():
        reset_status()
    return json.loads(STATUS_FILE.read_text())

def get_log():
    return LOG_FILE.read_text() if LOG_FILE.exists() else "No logs yet."
