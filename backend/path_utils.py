from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # one level above /backend
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
