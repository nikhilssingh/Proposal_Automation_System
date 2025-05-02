from datetime import datetime as dt
from pathlib import Path

def read_ts(path: Path):
    return dt.fromisoformat(path.read_text()) if path.exists() else None

start = read_ts(Path("logs/pipeline_start.txt"))
end   = read_ts(Path("logs/pipeline_end.txt"))

if start and end:
    print("Pipeline duration: %.1f minutes" % ((end-start).total_seconds()/60))
elif start:
    print("Pipeline still running…")
else:
    print("No run recorded yet.")
