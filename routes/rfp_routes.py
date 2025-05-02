from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.agentic_pipeline import proposal_agentic_graph
from backend.llm_utils import llm_usage_count
from backend.agent_status_tracker import reset_status, update_status
import os
import threading
import json
import uuid
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Global pipeline tracker
active_pipelines = {}

rfp_router = APIRouter()
UPLOAD_DIR = "uploaded_rfps"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track active pipelines
active_pipelines = {}
pipeline_lock = threading.Lock()

@rfp_router.get("/reset_status")
def reset_status_endpoint():
    reset_status()
    with pipeline_lock:
        active_pipelines.clear()
    return {"status": "reset done"}

from backend.agent_status_tracker import mark_pipeline_start


@rfp_router.post("/upload_rfp")
async def upload_rfp(file: UploadFile = File(...)):
    try:
        reset_status()
        mark_pipeline_start()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        from backend import llm_utils
        llm_utils.llm_usage_count = 0

        pipeline_id = str(uuid4())

        # ✅ Initialize status before thread starts
        active_pipelines[pipeline_id] = {
            "status": "processing",
            "result": None
        }

        def run_pipeline():
            try:
                output = proposal_agentic_graph.invoke({"rfp_path": file_path})
                output["llm_usage_count"] = llm_utils.llm_usage_count
                active_pipelines[pipeline_id]["status"] = "complete"
                active_pipelines[pipeline_id]["result"] = output
            except Exception as e:
                active_pipelines[pipeline_id]["status"] = "failed"
                active_pipelines[pipeline_id]["result"] = {"error": str(e)}

        threading.Thread(target=run_pipeline, daemon=True).start()

        return {"message": "Pipeline started", "pipeline_id": pipeline_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rfp_router.get("/result/{pipeline_id}")
def get_result(pipeline_id: str):
    try:
        if pipeline_id not in active_pipelines:
            return {"status": "not_found"}

        data = active_pipelines[pipeline_id]
        return {
            "status": data["status"],
            **(data["result"] or {})
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
