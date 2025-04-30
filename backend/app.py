# app.py

from fastapi import FastAPI
from routes import api_router  # ✅ Central route import
from backend.agentic_pipeline import proposal_agentic_graph  # ✅ Import graph here
from backend.llm_utils import llm_usage_count
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(title="RFP Automation API", version="1.0")

# Register routes
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Welcome to the RFP Automation API"}

# (Optional) Internal function to trigger pipeline programmatically
def invoke_proposal_pipeline(payload):
    final_state = proposal_agentic_graph.invoke(payload)
    final_state["llm_usage_count"] = llm_usage_count
    return final_state

# Run locally if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
