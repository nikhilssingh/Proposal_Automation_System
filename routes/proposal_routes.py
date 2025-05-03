# routes/proposal_routes.py

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.llm_utils import (
    expand_rfp,
    refine_proposal,
    conversation_memory,
    remove_unsupported_unicode,
    check_compliance,
    score_proposal_quality,
)
from backend.pinecone_utils import retrieve_similar_docs
from backend.agent_status_tracker import get_status, get_log, update_status

import os

proposal_router = APIRouter()

class RFPRequest(BaseModel):
    rfp_text: str
    retrieved_docs: list = []

@proposal_router.post("/generate_proposal")
def generate_proposal(request: RFPRequest):
    """
    Generate a proposal in response to an RFP while leveraging
    a pipeline (LangGraph).
    """
    from backend.agentic_pipeline import proposal_agentic_graph
    try:
        rfp_text = request.rfp_text.strip()
        if not rfp_text:
            raise HTTPException(status_code=400, detail="RFP text cannot be empty.")

        # Kick off pipeline
        result = proposal_agentic_graph.invoke({"rfp_text": rfp_text})

        proposal = remove_unsupported_unicode(result["proposal"])
        retrieved_docs = result["retrieved_docs"]
        compliance = remove_unsupported_unicode(result["compliance_report"])
        score = remove_unsupported_unicode(result["score_report"])

        # Store in conversation memory
        conversation_memory["latest_proposal"] = proposal

        return {
            "proposal": proposal,
            "retrieved_docs": retrieved_docs,
            "compliance_report": compliance,
            "score_report": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

class RefineRequest(BaseModel):
    current_proposal: str = None
    user_feedback: str

@proposal_router.post("/refine_proposal")
def refine_proposal_endpoint(refine_data: RefineRequest):
    try:
        update_status("Refiner", "🧠 In Progress")

        feedback = refine_data.user_feedback
        if not feedback:
            raise HTTPException(status_code=400, detail="User feedback is required.")

        current_proposal = refine_data.current_proposal or conversation_memory.get("latest_proposal", "")
        if not current_proposal:
            raise HTTPException(status_code=400, detail="No existing proposal to refine.")

        refined_result = refine_proposal(current_proposal, feedback)

        refined_proposal = remove_unsupported_unicode(refined_result["refined_proposal"])
        compliance = refined_result.get("compliance_report", "")
        score = refined_result.get("score_report", "")

        conversation_memory["latest_proposal"] = refined_proposal

        update_status("Refiner", "✅ Done")

        return {
            "refined_proposal": refined_proposal,
            "compliance_report": compliance,
            "score_report": score
        }

    except Exception as e:
        logging.exception("❌ Refinement failed")
        update_status("Refiner", "❌ Failed")
        return JSONResponse(status_code=500, content={"error": f"Refinement failed: {str(e)}"})

@proposal_router.get("/get_latest_proposal")
def get_latest_proposal():
    """
    Retrieve the latest refined proposal from memory.
    """
    latest_proposal = conversation_memory.get("latest_proposal", "")
    if not latest_proposal:
        return {"proposal": "No proposal found. Please generate or refine first."}
    return {"proposal": latest_proposal}

class StoreProposalRequest(BaseModel):
    proposal: str

@proposal_router.post("/store_proposal")
def store_proposal_endpoint(proposal: StoreProposalRequest):
    """
    Store a newly-provided proposal in memory.
    """
    cleaned_proposal = remove_unsupported_unicode(proposal.proposal)
    conversation_memory["latest_proposal"] = cleaned_proposal
    return {"message": "Proposal stored successfully."}

@proposal_router.get("/agent_status")
def agent_status():
    return get_status()

@proposal_router.get("/agent_log")
def agent_log():
    return get_log()

