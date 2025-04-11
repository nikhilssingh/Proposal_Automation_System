# backend/models.py

from pydantic import BaseModel
from typing import List, Optional

class RFPRequest(BaseModel):
    """Model for processing RFP requests."""
    rfp_text: str

class RFPResponse(BaseModel):
    """Model for storing RFP responses."""
    filename: str
    extracted_text: str
    complexity_score: int

class ProposalRequest(BaseModel):
    """Model for proposal generation."""
    rfp_text: str

class ProposalResponse(BaseModel):
    """Model for storing generated proposals."""
    summary: str
    proposal: str

class RetrievalRequest(BaseModel):
    """Model for document retrieval requests."""
    query: str

class RetrievalResponse(BaseModel):
    """Model for retrieved documents."""
    retrieved_docs: List[str]
