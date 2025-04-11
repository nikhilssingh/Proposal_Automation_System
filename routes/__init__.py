# routes/__init__.py

from fastapi import APIRouter
from .rfp_routes import rfp_router
from .proposal_routes import proposal_router
from .retrieval_routes import retrieval_router

# Create a central router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(rfp_router, prefix="/rfp", tags=["RFP Management"])
api_router.include_router(proposal_router, prefix="/proposal", tags=["Proposal Generation"])
api_router.include_router(retrieval_router, prefix="/retrieval", tags=["Document Retrieval"])
