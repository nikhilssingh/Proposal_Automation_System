# routes/retrieval_routes.py

from fastapi import APIRouter, HTTPException, Query
from backend.pinecone_utils import retrieve_similar_docs

retrieval_router = APIRouter()

@retrieval_router.get("/retrieve_docs")
def retrieve_documents(
    query: str = Query(..., description="Search query for document retrieval")
):
    """
    Retrieve the single most relevant document from Pinecone based on a query string.
    """
    try:
        # pull just top-1
        retrieved_docs = retrieve_similar_docs(query, top_k=1)
        if not retrieved_docs:
            retrieved_docs = ["No similar documents found."]
        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")
