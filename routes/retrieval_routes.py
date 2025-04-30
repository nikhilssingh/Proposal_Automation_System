# routes/retrieval_routes.py

from fastapi import APIRouter, HTTPException, Query
from backend.pinecone_utils import retrieve_similar_docs

retrieval_router = APIRouter()

@retrieval_router.get("/retrieve_docs")
def retrieve_documents(query: str = Query(..., description="Search query for document retrieval")):
    """
    Retrieve relevant documents from Pinecone based on a query string.
    """
    try:
        print(f"🔍 Debug: Received Query - {query}")
        retrieved_docs = retrieve_similar_docs(query)
        print(f"📄 Debug: Retrieved Docs - {retrieved_docs}")

        if not retrieved_docs:
            retrieved_docs = ["No similar documents found."]

        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        print(f"❌ Debug: Retrieval Error - {str(e)}")
        return {"retrieved_docs": [f"Error retrieving documents: {str(e)}"]}
