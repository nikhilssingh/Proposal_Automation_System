# backend/pinecone_utils.py

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from backend.embeddings_setup import embeddings
from dotenv import load_dotenv
load_dotenv()  
import os
import logging

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)

# ✅ Load Pinecone API settings
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
index_name = "my-proposals-index"


# Create Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Get the index from the client
index = pc.Index(name=index_name)

# ✅ Function to retrieve similar documents
def retrieve_similar_docs(query: str, top_k: int = 3):
    """ Retrieves relevant RFP documents from Pinecone using similarity search. """
    try:
        # ✅ Print embedding shape before querying Pinecone
        vector = embeddings.embed_query(query)
        print(f"🔍 Generated embedding vector shape: {len(vector)}")  # Should be 1536
        # Initialize PineconeVectorStore
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

        # Perform similarity search
        docs = vector_store.similarity_search(query, k=top_k)

        if docs:
            retrieved_texts = [doc.page_content for doc in docs]
            logging.info(f"✅ Retrieved Documents:\n{retrieved_texts}")
            return retrieved_texts
        else:
            logging.warning("⚠️ No similar documents found.")
            return ["No similar documents found."]
    except Exception as e:
        logging.error(f"❌ Error retrieving documents: {str(e)}")
        return [f"Error retrieving documents: {str(e)}"]
    
    
    