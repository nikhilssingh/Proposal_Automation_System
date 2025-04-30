# backend/pinecone_utils.py

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from backend.embeddings_setup import embeddings

# 1) Load credentials
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")  # Adjust to your region
index_name = "my-proposals-index"

# 2) Create a Pinecone client instance
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# 3) Validate index existence
existing_indexes = pc.list_indexes().names()  # returns an IndexNameList
if index_name not in existing_indexes:
    raise ValueError(f"Index '{index_name}' not found in Pinecone.\n"
                     f"Existing indexes: {existing_indexes}")

# 4) Initialize a reference to your index
index = pc.Index(name=index_name)

def retrieve_similar_docs(query: str, top_k: int = 3):
    """
    Retrieves relevant documents from Pinecone using similarity search.
    Returns a list of doc page_content strings.
    """
    try:
        vector = embeddings.embed_query(query)
        logging.info(f"🔍 Generated embedding vector shape: {len(vector)}")

        # Use the LangChain VectorStore wrapper
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

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
