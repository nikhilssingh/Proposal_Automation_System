# backend/embeddings_setup.py

from langchain_openai import OpenAIEmbeddings

# ✅ Ensure this matches Pinecone's 1536 dimensions
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
