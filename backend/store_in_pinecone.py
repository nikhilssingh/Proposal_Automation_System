# backend/store_in_pinecone.py

import os
from typing import List
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore as LCPinecone
from langchain.schema import Document
from embeddings_setup import embeddings

# Check embedding dimension
test_text = "Test embedding"
test_vector = embeddings.embed_query(test_text)
dim = len(test_vector)
print(f"🔍 Debug: Generated embedding vector shape: {dim}")
if dim != 1536:
    raise ValueError(f"❌ Embedding mismatch! (Expected 1536, got {dim})")

# Pull API key and environment from .env
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")  # adjust to your region
index_name = "my-proposals-index"

# Create a Pinecone client instance
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Check if index exists, if not create it
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    print(f"Creating index '{index_name}' since it doesn't exist yet.")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp" depending on your setup
            region=pinecone_env
        )
    )

index = pc.Index(name=index_name)

# Optionally clear old data
try:
    index_stats = index.describe_index_stats()
    existing_ns = index_stats.get("namespaces", {})
    if existing_ns:
        for ns in existing_ns:
            index.delete(delete_all=True, namespace=ns)
            print(f"Deleted all vectors in namespace '{ns}'.")
    else:
        print("No namespaces found to delete.")
except Exception as e:
    print(f"Error clearing old data: {e}")

# Now read your local docs
docs_dir = "docs"
if not os.path.exists(docs_dir):
    raise FileNotFoundError(f"No '{docs_dir}' folder found. Please create it and add docs.")

doc_texts = []
for fname in os.listdir(docs_dir):
    path = os.path.join(docs_dir, fname)
    if not os.path.isfile(path):
        continue  # skip subdirectories

    if fname.lower().endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        doc_texts.append(text)
        print(f"Read text file '{fname}'.")
    elif fname.lower().endswith(".pdf"):
        extracted = ""
        with open(path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                extracted += page_text + "\n"
        if extracted:
            doc_texts.append(extracted)
            print(f"Extracted text from PDF '{fname}'.")
        else:
            print(f"No text found in PDF '{fname}'.")
    else:
        print(f"Skipping unsupported file type: {fname}")

# Split text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = []
def create_meaningful_chunks(text: str) -> List[str]:
    """Split text by meaningful sections from the template"""
    sections = []
    current_section = []
    
    for line in text.split('\n'):
        if line.startswith('# ') or line.startswith('## '):  # Heading detected
            if current_section:  # Save previous section
                sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(line)
    
    if current_section:  # Add last section
        sections.append('\n'.join(current_section))
    
    return sections

# Then modify the PDF processing:
for full_text in doc_texts:
    chunks = create_meaningful_chunks(full_text)  # Instead of using splitter
    for chunk in chunks:
        if len(chunk) > 100:  # Only store meaningful chunks
            docs.append(Document(
                page_content=chunk,
                metadata={"source": fname}
            ))

# Upload to Pinecone
vec_store = LCPinecone.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name
)
print("✅ Documents uploaded to Pinecone!")
