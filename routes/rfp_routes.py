# routes/rfp_routes.py

from fastapi import APIRouter, UploadFile, File
from backend.parse_rfp_pdf import parse_rfp_pdf
import os

rfp_router = APIRouter()

UPLOAD_DIR = "uploaded_rfps"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@rfp_router.post("/upload_rfp")
async def upload_rfp(file: UploadFile = File(...)): 
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = parse_rfp_pdf(file_path)

    return {"filename": file.filename, "extracted_text": extracted_text[:500]}
