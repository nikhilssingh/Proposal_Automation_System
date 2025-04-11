#parse_rfp_pdf.py

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# If Tesseract-OCR is not installed at default location, set path manually:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows Example

def parse_rfp_pdf(pdf_path: str) -> str:
    """
    Extracts text, tables, and image metadata from a PDF file.
    Uses pdfplumber for structured content and OCR as fallback.
    """
    extracted_text = ""
    tables = []
    image_descriptions = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # --- Extract normal text
                page_text = page.extract_text()
                if page_text:
                    extracted_text += f"\n[Text from Page {i+1}]\n{page_text}\n"

                # --- Extract tables
                extracted_tables = page.extract_tables()
                for tbl in extracted_tables:
                    if tbl:
                        table_str = "\n".join([" | ".join(row) for row in tbl if any(row)])
                        tables.append(f"\n📊 Table from Page {i+1}:\n{table_str}\n")

                # --- OCR on image of the page
                image = page.to_image(resolution=300)
                img = image.original
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    extracted_text += f"\n[OCR from Page {i+1} Image]\n{ocr_text.strip()}\n"

                # --- Log presence of embedded images
                for img in page.images:
                    image_descriptions.append(
                        f"📷 Image found on Page {i+1} (Position: x={img['x0']}, y={img['top']})"
                    )

        full_text = "\n".join([
            extracted_text.strip(),
            "\n".join(tables),
            "\n".join(image_descriptions)
        ])

        logging.info(f"✅ Extracted text, tables, and image metadata from {pdf_path}")
        return full_text.strip()

    except Exception as e:
        logging.error(f"❌ Failed to parse PDF: {pdf_path} – {e}")
        return "Error: Unable to process the PDF."

