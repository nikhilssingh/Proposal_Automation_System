import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_and_tables(pdf_path: str):
    text_body = ""
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract regular text
            text = page.extract_text()
            if text:
                text_body += text + "\n"

            # Extract tables (as 2D arrays)
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)

    return text_body.strip(), all_tables

def extract_ocr_text_from_images(pdf_path: str):
    images = convert_from_path(pdf_path)
    ocr_text = ""

    for img in images:
        gray = img.convert("L")
        ocr_text += pytesseract.image_to_string(gray) + "\n"
        img = img.convert("RGB")
        ocr_text += pytesseract.image_to_string(img) + "\n"



    return ocr_text.strip()

def parse_rfp_pdf(pdf_path: str):
    text_body, tables = extract_text_and_tables(pdf_path)
    ocr_text = extract_ocr_text_from_images(pdf_path)

    return {
        "text_body": text_body,
        "tables": tables,
        "ocr_text": ocr_text
    }
