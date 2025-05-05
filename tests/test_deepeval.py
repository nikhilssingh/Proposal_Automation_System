# tests/test_deepeval.py

from datetime import time
from pathlib import Path
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.evaluate import assert_test


def test_proposal_quality():
    # pick the first RFP in uploaded_rfps/ as our sample
    upload_dir = Path("uploaded_rfps")
    files = list(upload_dir.iterdir())
    if not files:
        raise FileNotFoundError("No files found in uploaded_rfps/ – please upload at least one RFP.")

    sample = files[0]
    # if it's a PDF, extract text via PyPDF2
    if sample.suffix.lower() == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("Please install PyPDF2 to extract text from PDF test fixtures.")
        reader = PdfReader(str(sample))
        rfp_text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    else:
        rfp_text = sample.read_text(encoding="utf-8")

    # call your generate_proposal endpoint
    resp = requests.post(
        "http://localhost:8000/proposal/generate_proposal",
        json={"rfp_text": rfp_text,
              "rfp_path": str(sample.resolve())
              },
        timeout=360
    )
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    proposal = resp.json().get("proposal", "")

    test_case = LLMTestCase(input=rfp_text, actual_output=proposal)

    correctness = GEval(
        name="Correctness",
        criteria="Does the proposal fully address the client's requirements?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        threshold=0.6
    )
    persuasiveness = GEval(
        name="Persuasiveness",
        criteria="Is the tone of the proposal persuasive and business‑appropriate?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6
    )

    assert_test(test_case, [correctness, persuasiveness])
