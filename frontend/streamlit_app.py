import streamlit as st
import requests
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# register a Unicode font once at import time
pdfmetrics.registerFont(TTFont("DejaVu", "fonts/DejaVuSans.ttf"))  # put the TTF here
# Allow “backend.” imports when running from /frontend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------- CONFIG ----------------------------
API_URL        = "http://127.0.0.1:8000"
POLL_INTERVAL  = 5          # seconds between polling backend
MAX_POLL_TIME  = 1800       # 30 min safety timeout
OUTPUT_DIR     = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------- SESSION HELPERS ------------------------
def init_session_state():
    defaults = {
        "current_proposal": "",
        "proposal_generated": False,
        "proposal_refined":  False,
        "compliance_report": "",
        "score_report":      "",
        "table_summaries":   [],
        "llm_calls":         0,
        "last_status":       {},
        "last_log":          "",
        "last_status_check": datetime.now() - timedelta(seconds=10),
        "last_log_check":    datetime.now() - timedelta(seconds=10),
        "file_uploaded":     False,
        "processing_started":False,
        "pipeline_id":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

PAGE_WIDTH, PAGE_HEIGHT = 595, 842   # A4 in points
LEFT, TOP, BOTTOM = 40, 800, 40      # margins (x, y_start, y_bottom)
LEADING = 14                         # line height (11-pt font + 3 pt)

def build_pdf(text: str) -> bytes:
    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    def new_text_object():
        t = c.beginText(LEFT, TOP)
        t.setFont("DejaVu", 11)
        return t

    text_obj = new_text_object()
    y_cursor = TOP

    for line in text.splitlines():
        # hard-wrap very long “words” so they break nicely
        for frag in _wrap_long_word(line, max_chunk=80):
            if y_cursor < BOTTOM + LEADING:          # need new page
                c.drawText(text_obj)
                c.showPage()
                text_obj = new_text_object()
                y_cursor = TOP
            text_obj.textLine(frag)
            y_cursor -= LEADING

    c.drawText(text_obj)
    c.showPage()
    c.save()
    return buf.getvalue()

def _wrap_long_word(line: str, max_chunk: int = 80):
    """
    Yield the line split so no single word exceeds max_chunk chars.
    """
    import textwrap, re
    def breaker(match):
        word = match.group(0)
        return "\n".join(textwrap.wrap(word, max_chunk))
    return re.sub(r"\S{" + str(max_chunk) + r",}", breaker, line).split("\n")


def reset_backend():
    try:
        requests.get(f"{API_URL}/rfp/reset_status", timeout=2)
    except Exception:
        pass

def get_cached_status():
    if (datetime.now() - st.session_state.last_status_check).seconds < POLL_INTERVAL:
        return st.session_state.last_status
    try:
        res = requests.get(f"{API_URL}/proposal/agent_status", timeout=2)
        if res.ok:
            st.session_state.last_status = res.json()
            st.session_state.last_status_check = datetime.now()
    except Exception:
        pass
    return st.session_state.last_status

def get_cached_log():
    if (datetime.now() - st.session_state.last_log_check).seconds < POLL_INTERVAL:
        return st.session_state.last_log
    try:
        res = requests.get(f"{API_URL}/proposal/agent_log", timeout=2)
        if res.ok:
            st.session_state.last_log = res.text
            st.session_state.last_log_check = datetime.now()
    except Exception:
        pass
    return st.session_state.last_log

# -------------------------- UI START ----------------------------
init_session_state()
st.title("📄 AI-Powered RFP Automation System")

# ----------- 1) Upload RFP ----------
if not st.session_state.file_uploaded:
    st.header("📂 Upload an RFP Document")
    up_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if up_file and st.button("Start Proposal Pipeline"):
        with st.spinner("📤 Uploading file…"):
            try:
                files = {"file": (up_file.name, up_file.getvalue())}
                res   = requests.post(f"{API_URL}/rfp/upload_rfp", files=files, timeout=10)
                if res.ok:
                    st.session_state.pipeline_id      = res.json().get("pipeline_id")
                    st.session_state.file_uploaded    = True
                    st.session_state.processing_started = True
                    st.success("✅ File uploaded. Processing…")
                else:
                    st.error("❌ File upload failed.")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

# ----------- 2) Poll until proposal is ready ----------
if st.session_state.processing_started and not st.session_state.proposal_generated:
    with st.spinner("🧠 Generating proposal…"):
        start = datetime.now()
        while (datetime.now() - start).seconds < MAX_POLL_TIME:
            try:
                res = requests.get(
                    f"{API_URL}/rfp/result/{st.session_state.pipeline_id}", timeout=5
                )
                if res.ok:
                    result = res.json()
                    if result.get("status") == "complete":
                        st.session_state.update(
                            current_proposal  = result.get("proposal", ""),
                            proposal_generated=True,
                            compliance_report = result.get("compliance_report", ""),
                            score_report      = result.get("score_report", ""),
                            table_summaries   = result.get("summarized_tables", []),
                            llm_calls         = result.get("llm_usage_count", 0),
                        )
                        st.rerun()
                        break
                    if result.get("status") == "failed":
                        st.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
                        st.session_state.processing_started = False
                        break
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                st.warning(f"⚠️ Temporary connection issue: {e}")
                time.sleep(POLL_INTERVAL)
        else:
            st.error("❌ Timed out waiting for proposal generation.")
            st.session_state.processing_started = False

# ----------- 3) Display proposal & refinement ----------
if st.session_state.proposal_generated:
    st.header("📄 Generated Proposal")
    st.write(st.session_state.current_proposal)

    # ---------- Download as PDF (ReportLab) ----------
    if st.button("⬇️ Download proposal as PDF"):
        pdf_bytes = build_pdf(st.session_state.current_proposal)
        st.download_button("Download PDF", pdf_bytes,
                           file_name="proposal.pdf",
                           mime="application/pdf")

    # ---------- Refine ----------
    st.header("🛠️ Refine Proposal")
    feedback = st.text_area("Your feedback", height=100)
    if st.button("Refine Proposal") and feedback:
        try:
            res=requests.post(
                f"{API_URL}/proposal/refine_proposal",
                json={
                    "current_proposal": st.session_state.current_proposal,
                    "user_feedback":    feedback
                },
                timeout=60
            )
            if res.ok:
                data=res.json()
                st.session_state.update(
                    current_proposal =data["refined_proposal"],
                    proposal_refined =True,
                    compliance_report=data.get("compliance_report",""),
                    score_report     =data.get("score_report","")
                )
                st.success("✅ Refined!")
                st.rerun()
            else:
                st.error(res.text)
        except Exception as e:
            st.error(f"❌ {e}")

# ----------- 4) Compliance & scoring ----------
if st.session_state.compliance_report:
    st.subheader("📋 Compliance Check")
    st.write(st.session_state.compliance_report)

if st.session_state.score_report:
    st.subheader("📈 Proposal Scorecard")
    st.write(st.session_state.score_report)

# ----------- 5) Agent dashboard ----------
st.markdown("---")
st.subheader("🧠 Agent-Status Dashboard")
def fmt(s):  # color helper
    return (
        f":green[{s}]" if "✅" in s else
        f":red[{s}]"  if "❌" in s else
        f":blue[{s}]" if "🧠" in s else
        f":gray[{s}]"
    )
for agent, info in get_cached_status().items():
    st.markdown(f"**{agent}**: {fmt(info['state'])} _(at {info['timestamp']})_")

st.subheader("📜 Execution Log")
st.text_area("Log", get_cached_log(), height=200, key="agent_log_display")

# ----------- 6) Reset ----------
if st.button("♻️ Reset Session"):
    reset_backend()
    st.session_state.clear()
    st.rerun()
