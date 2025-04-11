import streamlit as st
import requests
from fpdf import FPDF
import time
import os

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.title("📄 AI-Powered RFP Automation System")

# Initialize Session State Variables
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = ""
if "proposal_generated" not in st.session_state:
    st.session_state.proposal_generated = False
if "proposal_refined" not in st.session_state:
    st.session_state.proposal_refined = False

# For compliance & score reports
if "compliance_report" not in st.session_state:
    st.session_state.compliance_report = ""
if "score_report" not in st.session_state:
    st.session_state.score_report = ""

# --- Step 1: Upload and Generate Proposal (only once) ---
st.header("📂 Upload an RFP Document")
if not st.session_state.proposal_generated:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/rfp/upload_rfp", files=files)
        if response.status_code == 200:
            result = response.json()
            extracted_rfp_text = result["extracted_text"]
            st.success(f"✅ File Uploaded: {result['filename']}")
            st.write("📜 **Extracted Text Preview:**", extracted_rfp_text[:500])

            # Generate the initial proposal
            proposal_response = requests.post(
                f"{API_URL}/proposal/generate_proposal",
                json={
                    "rfp_text": extracted_rfp_text,
                    "retrieved_docs": []
                }
            )
            if proposal_response.status_code == 200:
                gen_result = proposal_response.json()

                # Store text-based fields
                st.session_state.current_proposal = gen_result.get("proposal", "")
                st.session_state.proposal_generated = True

                # OPTIONAL: If your backend returns these, store them
                st.session_state.compliance_report = gen_result.get("compliance_report", "")
                st.session_state.score_report = gen_result.get("score_report", "")

                st.success("✅ Proposal Generated!")
                st.write("📌 **Generated Proposal:**", st.session_state.current_proposal)

                # Store the proposal in the backend
                requests.post(
                    f"{API_URL}/proposal/store_proposal",
                    json={"proposal": st.session_state.current_proposal}
                )
            else:
                st.error(f"❌ Error generating proposal: {proposal_response.text}")
        else:
            st.error("❌ File upload failed.")
else:
    st.write("Using previously generated proposal:")
    st.write(st.session_state.current_proposal)

# --- Step 2: Refine the Proposal ---
st.header("🛠️ Refine Proposal")
user_feedback = st.text_area("Your Feedback", height=100)

if st.button("Refine Proposal"):
    if user_feedback:
        refine_response = requests.post(
            f"{API_URL}/proposal/refine_proposal",
            json={"user_feedback": user_feedback}
        )
        if refine_response.status_code == 200:
            ref_result = refine_response.json()
            refined_proposal = ref_result.get("refined_proposal", "")

            if refined_proposal:
                st.session_state.current_proposal = refined_proposal
                st.session_state.proposal_refined = True

                # OPTIONAL: If your backend also returns compliance/score here:
                st.session_state.compliance_report = ref_result.get("compliance_report", "")
                st.session_state.score_report = ref_result.get("score_report", "")

                st.success("✅ Proposal Refined!")
                st.write("📌 **Refined Proposal:**", refined_proposal)

                # Update the backend with the refined proposal
                requests.post(
                    f"{API_URL}/proposal/store_proposal",
                    json={"proposal": refined_proposal}
                )
            else:
                st.warning("⚠️ No changes were made.")
        else:
            st.error(f"❌ Error refining proposal: {refine_response.text}")

# --- Step 3: Export the Latest Proposal as PDF ---
st.header("📤 Finalize & Export")
if st.button("Submit and Export as PDF"):
    # Always fetch the latest proposal from the backend (with cache busting)
    response = requests.get(f"{API_URL}/proposal/get_latest_proposal?timestamp={time.time()}")
    if response.status_code == 200:
        final_proposal_text = response.json().get("proposal", "")
        # Debug: display the proposal fetched from the backend
        st.write("DEBUG: Backend returned proposal:", final_proposal_text)
    else:
        st.error("Failed to fetch the latest proposal.")
        final_proposal_text = ""

    if not final_proposal_text.strip():
        st.error("❌ No final proposal found. Please refine or generate the proposal first.")
    else:
        # Update session state with the fetched proposal
        st.session_state.current_proposal = final_proposal_text

        # Create the PDF using the latest refined proposal
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        current_dir = os.path.dirname(os.path.abspath(__file__))

        font_path = os.path.join(current_dir, "fonts", "DejaVuSans.ttf")

        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 11)
    
        pdf.set_left_margin(10)
        pdf.set_right_margin(10)
        formatted_text = "\n".join(
            line.strip() for line in final_proposal_text.split("\n") if line.strip()
        )
        pdf.multi_cell(0, 8, txt=formatted_text, border=0)
        pdf_output = bytes(pdf.output(dest="S"))

        st.download_button(
            label="📥 Download Proposal PDF",
            data=pdf_output,
            file_name=f"final_proposal_{int(time.time())}.pdf",  # unique filename to avoid caching issues
            mime="application/pdf"
        )

# --- Compliance & Score (Optional) ---
# Display them if we have them in session state:
if st.session_state.compliance_report:
    st.subheader("📋 Compliance Check")
    st.write(st.session_state.compliance_report)

if st.session_state.score_report:
    st.subheader("📊 Proposal Scorecard")
    st.write(st.session_state.score_report)

st.markdown("---")
st.subheader("🤖 Agent Status Dashboard + Audit Log")

# Toggle to activate real-time polling
live_mode = st.checkbox("🔄 Enable Live Polling (30s)", value=False)

agent_keys = [
    "RFP Analyzer",
    "Context Retriever",
    "Table Summarizer",
    "Proposal Generator",
    "Strategy Optimizer",
    "Compliance Checker",
    "Scorer"
]

def fetch_agent_status():
    try:
        res = requests.get(f"{API_URL}/proposal/agent_status")
        if res.status_code == 200:
            return res.json()
    except:
        return {k: {"state": "❌ Connection error", "timestamp": "—"} for k in agent_keys}


def fetch_agent_log():
    try:
        res = requests.get(f"{API_URL}/proposal/agent_log")
        return res.text if res.status_code == 200 else "Log unavailable."
    except:
        return "Error retrieving logs."

def format_status(status):
    if "✅" in status:
        return f":green[{status}]"
    elif "❌" in status:
        return f":red[{status}]"
    elif "🧠" in status:
        return f":blue[{status}]"
    else:
        return f":gray[{status}]"

status_placeholder = st.empty()
log_placeholder = st.empty()

if live_mode:
    for _ in range(10):  # ~30 seconds of polling
        agent_status = fetch_agent_status()
        agent_log = fetch_agent_log()

        with status_placeholder.container():
            st.subheader("🧠 Current Agent Status")
            for agent, obj in agent_status.items():
                st.markdown(f"**{agent}**: {format_status(obj['state'])} _(at {obj['timestamp']})_")

        with log_placeholder.container():
            st.subheader("📜 Agent Execution Log")
            st.text_area("Audit Log", agent_log, height=250)

        time.sleep(3)
        st.experimental_rerun()
else:
    agent_status = fetch_agent_status()
    agent_log = fetch_agent_log()

    with status_placeholder.container():
        st.subheader("🧠 Current Agent Status")
        for agent, obj in agent_status.items():
            st.markdown(f"**{agent}**: {format_status(obj['state'])} _(at {obj['timestamp']})_")

    with log_placeholder.container():
        st.subheader("📜 Agent Execution Log")
        st.text_area("Audit Log", agent_log, height=250)

st.caption(f"⏱️ Last updated: {time.strftime('%H:%M:%S')}")

# Reset session button
if st.button("♻️ Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

