# backend/agentic_pipeline.py (LangGraph-based)

from langgraph.graph import StateGraph, END
from typing import TypedDict
from backend.llm_utils import llm_usage_count

# Pinecone-based doc retrieval
from backend.pinecone_utils import retrieve_similar_docs

# LLM utility functions
from backend.llm_utils import (
    expand_rfp,
    optimize_proposal_tone,
    check_compliance,
    score_proposal_quality,
    extract_rfp_metadata,
    summarize_table,
    summarize_text
)

# Status tracker
from backend.agent_status_tracker import update_status

class ProposalState(TypedDict, total=False):
    rfp_path: str
    rfp_text: str
    metadata: dict
    industry: str
    region: str
    constraints: list
    client_needs: list
    retrieved_docs: list
    summarized_tables: list
    proposal: str
    compliance_report: str
    compliance_passed: bool
    score_report: str
    raw_tables: list
    ocr_text: str
    compliance_retries: int
    optimize_attempts: int
    llm_usage_count: int

# ------------------ Node definitions ------------------

def extract_pdf_node(state: ProposalState) -> ProposalState:
    from backend.parse_rfp_pdf import parse_rfp_pdf
    update_status("RFP Analyzer", "📄 Extracting PDF contents")
    parsed = parse_rfp_pdf(state["rfp_path"])
    initial_state = {
        **state,
        "rfp_text": parsed["text_body"] + "\n\n" + parsed["ocr_text"],
        "raw_tables": parsed["tables"],
        "ocr_text": parsed["ocr_text"],
        "optimize_attempts": 0,
        "compliance_retries": 0
    }
    return initial_state

def enrich_rfp_node(state: ProposalState) -> ProposalState:
    update_status("RFP Analyzer", "🧠 In Progress")
    rfp_text = state.get("rfp_text", "").strip()
    metadata = extract_rfp_metadata(rfp_text)
    update_status("RFP Analyzer", "✅ Done")
    return {
        **state,
        "rfp_text": rfp_text,
        "metadata": metadata,
        "industry": metadata.get("industry", "generic"),
        "region": metadata.get("region", "global"),
        "constraints": metadata.get("constraints", []),
        "client_needs": metadata.get("client_needs", [])
    }

def retrieve_docs_node(state):
    if "retrieved_docs" in state and state["retrieved_docs"]:
        print("⚠️ Retrieval already done. Skipping redundant embedding/retrieval.")
        return state

    update_status("Context Retriever", "🧠 In Progress")
    rfp_text = state["rfp_text"]
    if len(rfp_text) > 3000:
        context = f"{rfp_text[:3000]}\n\n...\n\n{rfp_text[-1000:]}"
        try:
            docs = retrieve_similar_docs(context, top_k=3)
        except Exception as e:
            print(f"⚠️ Retrieval failed: {e}")
            docs = []
    else:
        try:
            docs = retrieve_similar_docs(rfp_text, top_k=3)
        except Exception as e:
            print(f"⚠️ Retrieval failed: {e}")
            docs = []

    update_status("Context Retriever", "✅ Done")
    return {**state, "retrieved_docs": docs[:3]}

def table_summary_node(state: ProposalState) -> ProposalState:
    update_status("Table Summarizer", "🧠 In Progress")
    summarized_tables = []
    for table_data in state.get("raw_tables", []):
        markdown_table = "\n".join([
            " | ".join(str(cell or "") for cell in row) for row in table_data if row
        ])
        summary = summarize_table(markdown_table)
        summarized_tables.append(
            f"📊 Table\n{markdown_table}\n\n📝 Summary: {summary}"
        )

    state["summarized_tables"] = summarized_tables
    update_status("Table Summarizer", "✅ Done")
    return state

def generate_proposal_node(state: ProposalState) -> ProposalState:
    update_status("Proposal Generator", "🧠 In Progress")
    proposal = expand_rfp(
        state["rfp_text"],
        state["retrieved_docs"],
        summarized_tables=state.get("summarized_tables", [])
    )
    update_status("Proposal Generator", "✅ Done")
    return {**state, "proposal": proposal}

def optimize_proposal_node(state: ProposalState) -> ProposalState:
    update_status("Strategy Optimizer", "🧠 In Progress")
    optimized = optimize_proposal_tone(
        state["proposal"],
        vertical=state.get("industry", "generic"),
        tone="persuasive"
    )
    update_status("Strategy Optimizer", "✅ Done")
    return {
        **state,
        "proposal": optimized,
        "optimize_attempts": state.get("optimize_attempts", 0) + 1
    }

def check_compliance_node(state: ProposalState) -> ProposalState:
    update_status("Compliance Checker", "🧠 In Progress")
    report = check_compliance(state["rfp_text"], state["proposal"])
    state["compliance_report"] = report
    state["compliance_passed"] = ("❌" not in report)
    update_status("Compliance Checker", "✅ Done")
    return state

def compliance_condition(state: ProposalState) -> str:
    max_attempts = 2
    if state.get("compliance_passed"):
        return "Score Proposal"
    elif state.get("optimize_attempts", 0) >= max_attempts:
        print("⚠️ Max optimization attempts reached. Proceeding anyway.")
        return "Score Proposal"
    else:
        return "Optimize Tone"

def score_proposal_node(state: ProposalState) -> ProposalState:
    update_status("Scorer", "🧠 In Progress")
    score_report = score_proposal_quality(state["proposal"])
    update_status("Scorer", "✅ Done")
    return {
        **state,
        "score_report": score_report,
        "llm_usage_count": llm_usage_count
    }

# ------------------ Building the LangGraph ------------------

builder = StateGraph(state_schema=ProposalState)

builder.add_node("Extract PDF", extract_pdf_node)
builder.add_node("Enrich RFP", enrich_rfp_node)
builder.add_node("Retrieve Docs", retrieve_docs_node)
builder.add_node("Summarize Tables", table_summary_node)
builder.add_node("Generate Proposal", generate_proposal_node)
builder.add_node("Optimize Tone", optimize_proposal_node)
builder.add_node("Check Compliance", check_compliance_node)
builder.add_node("Score Proposal", score_proposal_node)

builder.set_entry_point("Extract PDF")
builder.add_edge("Extract PDF", "Enrich RFP")
builder.add_edge("Enrich RFP", "Retrieve Docs")
builder.add_edge("Retrieve Docs", "Summarize Tables")
builder.add_edge("Summarize Tables", "Generate Proposal")
builder.add_edge("Generate Proposal", "Optimize Tone")
builder.add_edge("Optimize Tone", "Check Compliance")

builder.add_conditional_edges(
    "Check Compliance",
    compliance_condition,
    {
        "Score Proposal": "Score Proposal",
        "Optimize Tone": "Optimize Tone"
    }
)
builder.add_edge("Score Proposal", END)

proposal_agentic_graph = builder.compile()