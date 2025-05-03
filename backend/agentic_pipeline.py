# backend/agentic_pipeline.py

from langgraph.graph import StateGraph, END
from typing import TypedDict

from backend.llm_utils import (
    llm_usage_count,
    expand_rfp,
    optimize_proposal_tone,
    check_compliance,
    score_proposal_quality,
    extract_rfp_metadata,
    summarize_table,
)
from backend.pinecone_utils import retrieve_similar_docs, upsert_proposal
from backend.agent_status_tracker import update_status, mark_pipeline_end

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
    proposal_indexed: bool

def extract_pdf_node(state: ProposalState) -> ProposalState:
    from backend.parse_rfp_pdf import parse_rfp_pdf
    update_status("RFP Analyzer", "📄 Extracting PDF contents", force=True)
    parsed = parse_rfp_pdf(state["rfp_path"])

    # If no tables, skip table summarizer
    if not parsed["tables"]:
        update_status("Table Summarizer", "✅ Skipped (no tables)", force=True)

    new_state = {
        **state,
        "rfp_text":   parsed["text_body"] + "\n\n" + parsed["ocr_text"],
        "raw_tables": parsed["tables"],
        "ocr_text":   parsed["ocr_text"],
        "optimize_attempts": 0,
        "compliance_retries": 0,
    }
    update_status("RFP Analyzer", "✅ Done", force=True)
    return new_state

def enrich_rfp_node(state: ProposalState) -> ProposalState:
    update_status("RFP Analyzer", "🧠 In Progress")
    rfp_text = state.get("rfp_text", "").strip()
    metadata = extract_rfp_metadata(rfp_text)
    update_status("RFP Analyzer", "✅ Done")
    return {
        **state,
        "rfp_text":    rfp_text,
        "metadata":    metadata,
        "industry":    metadata.get("industry", "generic"),
        "region":      metadata.get("region", "global"),
        "constraints": metadata.get("constraints", []),
        "client_needs":metadata.get("client_needs", []),
    }

def retrieve_docs_node(state: ProposalState) -> ProposalState:
    update_status("Context Retriever", "🧠 In Progress")
    docs = retrieve_similar_docs(state["rfp_text"], top_k=3)
    update_status("Context Retriever", "✅ Done")
    return {**state, "retrieved_docs": docs}

def table_summary_node(state: ProposalState) -> ProposalState:
    update_status("Table Summarizer", "🧠 In Progress", force=True)
    if not state.get("raw_tables"):
        update_status("Table Summarizer", "✅ Done (no tables)", force=True)
        return state

    summarized = []
    for table_data in state["raw_tables"]:
        markdown = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table_data if row
        )
        summary = summarize_table(markdown)
        summarized.append(f"📊 Table\n{markdown}\n\n📝 Summary: {summary}")

    state["summarized_tables"] = summarized
    update_status("Table Summarizer", "✅ Done", force=True)
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
        "proposal":        optimized,
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
    if state.get("compliance_passed"):
        return "Score Proposal"
    if state.get("optimize_attempts", 0) >= 2:
        return "Score Proposal"
    return "Optimize Tone"

def score_proposal_node(state: ProposalState) -> ProposalState:
    update_status("Scorer", "🧠 In Progress")
    score = score_proposal_quality(state["proposal"])

    # ← only upsert once per run
    if not state.get("proposal_indexed", False):
        from backend.pinecone_utils import upsert_proposal
        upsert_proposal(state["proposal"])
        state["proposal_indexed"] = True

    mark_pipeline_end()
    update_status("Scorer", "✅ Done")

    return {
        **state,
        "score_report": score,
        "llm_usage_count": llm_usage_count
    }

# Build & compile the LangGraph
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
    {"Score Proposal": "Score Proposal", "Optimize Tone": "Optimize Tone"}
)
builder.add_edge("Score Proposal", END)

proposal_agentic_graph = builder.compile()
