# backend/agentic_pipeline.py (LangGraph-based)

from langgraph.graph import StateGraph, END
from backend.llm_utils import (
    expand_rfp,
    optimize_proposal_tone,
    check_compliance,
    score_proposal_quality
)
from backend.pinecone_utils import retrieve_similar_docs

from backend.llm_utils import (
    expand_rfp,
    optimize_proposal_tone,
    check_compliance,
    score_proposal_quality,
    extract_rfp_metadata,
    summarize_table  # ✅ add this
)

from backend.agent_status_tracker import update_status

# Define shared state type (dict-style)
from backend.llm_utils import extract_rfp_metadata

def enrich_rfp_node(state):
    update_status("RFP Analyzer", "🧠 In Progress")
    rfp_text = state.get("rfp_text", "").strip()

    metadata = extract_rfp_metadata(rfp_text)
    update_status("RFP Analyzer", "✅ Done")
    return {
        "rfp_text": rfp_text,
        "metadata": metadata,  # structured metadata dictionary
        "industry": metadata.get("industry", "generic"),
        "region": metadata.get("region", "global"),
        "constraints": metadata.get("constraints", []),
        "client_needs": metadata.get("client_needs", [])
    }

def retrieve_docs_node(state):
    update_status("Context Retriever", "🧠 In Progress")
    rfp_text = state["rfp_text"]
    retrieved_docs = retrieve_similar_docs(rfp_text)
    update_status("Context Retriever", "✅ Done")
    return {**state, "retrieved_docs": retrieved_docs}


def table_summary_node(state):
    update_status("Table Summarizer", "🧠 In Progress")
    summarized_tables = []
    for text_block in state.get("retrieved_docs", []):
        if "📊 Table" in text_block:
            table_blocks = [block for block in text_block.split("\n\n") if "📊 Table" in block]
            for tbl in table_blocks:
                summary = summarize_table(tbl)
                summarized_tables.append(f"{tbl}\n\n📝 Summary: {summary}")
    
    state["summarized_tables"] = summarized_tables
    update_status("Table Summarizer", "✅ Done")
    return state



def generate_proposal_node(state):
    update_status("Proposal Generator", "🧠 In Progress")
    proposal = expand_rfp(
        state["rfp_text"],
        state["retrieved_docs"],
        summarized_tables=state.get("summarized_tables", [])
    )
    update_status("Proposal Generator", "✅ Done")
    return {**state, "proposal": proposal}



def optimize_proposal_node(state):
    update_status("Strategy Optimizer", "🧠 In Progress")
    optimized = optimize_proposal_tone(
        state["proposal"],
        vertical=state.get("industry", "generic"),
        tone="persuasive"
)
    update_status("Strategy Optimizer", "✅ Done")
    return {**state, "proposal": optimized}


def check_compliance_node(state):
    update_status("Compliance Checker", "🧠 In Progress")
    report = check_compliance(state["rfp_text"], state["proposal"])
    state["compliance_report"] = report
    if "❌" in report:
        state["compliance_passed"] = False
    else:
        state["compliance_passed"] = True
    update_status("Compliance Checker", "✅ Done")
    return state


def score_proposal_node(state):
    update_status("Scorer", "🧠 In Progress")
    score_report = score_proposal_quality(state["proposal"])
    update_status("Scorer", "✅ Done")
    return {**state, "score_report": score_report}


# Build the graph
from typing import Dict, Any, TypedDict

class ProposalState(TypedDict, total=False):
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

# Pass it to the graph builder
builder = StateGraph(state_schema=ProposalState)


builder.add_node("Enrich RFP", enrich_rfp_node)
builder.add_node("Retrieve Docs", retrieve_docs_node)
builder.add_node("Summarize Tables", table_summary_node)
builder.add_node("Generate Proposal", generate_proposal_node)
builder.add_node("Optimize Tone", optimize_proposal_node)
builder.add_node("Check Compliance", check_compliance_node)
builder.add_node("Score Proposal", score_proposal_node)

# Define the main linear path
builder.set_entry_point("Enrich RFP")
builder.add_edge("Enrich RFP", "Retrieve Docs")
builder.add_edge("Retrieve Docs", "Summarize Tables")
builder.add_edge("Summarize Tables", "Generate Proposal")
builder.add_edge("Generate Proposal", "Optimize Tone")
builder.add_edge("Optimize Tone", "Check Compliance")

# Add conditional edge based on compliance
def compliance_condition(state):
    return "Score Proposal" if state.get("compliance_passed") else "Optimize Tone"

builder.add_conditional_edges("Check Compliance", compliance_condition, {
    "Score Proposal": "Score Proposal",
    "Optimize Tone": "Optimize Tone"
})

builder.add_edge("Score Proposal", END)

# Compile the graph
proposal_agentic_graph = builder.compile()
