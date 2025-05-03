# backend/llm_utils.py

import os
import json
import json5
import time
from functools import lru_cache
from dotenv import load_dotenv
from typing import Dict

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# In‑memory conversation and usage counters
conversation_memory = {"latest_proposal": ""}
llm_usage_count = 0

# Token tally and file path
from backend.path_utils import LOG_DIR
token_tally = {
    "fresh_prompt":  0,
    "cached_prompt": 0,
    "completion":    0,
    "training":      0,
    "embeddings":    0,   # ← newly added
}
TOKEN_FILE = LOG_DIR / "token_usage.json"

def _extract_usage(resp) -> dict:
    """Look for any token usage info in the LangChain/OpenAI response."""
    llm_out = getattr(resp, "llm_output", {}) or {}
    if isinstance(llm_out, dict):
        for key in ("token_usage", "usage"):
            if key in llm_out:
                return llm_out[key]
    if hasattr(resp, "usage"):
        # resp.usage may be a custom object; convert to dict if possible
        usage_obj = resp.usage
        try:
            return usage_obj.to_dict()
        except:
            return {
                "prompt_tokens":         getattr(usage_obj, "prompt_tokens", 0),
                "cached_prompt_tokens":  getattr(usage_obj, "cached_prompt_tokens", 0),
                "completion_tokens":     getattr(usage_obj, "completion_tokens", 0),
                "training_tokens":       getattr(usage_obj, "training_tokens", 0),
                "total_tokens":          getattr(usage_obj, "total_tokens", 0),
            }
    meta = getattr(resp, "additional_kwargs", {}) or {}
    return meta.get("usage", {})

def _record_usage(usage: dict):
    """Accumulate LLM call tokens into token_tally and write to disk."""
    token_tally["fresh_prompt"]  += usage.get("prompt_tokens", 0) - usage.get("cached_prompt_tokens", 0)
    token_tally["cached_prompt"] += usage.get("cached_prompt_tokens", 0)
    token_tally["completion"]    += usage.get("completion_tokens", 0)
    token_tally["training"]      += usage.get("training_tokens", 0)
    TOKEN_FILE.parent.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(token_tally, indent=2))

def _record_embedding_usage(usage: dict):
    """
    Accumulate embedding‐only tokens (the OpenAI embeddings endpoint usually
    reports them under `prompt_tokens` or `total_tokens`).
    """
    tokens = usage.get("prompt_tokens", usage.get("total_tokens", 0))
    token_tally["embeddings"] += tokens
    TOKEN_FILE.parent.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(token_tally, indent=2))

# LangChain‑OpenAI imports (community package)
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Instantiate the model
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.0
)

# Rate limiting between calls
LAST_CALL_TIME = 0
MIN_CALL_INTERVAL = 0.5  # seconds

def _rate_limited_call():
    global LAST_CALL_TIME
    now = time.time()
    if now - LAST_CALL_TIME < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - (now - LAST_CALL_TIME))
    LAST_CALL_TIME = time.time()

def remove_unsupported_unicode(text: str) -> str:
    return text.encode('latin-1', errors='ignore').decode('latin-1')

@lru_cache(maxsize=32)
def summarize_text(long_text: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Summarize the user's text in ~100-200 tokens. "
            "Focus only on core details. Output plain text."
        )),
        HumanMessage(content=long_text)
    ]

    result = llm.generate([messages])
    summary = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for summarize_text")

    return summary


def extract_rfp_metadata(rfp_text: str) -> dict:
    global llm_usage_count
    extracted = {
        "project_name": "", "client_name": "", "deadline": "",
        "industry": "generic", "region": "global",
        "constraints": [], "client_needs": []
    }

    def safe_json_parse(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return json5.loads(content)

    chunk = rfp_text[:5000]
    messages = [
        SystemMessage(content=(
            "Extract this metadata from the RFP chunk in pure JSON, with keys:\n"
            " - project_name\n - client_name\n - deadline\n"
            " - industry\n - region\n - constraints\n - client_needs\n\n"
            "Return JSON only. If unknown, leave empty or null."
        )),
        HumanMessage(content=chunk)
    ]

    _rate_limited_call()
    llm_usage_count += 1
    result = llm.generate([messages])
    resp_text = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for extract_rfp_metadata")

    try:
        meta = safe_json_parse(resp_text)
        for k, v in meta.items():
            if isinstance(v, list):
                extracted[k].extend(v)
            elif v:
                extracted[k] = v
    except Exception:
        pass

    # dedupe
    extracted["constraints"] = list(set(extracted["constraints"]))
    extracted["client_needs"] = list(set(extracted["client_needs"]))
    return extracted


def expand_rfp(rfp_text, retrieved_docs, summarized_tables=None) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    refs = "\n\n".join(f"🔹 **Reference {i+1}**:\n{d}"
                       for i, d in enumerate(retrieved_docs or [])) \
           or "No similar documents found."
    tables = "\n\n".join(summarized_tables or [])

    prompt = f"""
You are a professional business consultant responding to a client’s RFP. Generate a thorough business proposal.

Client’s RFP:
{rfp_text}

Past Proposals:
{refs}

Table Insights:
{tables}

Format:
1) Cover Letter
2) Understanding of Client Needs
3) Proposed Solution
4) Project Plan & Implementation Timeline
5) Pricing & Payment Terms
6) Technical Approach
7) Company Experience
8) Case Studies & Testimonials
9) Conclusion & Call to Action
"""

    result = llm.generate([[HumanMessage(content=prompt)]])
    proposal = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for expand_rfp")

    return proposal


def refine_proposal(current_proposal: str, user_feedback: str) -> dict:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    prompt = f"""
You are an expert proposal writer. Refine the proposal below using the feedback.

Current Proposal:
{current_proposal}

Feedback:
{user_feedback}

Refined Proposal:
"""
    result = llm.generate([[HumanMessage(content=prompt)]])
    refined = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for refine_proposal")

    conversation_memory["latest_proposal"] = refined
    return {"refined_proposal": refined}


def optimize_proposal_tone(proposal: str, vertical: str = "generic", tone: str = "professional") -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    prompt = f"""
Optimize the following proposal for a {vertical} industry with a {tone} tone.

Original Proposal:
{proposal}

Optimized Proposal:
"""
    result = llm.generate([[HumanMessage(content=prompt)]])
    optimized = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for optimize_proposal_tone")

    return optimized


def check_compliance(rfp_text: str, proposal: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    prompt = f"""
You are a compliance auditor. Check our proposal against the RFP below.

RFP:
{rfp_text}

Proposal:
{proposal}

Report:
"""
    result = llm.generate([[HumanMessage(content=prompt)]])
    report = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for check_compliance")

    return report


def score_proposal_quality(proposal: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    prompt = f"""
Score this proposal (1–10) on:
1) Clarity
2) Persuasiveness
3) Technical Depth
4) Alignment with Client Needs
5) Overall Quality

Proposal:
{proposal}
"""
    result = llm.generate([[HumanMessage(content=prompt)]])
    score = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for score_proposal_quality")

    return score


def summarize_table(markdown_table: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1

    prompt = f"""
Summarize this table in 1–3 sentences:

{markdown_table}
"""
    result = llm.generate([[HumanMessage(content=prompt)]])
    summary = result.generations[0][0].text.strip()

    usage = _extract_usage(result)
    if usage:
        _record_usage(usage)
    else:
        print("⚠️ No usage info returned for summarize_table")

    return summary


# Ensure token file exists at startup
LOG_DIR.mkdir(exist_ok=True, parents=True)
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(json.dumps(token_tally, indent=2))
