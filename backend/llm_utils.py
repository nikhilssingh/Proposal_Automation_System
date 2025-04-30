# backend/llm_utils.py

import os
import re
import json
import json5
from dotenv import load_dotenv
import time
from typing import Dict
import fitz
import re

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

conversation_memory = {"latest_proposal": ""}
llm_usage_count = 0


# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Create a GPT-based LLM with a single import for ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",  # or "gpt-3.5-turbo" or "gpt-4"
    temperature=0.0
)

# Add rate limiting
LAST_CALL_TIME = 0
MIN_CALL_INTERVAL = 0.5  # 500ms between calls

def _rate_limited_call():
    global LAST_CALL_TIME
    current_time = time.time()
    elapsed = current_time - LAST_CALL_TIME
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    LAST_CALL_TIME = time.time()
    
from functools import lru_cache

import fitz  # PyMuPDF
from typing import Dict




def remove_unsupported_unicode(text: str) -> str:
    """Remove characters not supported by latin-1 encoding."""
    return text.encode('latin-1', errors='ignore').decode('latin-1')

@lru_cache(maxsize=32)
def summarize_text(long_text: str) -> str:
    """
    Summarize the text using the global LLM.
    This function sends a prompt to the LLM to create a short (~100-200 token) summary.
    """
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    try:
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Summarize the user's text in ~100-200 tokens. "
                "Focus only on core details. Output plain text."
            )),
            HumanMessage(content=long_text)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ Summarize text failed: {e}")
        return long_text[:1000]
    
def extract_rfp_metadata(rfp_text: str) -> dict:
    global llm_usage_count
    extracted_metadata = {
        "project_name": "",
        "client_name": "",
        "deadline": "",
        "industry": "generic",
        "region": "global",
        "constraints": [],
        "client_needs": []
    }

    def safe_json_parse(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ JSON decode failed. Attempting recovery with json5...")
            try:
                return json5.loads(content.strip())
            except Exception as e:
                print(f"❌ Recovery with json5 also failed: {e}")
                return {}

    chunk_size = 5000
    chunks = [rfp_text[i:i + chunk_size] for i in range(0, len(rfp_text), chunk_size)]

    for idx, chunk in enumerate(chunks[:1]):  # Only use first 1 chunk for metadata
        messages = [
            SystemMessage(content=(
                "Extract this metadata from the RFP chunk in pure JSON, with keys:\n"
                " - project_name\n - client_name\n - deadline\n"
                " - industry\n - region\n - constraints\n - client_needs\n\n"
                "Return JSON only. If unknown, leave them empty or null."
            )),
            HumanMessage(content=chunk)
        ]
        try:
            _rate_limited_call()
            llm_usage_count += 1
            response = llm.invoke(messages)
            content = response.content.strip()
            if not content.startswith("{"):
                print(f"⚠️ Chunk {idx} returned non-JSON:\n{content[:300]}")
                continue
            metadata_chunk = safe_json_parse(content)
            for k, v in metadata_chunk.items():
                if isinstance(v, list):
                    extracted_metadata.setdefault(k, [])
                    extracted_metadata[k].extend(v)
                elif v and not extracted_metadata.get(k):
                    extracted_metadata[k] = v
        except Exception as e:
            print(f"⚠️ Failed on chunk {idx}: {e}")
            continue

    # Deduplicate
    if isinstance(extracted_metadata["constraints"], list):
        extracted_metadata["constraints"] = list(set(extracted_metadata["constraints"]))
    if isinstance(extracted_metadata["client_needs"], list):
        extracted_metadata["client_needs"] = list(set(extracted_metadata["client_needs"]))

    return extracted_metadata


def expand_rfp(rfp_text, retrieved_docs, summarized_tables=None):
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    if retrieved_docs and isinstance(retrieved_docs, list):
        structured_context = "\n\n".join([
            f"🔹 **Reference Proposal {i+1}**:\n{doc}" for i, doc in enumerate(retrieved_docs)
        ])
    else:
        structured_context = "No similar documents found."
    summarized_tables = summarized_tables or []
    table_context = "\n\n".join(summarized_tables)
    prompt = f"""
You are a professional business consultant responding to a client’s RFP. 
Generate a **thorough business proposal** that addresses the client's needs.

--- 
**📜 Client’s RFP:**
{rfp_text}

--- 
**📂 Past Successful Proposals (Reference them as needed)**:
{structured_context}

--- 
**🧾 Summarized Table Insights**:
{table_context}

--- 
**Proposal Format**:
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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def refine_proposal(current_proposal: str, user_feedback: str) -> dict:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    prompt = f"""
You are an expert proposal writer. Given the current proposal below and the user feedback provided,
generate a refined proposal that incorporates the feedback and improves upon the original.

Current Proposal:
{current_proposal}

User Feedback:
{user_feedback}

Refined Proposal:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    refined_proposal = response.content.strip()
    conversation_memory["latest_proposal"] = refined_proposal
    return {"refined_proposal": refined_proposal}

def optimize_proposal_tone(proposal: str, vertical: str = "generic", tone: str = "professional") -> str:
    global llm_usage_count
    llm_usage_count += 1
    prompt = f"""
You are a senior business strategist. Optimize the following proposal to align with the '{vertical}' industry and a '{tone}' tone.

Original Proposal:
{proposal}

Revised / Optimized Proposal:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def check_compliance(rfp_text: str, proposal: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    prompt = f"""
You are a compliance auditor. Given the client's RFP and our proposal, check if we address key requirements, constraints, and mandatory elements.

Client RFP:
{rfp_text}

Our Proposal:
{proposal}

List major areas:
- ✅ Fully addressed
- ⚠️ Partially addressed
- ❌ Missing

Be detailed.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def score_proposal_quality(proposal: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    prompt = f"""
You are a senior proposal reviewer. Evaluate the following proposal from 1 to 10 in:
1) Clarity
2) Persuasiveness
3) Technical Depth
4) Alignment with Client Needs
5) Overall Quality

Proposal:
{proposal}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def summarize_table(markdown_table: str) -> str:
    global llm_usage_count
    _rate_limited_call()
    llm_usage_count += 1
    prompt = f"""
You are a business analyst. Summarize the purpose, structure, and key insights of this table:

{markdown_table}

Reply in 1-3 sentences.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()




