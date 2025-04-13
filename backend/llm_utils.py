# backend/llm_utils.py
import os
import re
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.0
)

def extract_rfp_metadata(rfp_text: str) -> dict:
    prompt = f"""
You are an intelligent assistant extracting structured metadata from a client's Request for Proposal (RFP).

Given the following RFP text:

{rfp_text}

Extract and return a JSON object with the following fields:
- project_name
- client_name (if available)
- deadline
- industry (e.g., retail, finance, healthcare)
- region (e.g., North America, Europe)
- constraints (list of limitations or must-haves)
- client_needs (list of pain points or goals mentioned)

Respond with only the JSON.
"""

    response = llm.invoke(prompt)
    
    try:
        import json
        return json.loads(response.content)
    except Exception as e:
        print("⚠️ Metadata extraction failed:", e)
        return {
            "project_name": "",
            "client_name": "",
            "deadline": "",
            "industry": "generic",
            "region": "global",
            "constraints": [],
            "client_needs": []
        }


def expand_rfp(rfp_text, retrieved_docs, summarized_tables=None):
    """Generates a thorough business proposal in response to an RFP, leveraging past proposals and summarized table insights."""

    structured_context = "\n\n".join([
        f"🔹 **Reference Proposal {i+1}**:\n{doc}" for i, doc in enumerate(retrieved_docs)
    ]) if retrieved_docs else "No similar documents found."
    
    summarized_tables = summarized_tables or []
    table_context = "\n\n".join(summarized_tables)

    prompt = f"""
You are a professional business consultant responding to a client’s RFP. Your task is to generate a **thorough business proposal** that directly addresses the client's needs.

---
**📜 Client’s RFP to Respond To:**
{rfp_text}

---
**📂 Past Successful Proposals (USE THESE TO SHAPE THE RESPONSE and fill in the company name, contact information, etc. and structure which is redundant from the past proposals):**
{structured_context}

---
**🧾 Table Insights (Summarized Explanations):**
{table_context}

---
**Proposal Format:**

📌 **Cover Letter**  
- Start with a compelling opening that differentiates us.  
- Showcase our expertise and success in similar projects.  
- End with a warm call to action.  

📌 **Understanding of Client Needs**  
- Identify key challenges mentioned in the RFP.  
- Use retrieved proposals to match solutions to the client’s goals.  

📌 **Proposed Solution**  
- Tailor the response using **retrieved past proposals** (inventory optimization, customer recommendations, etc.).  
- Clearly describe the AI-driven enhancements.  

📌 **Project Plan & Implementation Timeline**  
- **Assign team members** to each phase for credibility.  
- Provide detailed milestones and clear deliverables.  

📌 **Pricing & Payment Terms**  
- Extract competitive pricing from past proposals.  
- Justify the investment with **ROI-driven language**.  

📌 **Technical Approach**  
- Explain AI models, data processing, and security measures.  

📌 **Company Experience**  
- Highlight **measurable successes** from past projects.  
- Include relevant testimonials and case studies.  

📌 **Case Studies & Testimonials**  
- Use real success stories with **quantifiable impact** (e.g., 20% increase in efficiency).  

📌 **Conclusion & Call to Action**  
- End with a clear **next step** (e.g., scheduling a consultation call).  
- Ensure persuasive, client-centered writing.  

🎯 **Important:**  
- Reference **retrieved proposals** in relevant sections.  
- Use **table summaries** to justify decisions, showcase features, or support pricing or planning logic.
"""

    print(f"\n📝 Sending this prompt to GPT:\n{prompt[:1500]}")  # ✅ Debugging output

    response = llm.invoke(prompt)
    return response.content.strip()



# Maintain memory for ongoing refinements
conversation_memory = {"latest_proposal": ""}

# --- Create a conversational chain for proposal refinement ---

# Define a prompt template that will include the conversation history and new feedback.
refine_prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
You are an expert proposal writer tasked with refining a business proposal based on user feedback.

Conversation History:
{chat_history}

User Feedback:
{input}

Please produce an updated proposal that incorporates this feedback while preserving all previous refinements.
"""
)

# Create a memory object to hold the conversation history.
refine_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational chain using the prompt template and memory.
refine_chain = ConversationChain(
    llm=llm,
    prompt=refine_prompt_template,
    memory=refine_memory
)
 
def refine_proposal(current_proposal: str, user_feedback: str) -> dict:
    # Construct a prompt that combines the current proposal and the user feedback.
    prompt = f"""
You are an expert proposal writer. Given the current proposal below and the user feedback provided, generate a refined proposal that incorporates the feedback and improves upon the original.

Current Proposal:
{current_proposal}

User Feedback:
{user_feedback}

Refined Proposal:
"""
    # Call the LLM directly with the new prompt.
    response = llm.invoke(prompt)
    refined_proposal = response.content.strip()
    
    # Update the global conversation memory with the new refined proposal.
    conversation_memory["latest_proposal"] = refined_proposal
    
    return {"refined_proposal": refined_proposal}


def optimize_proposal_tone(proposal: str, vertical: str = "generic", tone: str = "professional") -> str:
    prompt = f"""
You are a senior business strategist. Your task is to optimize the following proposal to better align with the target industry and client expectations.

---
📝 **Original Proposal:**
{proposal}

🎯 **Target Industry (Vertical)**: {vertical}
🎙️ **Preferred Tone**: {tone}

---
Please revise the proposal accordingly. Ensure it's still well-structured, clear, and persuasive.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def check_compliance(rfp_text: str, proposal: str) -> str:
    prompt = f"""
You are a compliance auditor. Given the client's RFP and our current proposal draft, check if the proposal fully addresses all key requirements, constraints, and mandatory elements.

---
📜 **Client RFP:**
{rfp_text}

📝 **Our Proposal:**
{proposal}

---
List all major areas:
- ✅ Fully addressed
- ⚠️ Partially addressed
- ❌ Missing

Be detailed and structured.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def score_proposal_quality(proposal: str) -> str:
    prompt = f"""
You are a senior proposal reviewer. Evaluate the following proposal and assign scores (1 to 10) for:

1. Clarity of Communication
2. Persuasiveness & Tone
3. Technical Depth & Feasibility
4. Alignment with Client Needs
5. Overall Quality

---
📝 Proposal:
{proposal}

Return scores and a short explanation for each.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def summarize_table(markdown_table: str) -> str:
    prompt = f"""
You are a business analyst. Given the following table from a proposal or RFP, explain its purpose and contents in simple English.

Table:
{markdown_table}

Respond with a 1–3 sentence summary of what the table is about, what insights it provides, and which section of a proposal it might belong to.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def remove_unsupported_unicode(text: str) -> str:
    # Remove characters not supported by latin-1 encoding
    return text.encode('latin-1', errors='ignore').decode('latin-1')