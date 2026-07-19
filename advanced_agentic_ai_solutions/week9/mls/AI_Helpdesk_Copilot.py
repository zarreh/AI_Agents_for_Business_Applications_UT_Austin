"""AI Helpdesk Copilot - script version of `AI_Helpdesk_Copilot_.ipynb`.

Converts the notebook into a single runnable Python script that:
- Loads config/credentials, sets up the LLMs and LangSmith tracing
- Loads and preprocesses the ticket dataset
- Builds/loads the FAISS knowledge-base vectorstore
- Defines the full multi-agent LangGraph workflow (intake -> triage ->
  [retrieve -> draft -> validate] OR [escalate])
- Exposes a module-level compiled `graph`, so this same file can be run
  directly with LangGraph Studio via:

    langgraph dev

  (see `langgraph.json` in this same directory, which points to
  `AI_Helpdesk_Copilot.py:graph`)

Run standalone (data overview + demo test cases, same as the notebook):
    python AI_Helpdesk_Copilot.py
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Literal, TypedDict

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import Client, traceable
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent

# ===============================================================
# LLM and Agent Observability Setup
# ===============================================================

# Load credentials from config.json (searched for by walking up from this
# file, so it works no matter what directory this script is launched from).
_config_path = None
for _parent in THIS_DIR.parents:
    _candidate = _parent / "config.json"
    if _candidate.exists():
        _config_path = _candidate
        break

if _config_path is not None:
    with open(_config_path, "r") as file:
        config = json.load(file)

    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    OPENAI_API_BASE = config.get("OPENAI_API_BASE")

    LANGCHAIN_TRACING_V2 = config.get("LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY = config.get("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = config.get("LANGCHAIN_PROJECT")

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if OPENAI_API_BASE:
        os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE
    if LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    if LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Initialize LLM
# a lightweight model for primary reasoning and generation
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# a more complex model for evaluation and validation
llm_validator = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize LangSmith client to enable tracing and evaluation of the
# agentic workflow, with automatic fallback if configuration is unavailable.
try:
    langsmith_client = Client()
    LANGSMITH_ENABLED = True
    print("LangSmith tracing enabled")
except Exception:
    LANGSMITH_ENABLED = False
    print("LangSmith not configured - tracing disabled")


# ===============================================================
# Data Loading & Preprocessing
# ===============================================================

def preprocess_tickets(df: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    """
    Clean and prepare ticket data for use in our system.

    Steps:
    1. Fill missing values in text columns
    2. Filter by language (we focus on English tickets)
    3. Combine subject + body into a single 'text' field

    Args:
        df:       Raw ticket DataFrame
        language: Language to filter on ('en' or 'de')
    Returns:
        Cleaned DataFrame ready for processing
    """
    out = df.copy()

    # Step 1: Fill any missing text with empty string
    for col in ["subject", "body", "answer"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    # Step 2: Filter to only English tickets
    if "language" in out.columns:
        out = (
            out[out["language"].str.lower() == language.lower()]
            .reset_index(drop=True)
        )

    # Step 3: Create combined text field (subject + body)
    # This gives the model the full context of the ticket
    out["text"] = out["subject"].str.strip() + " " + out["body"].str.strip()

    return out


file_path = THIS_DIR / "ticket_data.csv"
raw_df = pd.read_csv(file_path)
df = preprocess_tickets(raw_df, language="en")

print(f"Dataset shape: {df.shape}")
print(f"Unique queues: {df['queue'].nunique()}")


# ===============================================================
# Multi-Agent System Architecture
# ===============================================================

class EnhancedCopilotState(TypedDict, total=False):
    """
    Shared state passed between all agents in the workflow.

    Think of this as a shared notepad that every agent can read and write.
    - 'total=False' means all fields are optional (not required at start)
    - Each agent adds its outputs to this state
    - Later agents can use outputs from earlier agents

    Fields:
        ticket_text:        Input ticket (set at start, never changes)
        triage:             Queue/type/priority labels + confidence
        evidence_docs:      KB documents retrieved for this ticket
        retrieval_reasoning: Log of the iterative retrieval process
        retrieval_iterations: How many retrieval rounds were performed
        draft:              The drafted response (before validation)
        claim_analysis:     Result of claim-level grounding check
        policy_check:       Result of policy compliance check
        final_response:     The final approved response (or escalation message)
        escalated:          True if ticket was escalated to human
        escalation_reason:  Why it was escalated (triage/grounding/policy)
        reasoning_trail:    Complete log of all decisions made
    """
    # Input
    ticket_text: str
    # Triage output
    triage: dict
    # Retrieval output
    evidence_docs: List[Document]
    retrieval_reasoning: List[str]
    retrieval_iterations: int
    # Drafting output
    draft: str
    # Validation output
    claim_analysis: dict
    policy_check: dict
    # Final output
    final_response: str
    escalated: bool
    escalation_reason: str
    # Audit trail
    reasoning_trail: List[str]


# ===============================================================
# Agent 1: Intake Agent
# ===============================================================

@traceable(name="node_intake")
def intake_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 1: Intake Agent

    Job: Clean and normalize the incoming ticket text.
    Simple but important - removes extra whitespace, etc.
    Starts the reasoning trail that tracks all decisions.
    """
    text = (state.get("ticket_text") or "").strip()

    return {
        "ticket_text": text,
        "reasoning_trail": ["✓ Intake: Ticket received and normalized"],
    }


# ===============================================================
# Agent 2: Triage Agent
# ===============================================================

# Extract valid label sets from training data
# We pass these to the LLM so it only picks from real labels
VALID_QUEUES = sorted(df["queue"].dropna().unique().tolist())
VALID_TYPES = sorted(df["type"].dropna().unique().tolist())
VALID_PRIORITIES = sorted(df["priority"].dropna().unique().tolist())

print(f"Valid queues: {len(VALID_QUEUES)} options")
print(f"Valid types:  {VALID_TYPES}")
print(f"Valid priorities: {VALID_PRIORITIES}")


class TriageResult(BaseModel):
    """
    Structured output from the LLM triage agent.

    The LLM fills in these fields based on the ticket content.
    Using structured output ensures we always get a valid JSON response.
    """
    queue: str = Field(..., description="Which team should handle this ticket")
    type: str = Field(..., description="Category of ticket: incident/request/problem/change")
    priority: str = Field(..., description="Urgency level of the ticket")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Brief explanation of why these labels were chosen")


# The triage prompt instructs the LLM to act as a support classifier
triage_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert support ticket classifier.

Your job is to read a customer support ticket and predict:
1. Which team (queue) should handle it
2. What type of ticket it is
3. The priority level

Valid queues to choose from:
{valid_queues}

Valid types: {valid_types}
Valid priorities: {valid_priorities}

Rules:
- Pick the CLOSEST matching queue from the valid list
- Set confidence between 0.0 (very unsure) and 1.0 (very sure)
- If the ticket is vague or could match many queues, use lower confidence
- Always provide a brief reasoning for your classification

Return JSON matching the schema."""),
    ("human", "Ticket:\n{ticket_text}"),
])

# Chain: prompt → LLM with structured output (returns TriageResult object)
triage_chain = triage_prompt | llm_fast.with_structured_output(TriageResult)


@traceable(name="llm_triage_predict")
def llm_triage_predict(ticket_text: str) -> dict:
    """
    Use an LLM to classify a ticket into queue, type, and priority.

    Args:
        ticket_text: Combined subject + body of the ticket
    Returns:
        Dictionary with queue, type, priority, confidence, and reasoning
    """
    result = triage_chain.invoke({
        "ticket_text": ticket_text,
        "valid_queues": "\n".join(f"- {q}" for q in VALID_QUEUES[:20]),  # Top 20 to save tokens
        "valid_types": ", ".join(VALID_TYPES),
        "valid_priorities": ", ".join(str(p) for p in VALID_PRIORITIES),
    })

    return {
        "queue": {"label": result.queue, "confidence": result.confidence},
        "type": {"label": result.type, "confidence": result.confidence},
        "priority": {"label": result.priority, "confidence": result.confidence},
        "overall_confidence": result.confidence,
        "reasoning": result.reasoning,
    }


@traceable(name="node_triage")
def triage_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 2: Triage Agent (LLM-based)

    Job: Classify the ticket into queue/type/priority.
    Produces a confidence score that determines next step.

    LOW confidence  → escalate_node (human handles it)
    HIGH confidence → iterative_retrieve_node (AI handles it)
    """
    triage = llm_triage_predict(state["ticket_text"])

    trail = state.get("reasoning_trail", [])
    trail.append(
        f"✓ Triage: Queue={triage['queue']['label']} | "
        f"Type={triage['type']['label']} | "
        f"Confidence={triage['overall_confidence']:.2f}"
    )

    return {"triage": triage, "reasoning_trail": trail}


TRIAGE_CONF_THRESHOLD = 0.2


def route_after_triage(state: EnhancedCopilotState) -> Literal["retrieve", "escalate"]:
    """
    Decision function: After triage, where do we go?

    HIGH confidence (≥ threshold) → retrieve → draft → validate
    LOW confidence  (< threshold) → escalate → END

    This prevents the AI from handling tickets it's unsure about,
    reducing the risk of wrong routing.
    """
    conf = state["triage"].get("overall_confidence", 0.2)
    decision = "retrieve" if conf >= TRIAGE_CONF_THRESHOLD else "escalate"
    return decision


# ===============================================================
# Agent 3: Iterative Retrieval Agent
# ===============================================================

EMBED_MODEL = "BAAI/bge-large-en"
VECTORSTORE_PATH = str(THIS_DIR / "helpdesk_vectorstore")  # Save vectorstore to disk


def build_kb_documents(df: pd.DataFrame, max_docs: int = 500) -> List[Document]:
    """
    Build a knowledge base from historical ticket answers.

    Each document contains a past agent's answer, with metadata about
    what kind of ticket it came from. This forms our 'knowledge base'
    that the RAG system will search through.

    Args:
        df:       DataFrame containing historical tickets with answers
        max_docs: Maximum number of documents to include (keep small for demo)
    Returns:
        List of LangChain Document objects ready for embedding
    """
    small, _ = train_test_split(
        df,
        train_size=min(max_docs, len(df)),
        stratify=df["queue"],
        random_state=42
    )

    docs: List[Document] = []
    for i, row in small.iterrows():
        content = row.get("answer", "").strip()
        if not content:
            continue  # Skip tickets with no answer

        # Metadata helps filter and cite documents later
        meta = {
            "kb_id": f"KB_{i}",
            "queue": row.get("queue"),
            "type": row.get("type"),
            "priority": row.get("priority"),
            "subject": row.get("subject"),
        }
        docs.append(Document(page_content=content, metadata=meta))

    return docs


def get_vectorstore(docs: List[Document], save_path: str) -> FAISS:
    """
    Get vectorstore from disk (if exists) or create and save it.

    This function checks if we already built the vectorstore.
    If yes → load from disk (fast, free)
    If no  → build it and save to disk for next time

    Args:
        docs:      List of KB documents to embed
        save_path: Directory path to save/load the vectorstore
    Returns:
        FAISS vectorstore ready for similarity search
    """
    # Using HuggingFace embeddings - completely FREE
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU (works on all machines)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for better similarity
    )

    # Check if vectorstore already exists on disk
    if Path(save_path).exists():
        print(f"Loading existing vectorstore from: {save_path}")
        print("   (Skipping expensive embedding step - using cached version)")
        vs = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"⏳ Building vectorstore from {len(docs)} documents...")
        print("   (This runs once, then gets saved to disk)")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(save_path)
        print(f"Vectorstore saved to: {save_path}")

    return vs


# Build KB documents from training data answers
kb_docs = build_kb_documents(df, max_docs=500)

# Get or create the vectorstore
vectorstore = get_vectorstore(kb_docs, VECTORSTORE_PATH)
print(f"\n Knowledge base ready: {len(kb_docs)} documents indexed")


class RetrievalPlan(BaseModel):
    """
    Structured plan created by the Planner agent.

    Instead of just searching with the raw ticket text,
    the planner generates targeted queries to find exactly
    what's needed to answer this specific ticket.
    """
    queries: List[str] = Field(
        ...,
        description="2-5 specific search queries to find relevant KB articles"
    )
    info_needed: List[str] = Field(
        ...,
        description="What information is needed to answer this ticket"
    )
    confidence: str = Field(
        ...,
        description="Planner confidence: high/medium/low"
    )


planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a support assistant planning knowledge base searches.

Given a support ticket, create targeted search queries to find relevant articles.

Rules:
- Generate 2-5 short, specific queries (not full sentences)
- Focus on: product names, error codes, symptoms, processes
- Include a broad fallback query if unsure
- List what information you need to find

Return JSON matching the schema."""),
    ("human", """Ticket: {ticket_text}

Triage labels: {triage}"""),
])

# Note: using llm_fast here (gpt-4o-mini) for cost efficiency
planner_chain = planner_prompt | llm_fast.with_structured_output(RetrievalPlan)


@traceable(name="plan_retrieval")
def plan_retrieval(ticket_text: str, triage: dict) -> RetrievalPlan:
    """
    Plan what to search for in the knowledge base.

    The planner reads the ticket and triage labels, then generates
    specific search queries. This is smarter than just searching
    with the full ticket text.
    """
    return planner_chain.invoke({
        "ticket_text": ticket_text,
        "triage": triage,
    })


@traceable(name="retrieve_evidence")
def retrieve_evidence(vectorstore: FAISS, queries: List[str], k: int = 4) -> List[Document]:
    """
    Retrieve relevant KB documents using vector similarity search.

    For each query, we find the k most similar KB documents.
    We deduplicate by kb_id to avoid returning the same document twice.

    Args:
        vectorstore: The FAISS vector store containing KB documents
        queries:     List of search queries from the planner
        k:           Number of results per query
    Returns:
        Deduplicated list of relevant KB documents
    """
    seen = set()
    out: List[Document] = []

    for q in queries:
        # similarity_search finds documents whose embeddings are
        # closest to the query embedding in vector space
        hits = vectorstore.similarity_search(q, k=k)
        for d in hits:
            kb_id = d.metadata.get("kb_id")
            if kb_id and kb_id in seen:
                continue  # Skip duplicates
            seen.add(kb_id)
            out.append(d)

    return out


def format_evidence(docs: List[Document], max_chars: int = 3500) -> str:
    """
    Format KB documents into a readable evidence bundle for the drafter.

    Each document is formatted as [KB_ID] content, so the drafter
    can cite specific sources like "(source: KB_42)".

    Args:
        docs:      Retrieved KB documents
        max_chars: Maximum total characters to include (controls token usage)
    Returns:
        Formatted string of evidence with source IDs
    """
    parts = []
    total = 0

    for d in docs:
        kb_id = d.metadata.get("kb_id", "unknown")
        snippet = d.page_content.strip().replace("\n", " ")
        chunk = f"[{kb_id}] {snippet}\n\n"

        if total + len(chunk) > max_chars:
            break  # Stop if we're approaching the limit

        parts.append(chunk)
        total += len(chunk)

    return "".join(parts)


class RetrievalQuality(BaseModel):
    """
    Assessment of whether retrieved evidence is sufficient.

    After the first retrieval, the agent evaluates if it has
    enough information to answer the ticket. If not, it suggests
    refined queries to find the missing information.
    """
    sufficient: bool = Field(
        ...,
        description="True if evidence is sufficient to answer the ticket"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="What information is still missing"
    )
    refined_queries: List[str] = Field(
        default_factory=list,
        description="Better search queries to find missing information"
    )


evidence_assessor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are evaluating whether retrieved knowledge base articles
are sufficient to answer a support ticket.

Check if the evidence covers:
- The root cause or explanation of the issue
- Specific troubleshooting steps
- Any relevant policies or SLA information

If insufficient, suggest refined queries to find the missing pieces."""),
    ("human", """Ticket: {ticket}

Retrieved Evidence:
{evidence}

Original Search Queries: {original_queries}"""),
])

# Note: using llm_fast (gpt-4o-mini) for quality assessment
evidence_assessor = evidence_assessor_prompt | llm_fast.with_structured_output(RetrievalQuality)


@traceable(name="assess_retrieval_quality")
def assess_retrieval_quality(
    ticket_text: str,
    evidence_docs: List[Document],
    original_queries: List[str]
) -> RetrievalQuality:
    """
    Assess if retrieved evidence is sufficient to answer the ticket.

    This is the key step that makes retrieval "agentic" -
    the system evaluates its own retrieval and decides whether
    to search again with better queries.
    """
    evidence = format_evidence(evidence_docs, max_chars=2000)
    return evidence_assessor.invoke({
        "ticket": ticket_text,
        "evidence": evidence,
        "original_queries": original_queries,
    })


@traceable(name="iterative_retrieval")
def iterative_retrieval(ticket_text: str, triage: dict, vectorstore: FAISS, max_iterations: int = 2) -> tuple:
    """
    Perform multi-step retrieval with quality assessment.

    ITERATION 1:
      → Plan queries based on ticket + triage
      → Retrieve initial evidence
      → Assess quality

    ITERATION 2 (if needed):
      → Generate refined queries based on gaps
      → Retrieve additional evidence
      → Combine with iteration 1 results

    Args:
        ticket_text:    The support ticket text
        triage:         Triage labels (queue, type, priority)
        vectorstore:    The KB vector store
        max_iterations: Maximum retrieval rounds (usually 2 is enough)
    Returns:
        Tuple of (all_docs, reasoning_trail)
    """
    reasoning_trail = []
    all_docs = []

    # ----- ITERATION 1: Initial Retrieval -----
    plan = plan_retrieval(ticket_text, triage)
    reasoning_trail.append(f"ITERATION 1 - Queries: {plan.queries}")

    docs = retrieve_evidence(vectorstore, plan.queries, k=4)
    all_docs.extend(docs)
    reasoning_trail.append(f"ITERATION 1 - Retrieved {len(docs)} documents")

    # ----- QUALITY CHECK & OPTIONAL ITERATION 2 -----
    for iteration in range(2, max_iterations + 1):
        quality = assess_retrieval_quality(ticket_text, all_docs, plan.queries)
        reasoning_trail.append(
            f"ITERATION {iteration} - Evidence sufficient: {quality.sufficient}"
        )

        if quality.sufficient:
            reasoning_trail.append(f"ITERATION {iteration} - Stopping (evidence is sufficient)")
            break

        if not quality.refined_queries:
            reasoning_trail.append(f"ITERATION {iteration} - No refined queries, stopping")
            break

        # Retrieve with refined queries to fill the gaps
        reasoning_trail.append(
            f"ITERATION {iteration} - Refined queries: {quality.refined_queries}"
        )
        new_docs = retrieve_evidence(vectorstore, quality.refined_queries, k=3)
        all_docs.extend(new_docs)
        reasoning_trail.append(
            f"ITERATION {iteration} - Added {len(new_docs)} more documents"
        )

    return all_docs, reasoning_trail


print("Iterative retrieval module ready (up to 2 rounds)")


@traceable(name="node_iterative_retrieve")
def iterative_retrieve_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 3: Retrieval Planner + Retriever (Agentic RAG)

    Job: Find relevant KB documents to answer the ticket.
    Uses iterative retrieval with quality assessment:
      Round 1: Initial search with planned queries
      Round 2: Refined search if evidence is insufficient

    This agent demonstrates multi-step reasoning.
    """
    docs, reasoning = iterative_retrieval(
        state["ticket_text"],
        state["triage"],
        vectorstore,
        max_iterations=2
    )

    trail = state.get("reasoning_trail", [])
    trail.extend([f"  {r}" for r in reasoning])

    return {
        "evidence_docs": docs,
        "retrieval_reasoning": reasoning,
        "retrieval_iterations": len([r for r in reasoning if "ITERATION" in r]),
        "reasoning_trail": trail,
    }


# ===============================================================
# Agent 4: Response Drafting Agent
# ===============================================================

draft_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior support engineer writing a first-response email.

IMPORTANT RULES:
- Only use information from the provided evidence
- Cite every factual claim with (source: KB_X)
- Do NOT invent steps, settings, or policies
- If evidence is insufficient, say so and suggest escalation
- Be professional but friendly

Format: Brief acknowledgment → Steps/solution → Next steps"""),
    ("human", """Ticket: {ticket_text}

Triage: {triage}

Evidence from Knowledge Base:
{evidence}

Write the response now."""),
])

drafter_chain = draft_prompt | llm_fast | StrOutputParser()


@traceable(name="draft_response")
def draft_response(ticket_text: str, triage: dict, docs: List[Document]) -> str:
    """
    Generate a first-response draft using retrieved evidence.

    The drafter is given the ticket, its triage labels, and
    the retrieved KB documents. It must cite every claim.
    """
    evidence = format_evidence(docs)
    return drafter_chain.invoke({
        "ticket_text": ticket_text,
        "triage": triage,
        "evidence": evidence,
    })


print("Agentic RAG components ready (Planner + Retriever + Drafter)")


@traceable(name="node_draft")
def draft_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 4: Response Drafting Agent

    Job: Write the first-response email using retrieved evidence.
    Must cite every factual claim using (source: KB_X) format.
    The response is not final yet - it goes to the validator next.
    """
    draft = draft_response(
        state["ticket_text"],
        state["triage"],
        state["evidence_docs"],
    )

    trail = state.get("reasoning_trail", [])
    trail.append(f"✓ Draft: Generated response ({len(draft)} chars)")

    return {"draft": draft, "reasoning_trail": trail}


# ===============================================================
# Agent 5: Grounding & Validation Agent
# ===============================================================

class Claim(BaseModel):
    """
    Represents a single factual claim found in the response.

    Example claims:
    - "Password reset emails arrive within 5 minutes" → needs citation
    - "Thank you for contacting us" → no citation needed (general courtesy)
    - "Clear your browser cache" → needs citation (troubleshooting step)
    """
    claim_text: str = Field(..., description="The specific claim being made")
    needs_evidence: bool = Field(..., description="True if this claim needs a citation")
    cited_sources: List[str] = Field(
        default_factory=list,
        description="KB source IDs cited for this claim (e.g., KB_42)"
    )
    is_grounded: bool = Field(
        ...,
        description="True if claim has valid evidence citation"
    )


class ClaimAnalysis(BaseModel):
    """
    Complete analysis of all claims in a response.

    The validator extracts every factual claim and checks if each one
    has a valid source citation. If any claim is ungrounded → escalate.
    """
    claims: List[Claim] = Field(..., description="All claims found in the response")
    overall_grounded: bool = Field(
        ...,
        description="True only if ALL claims requiring evidence are grounded"
    )
    ungrounded_claims: List[str] = Field(
        default_factory=list,
        description="Text of claims that lack proper evidence"
    )


claim_extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous fact-checker for customer support responses.

Extract every factual claim from the response and check if it's properly cited.

A claim NEEDS evidence if it:
- States specific steps, settings, or procedures
- Makes promises about timelines or outcomes
- References policies or limits
- Gives technical troubleshooting advice

A claim does NOT need evidence if it:
- Is general courtesy ("Thank you for contacting us")
- Is a question to the customer
- Is an offer to help

For each claim, check if it cites a valid source using (source: KB_X) format.
Valid source IDs provided: {valid_sources}

Return JSON matching the schema."""),
    ("human", """Response to analyze:
{response}"""),
])

# Using llm_validator (gpt-4o) here for better scrutiny
claim_extractor = claim_extractor_prompt | llm_validator.with_structured_output(ClaimAnalysis)


@traceable(name="extract_and_validate_claims")
def extract_and_validate_claims(response: str, evidence_docs: List[Document]) -> ClaimAnalysis:
    """
    Extract every claim from the response and verify each has evidence.

    This is the core of our anti-hallucination system.
    Uses gpt-4o for thorough analysis.

    Args:
        response:      The drafted support response to validate
        evidence_docs: The KB documents retrieved for this ticket
    Returns:
        ClaimAnalysis with details on every claim
    """
    valid_sources = sorted({
        d.metadata.get("kb_id")
        for d in evidence_docs
        if d.metadata.get("kb_id")
    })

    return claim_extractor.invoke({
        "response": response,
        "valid_sources": valid_sources,
    })


# Company policies define what the AI is and isn't allowed to say
COMPANY_POLICIES = {
    "no_refund_promises": (
        "Never promise refunds without manager approval. "
        "Use: 'I'll check our refund policy for your specific case.'"
    ),
    "no_credential_requests": (
        "Never ask for passwords or login credentials. "
        "Always use official password reset flows only."
    ),
    "escalate_legal_issues": (
        "Any legal, compliance, or regulatory questions must be "
        "escalated immediately to the legal team."
    ),
    "data_privacy": (
        "Never share one customer's data with another. "
        "Always verify customer identity before discussing account details."
    ),
    "no_specific_timelines": (
        "Don't promise specific resolution times. "
        "Use: 'We'll work to resolve this as quickly as possible.'"
    ),
}


class PolicyCheckResult(BaseModel):
    """
    Result of checking a response against company policies.

    Normally, an LLM returns plain text. But we need structured data
    (compliant: True/False, violations: list, etc.) that we can use
    in our code. `.with_structured_output(PolicyCheckResult)` tells the LLM
    to return JSON that matches the PolicyCheckResult Pydantic model.
    LangChain then automatically parses this into a Python object.

    This is much more reliable than asking for JSON and parsing manually,
    because the LLM is specifically instructed to match the schema.
    """
    compliant: bool = Field(
        ...,
        description="True if response follows all company policies"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of policy violations found in the response"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="How to fix each violation"
    )


policy_checker_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a compliance officer reviewing support responses.

Company Policies:
{policies}

Check if the response violates any of these policies.
Be thorough - check for unauthorized promises, security issues,
missing escalation triggers, and privacy concerns.

Return JSON matching the schema."""),
    ("human", """Customer Ticket:
{ticket}

Support Response to Review:
{response}"""),
])

# Using llm_validator (gpt-4o) for policy checks too
policy_checker = policy_checker_prompt | llm_validator.with_structured_output(PolicyCheckResult)


@traceable(name="check_policy_compliance")
def check_policy_compliance(ticket_text: str, response: str) -> PolicyCheckResult:
    """
    Check if a response complies with all company policies.

    Uses gpt-4o for thorough compliance analysis.
    Even small violations are caught and reported.

    Args:
        ticket_text: Original ticket (context for policy check)
        response:    Drafted response to check
    Returns:
        PolicyCheckResult with compliance status and violations
    """
    policies_text = "\n".join([
        f"• {k}: {v}"
        for k, v in COMPANY_POLICIES.items()
    ])

    return policy_checker.invoke({
        "ticket": ticket_text,
        "response": response,
        "policies": policies_text,
    })


@traceable(name="node_validate")
def validate_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 5: Grounding & Validation Agent

    Job: Quality gate before the response is sent.
    Performs TWO checks:

    CHECK 1 - Claim-Level Grounding:
      Extracts every factual claim and verifies it has a valid citation.
      If any claim is ungrounded → escalate.

    CHECK 2 - Policy Compliance:
      Checks all company policies (no refund promises, no passwords, etc.)
      If any violation found → escalate.

    Only if BOTH checks pass → response is approved and sent.
    """
    # Run both validation checks
    claim_analysis = extract_and_validate_claims(
        state["draft"],
        state["evidence_docs"]
    )
    policy_check = check_policy_compliance(
        state["ticket_text"],
        state["draft"]
    )

    trail = state.get("reasoning_trail", [])
    trail.append(
        f"✓ Validation: Grounded={claim_analysis.overall_grounded} | "
        f"Policy={policy_check.compliant}"
    )

    # CHECK 1: Are all claims grounded?
    if not claim_analysis.overall_grounded:
        trail.append(
            f"✗ Escalating: Ungrounded claims found: {claim_analysis.ungrounded_claims[:2]}"
        )
        return {
            "claim_analysis": claim_analysis.model_dump(),
            "policy_check": policy_check.model_dump(),
            "escalated": True,
            "escalation_reason": (
                f"Ungrounded claims: {', '.join(claim_analysis.ungrounded_claims[:2])}"
            ),
            "final_response": (
                "I want to ensure you receive accurate information. "
                "Let me connect you with a specialist who can verify the details.\n\n"
                f"(Issue: Response contained claims without supporting evidence)"
            ),
            "reasoning_trail": trail,
        }

    # CHECK 2: Does response comply with policies?
    if not policy_check.compliant:
        trail.append(
            f"✗ Escalating: Policy violations: {policy_check.violations}"
        )
        return {
            "claim_analysis": claim_analysis.model_dump(),
            "policy_check": policy_check.model_dump(),
            "escalated": True,
            "escalation_reason": (
                f"Policy violations: {', '.join(policy_check.violations)}"
            ),
            "final_response": (
                "Thank you for contacting us. To ensure this is handled appropriately, "
                "I'm routing you to the right team.\n\n"
                f"(Note: Compliance review triggered)"
            ),
            "reasoning_trail": trail,
        }

    # ALL CHECKS PASSED → approve response
    trail.append("✓ Validation: All checks passed - response approved!")
    return {
        "claim_analysis": claim_analysis.model_dump(),
        "policy_check": policy_check.model_dump(),
        "final_response": state["draft"],
        "escalated": False,
        "reasoning_trail": trail,
    }


# ===============================================================
# Agent 6: Escalate Agent
# ===============================================================

@traceable(name="node_escalate")
def escalate_node(state: EnhancedCopilotState) -> dict:
    """
    AGENT 6: Escalation Agent

    Job: Handle tickets that couldn't be processed automatically.
    This is triggered when:
    - Triage confidence is too low (uncertain routing)

    The response tells the customer their issue is being
    routed to a human agent.
    """
    triage = state.get("triage", {})
    trail = state.get("reasoning_trail", [])
    trail.append(
        f"✗ Escalation: Low confidence ({triage.get('overall_confidence', 0):.2f}) "
        f"- routing to human"
    )

    return {
        "final_response": (
            "Thank you for reaching out we want to make sure this is handled correctly.\n\n"
            "I'm connecting you with a specialist for your request.\n"
            f"(Predicted team: {triage.get('queue', {}).get('label', 'Support Team')})"
        ),
        "escalated": True,
        "escalation_reason": "Low triage confidence",
        "reasoning_trail": trail,
    }


# ===============================================================
# Multi-Agent System Workflow
# ===============================================================

workflow = StateGraph(EnhancedCopilotState)

# Add all agent nodes to the graph
workflow.add_node("intake", intake_node)
workflow.add_node("triage", triage_node)
workflow.add_node("retrieve", iterative_retrieve_node)
workflow.add_node("draft", draft_node)
workflow.add_node("validate", validate_node)
workflow.add_node("escalate", escalate_node)

# Set the starting point
workflow.set_entry_point("intake")

# Add edges (connections between nodes)
workflow.add_edge("intake", "triage")  # Always go from intake to triage

# Conditional routing after triage (uses route_after_triage function)
workflow.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "retrieve": "retrieve",  # High confidence → retrieval
        "escalate": "escalate",  # Low confidence → escalation
    }
)

# The happy path: retrieve → draft → validate → END
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "validate")
workflow.add_edge("validate", END)
workflow.add_edge("escalate", END)

# Compile the graph into a runnable application.
# `graph` is what `langgraph dev` loads (see langgraph.json: "helpdesk_copilot": "./AI_Helpdesk_Copilot.py:graph")
graph = workflow.compile()
app = graph  # kept for parity with the notebook, which calls the compiled app `app`

print(" LangGraph multi-agent system compiled successfully!")
print("   Agents: intake → triage → [retrieve → draft → validate] OR [escalate]")


# ===============================================================
# Test Cases
# ===============================================================

TEST_CASES = [
    {
        "id": 1,
        "scenario": "Simple Password Reset (Common IT Ticket)",
        "ticket": (
            "Subject: Cannot access my account\n"
            "I forgot my password and need help resetting it. "
            "I tried clicking 'Forgot Password' but I'm not receiving the email. "
            "My email address is correct."
        ),
        "expected_outcome": "Full Pipeline - High confidence routing expected",
        "what_to_observe": (
            "Watch for: High triage confidence, retrieval finds password reset articles, "
            "response cites KB sources, validation passes."
        )
    },
    {
        "id": 2,
        "scenario": "Billing Dispute (Finance Queue)",
        "ticket": (
            "Subject: Incorrect charge on invoice\n"
            "I was charged twice for my monthly subscription this month. "
            "Invoice #INV-2024-1234 shows a duplicate charge of $49.99. "
            "Please review and refund the duplicate charge."
        ),
        "expected_outcome": "Full Pipeline or Policy Escalation (refund promise check)",
        "what_to_observe": (
            "Watch for: Policy checker catching any refund promises, "
            "validation using gpt-4o for careful scrutiny."
        )
    },
    {
        "id": 3,
        "scenario": "Technical Software Bug (Engineering Queue)",
        "ticket": (
            "Subject: Application crashes on startup\n"
            "The application crashes every time I try to open it since yesterday's update. "
            "Error message: 'Fatal error: module not found'. "
            "I'm on Windows 11, version 22H2."
        ),
        "expected_outcome": "Full Pipeline - Technical ticket should be well-handled",
        "what_to_observe": (
            "Watch for: Iterative retrieval (may need 2 rounds to find crash solutions), "
            "specific troubleshooting steps cited from KB."
        )
    },
    {
        "id": 4,
        "scenario": "Real Ticket from Dataset (Unknown Confidence)",
        "ticket": df.sample(1, random_state=42).iloc[0]["text"],
        "expected_outcome": "Depends on triage confidence",
        "what_to_observe": (
            "Watch for: Real-world ticket behavior, confidence score, "
            "whether AI routes correctly vs escalates."
        )
    },
    {
        "id": 5,
        "scenario": "Vague Incomplete Request (Low Confidence Expected)",
        "ticket": (
            "Subject: It's not working\n"
            "Hi, I have a problem with the system. "
            "Nothing is working properly and I need help urgently. "
            "Please fix it."
        ),
        "expected_outcome": "Escalation - Too vague for confident triage",
        "what_to_observe": (
            "Watch for: Low confidence score (< 0.20), immediate escalation, "
            "no retrieval or drafting steps performed."
        )
    },
]


def run_test_case(test: dict, app) -> dict:
    """
    Run a single test case and display formatted results.

    Args:
        test:     Test case dictionary
        app:      Compiled LangGraph application
    Returns:
        Dictionary of metrics for summary table
    """

    # --- INPUT ---
    print(f"\n📋 INPUT TICKET:")
    print("-" * 40)
    print(test["ticket"][:300] + ("..." if len(test["ticket"]) > 300 else ""))
    print(f"\n📌 Expected: {test['expected_outcome']}")
    print(f"👀 Observe:  {test['what_to_observe']}")

    # --- RUN THROUGH PIPELINE ---
    print(f"\n⚙️  RUNNING THROUGH MULTI-AGENT PIPELINE...")
    result = app.invoke({"ticket_text": test["ticket"]})

    # --- TRIAGE RESULTS ---
    triage = result.get("triage", {})
    conf = triage.get("overall_confidence", 0)

    print(f"\n🎯 TRIAGE RESULTS:")
    print(f"   Queue:      {triage.get('queue', {}).get('label', 'N/A')}")
    print(f"   Type:       {triage.get('type', {}).get('label', 'N/A')}")
    print(f"   Priority:   {triage.get('priority', {}).get('label', 'N/A')}")
    print(f"   Confidence: {conf:.3f} {'✅' if conf >= TRIAGE_CONF_THRESHOLD else '⚠️ (below threshold → escalate)'}")
    if triage.get("reasoning"):
        print(f"   Reasoning:  {triage.get('reasoning', '')[:120]}...")

    # --- REASONING TRAIL ---
    trail = result.get("reasoning_trail", [])
    print(f"\n🔍 REASONING TRAIL ({len(trail)} steps):")
    for step in trail:
        print(f"   {step}")

    # --- RETRIEVAL DETAILS (if reached) ---
    iterations = result.get("retrieval_iterations", 0)
    if iterations > 0:
        docs = result.get("evidence_docs", [])
        print(f"\n📚 RETRIEVAL:")
        print(f"   Iterations:  {iterations}")
        print(f"   Total docs:  {len(docs)}")
        if docs:
            print(f"   Sources:     {[d.metadata.get('kb_id') for d in docs[:5]]}")

    # --- VALIDATION DETAILS (if reached) ---
    claim_analysis = result.get("claim_analysis", {})
    policy_check = result.get("policy_check", {})

    if claim_analysis:
        print(f"   Claims grounded:  {claim_analysis.get('overall_grounded', 'N/A')}")
        ungrounded = claim_analysis.get("ungrounded_claims", [])
        if ungrounded:
            print(f"   Ungrounded:       {ungrounded[:2]}")

    if policy_check:
        print(f"   Policy compliant: {policy_check.get('compliant', 'N/A')}")
        violations = policy_check.get("violations", [])
        if violations:
            print(f"   Violations:       {violations[:2]}")

    # --- FINAL RESPONSE ---
    escalated = result.get("escalated", False)
    print(f"\n{'🚨 ESCALATED' if escalated else '✅ APPROVED'} RESPONSE:")
    if escalated:
        print(f"   Reason: {result.get('escalation_reason', 'N/A')}")
    print("-" * 40)
    response = result.get("final_response", "")
    print(response[:400] + ("..." if len(response) > 400 else ""))

    # --- OBSERVATIONS ---
    print(f"\n📝 OBSERVATIONS:")
    reasoning_steps = len(trail)
    if reasoning_steps <= 3:
        print("   → Early escalation path (only 3 steps: intake → triage → escalate)")
    elif not escalated:
        print("   → Full pipeline executed (7+ steps including retrieval + validation)")
    else:
        print("   → Pipeline ran but validation/policy triggered escalation")

    if "(source:" in response:
        citation_count = len(re.findall(r"\(source:\s*KB_\d+\)", response))
        print(f"   → Response contains {citation_count} source citation(s) ✅")
    elif not escalated:
        print("   → ⚠️ Response lacks source citations")

    print(f"   → Check LangSmith traces at: https://smith.langchain.com")
    print(f"      Project: helpdesk-copilot-student-demo")

    return {
        "test_id": test["id"],
        "scenario": test["scenario"][:40],
        "escalated": escalated,
        "triage_conf": conf,
        "reasoning_steps": reasoning_steps,
        "retrieval_iterations": iterations,
        "grounded": claim_analysis.get("overall_grounded", "N/A"),
        "policy_ok": policy_check.get("compliant", "N/A"),
    }


TEST_RUN_HISTORY = []


def show_test_summary():
    """
    Display cumulative summary of all executed tests.
    """
    if not TEST_RUN_HISTORY:
        print("No tests executed yet.")
        return

    summary_df = pd.DataFrame(TEST_RUN_HISTORY)

    print("\n" + "=" * 70)
    print("CUMULATIVE TEST SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nKEY METRICS")
    print("-" * 40)
    print(f"Total Runs:             {len(summary_df)}")
    print(f"Escalation Rate:        {summary_df['escalated'].mean():.1%}")
    print(f"Avg Triage Confidence:  {summary_df['triage_conf'].mean():.3f}")
    print(f"Avg Reasoning Steps:    {summary_df['reasoning_steps'].mean():.1f}")
    print(f"Avg Retrieval Rounds:   {summary_df['retrieval_iterations'].mean():.1f}")


def main() -> None:
    for test_case_id in range(len(TEST_CASES)):
        TEST_RUN_HISTORY.append(run_test_case(TEST_CASES[test_case_id], app))

    show_test_summary()


if __name__ == "__main__":
    main()
