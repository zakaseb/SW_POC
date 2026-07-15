from __future__ import annotations
import os
from pathlib import Path
import json
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
load_dotenv()

# Global Application Constants
MAX_HISTORY_TURNS = (
    3  # Number of recent user/assistant turn pairs to include in history
)
K_SEMANTIC = 20 # Number of results for semantic search
K_BM25 = 20 # Number of results for BM25 search
K_RRF_PARAM = 60  # Constant for Reciprocal Rank Fusion (RRF)
TOP_K_FOR_RERANKER = 40 #Number of docs from hybrid search to pass to reranker
FINAL_TOP_N_FOR_CONTEXT = 15  # Number of docs reranker should return for LLM context

# Prompt Templates
GENERAL_QA_PROMPT_TEMPLATE = """
You are an expert research assistant in Systems Engineering. Use the provided document context, conversation history, and persistent memory to answer the current query.
If the query is a follow-up question, use the conversation history to understand the context.
If unsure, state that you don't know. Be concise and factual.

Persistent Memory (if any):
{persistent_memory}

Conversation History (if any):
{conversation_history}

Document Context:
{document_context}

Current Query: {user_query}
Answer:
"""

REQUIREMENT_JSON_PROMPT_TEMPLATE = """
You are an expert system engineer specialized in requirement extraction.
Your task is to extract every sentence or statement in the Text Chunk that expresses a requirement, behavior, capability, constraint, limit, interface, configuration rule, expected action, or design decision.
For each requirement found in the text chunk, you must generate a single JSON object that follows the schema below.
If a chunk contains multiple requirements, generate a list of JSON objects.
If the chunk contains no requirements, return an empty list [].
Your response MUST be only the JSON data (a single object or a list of objects) and nothing else. Do not include any prefixes, suffixes, or explanations.

SCOPE:
- The Text Chunk is the ONLY source of requirements. VERIFICATION METHODS REFERENCE, REQUIREMENT TYPE REFERENCE, and GENERAL CONTEXT are read-only — use them only to fill fields, never as a requirement source.
- A source statement does NOT need "shall"/"should" — imperative instructions, parameter/limit specs, tables, and "is"/"are"/"will" statements all qualify. Rewrite each into EARS 'shall' format in Description, same as a native "shall" sentence.
- COMPLETENESS: Scan the full chunk end-to-end. Extract EVERY qualifying statement, including every row of a list/table, not just the first few. Re-check before finalizing that nothing near the end or in repeated rows was skipped. When unsure whether something qualifies, extract it.

JSON Schema:
{{
  "Name": "short descriptive title",
  "Description": "single EARS 'shall' statement, content drawn from the Text Chunk",
  "VerificationMethod": "one of [Analysis, Inspection, Test, Demonstration]. Use the VERIFICATION METHODS REFERENCE below to select the correct method.",
  "Tags": ["TBD" if description has TBD/TBC],
  "RequirementType": "one of [Functional Req., Performance Req., Interface Req., Constraint Req.]. Use the REQUIREMENT TYPE REFERENCE below to select the correct type.",
  "DocumentRequirementID": "ID copied verbatim if present (e.g. SYS_REQ-12), else empty string — never invent one"
}}

QUALITY & EARS SYNTAX: 
Each Description must be verifiable, unambiguous, active voice, exactly one 'shall', named specific system (never vague "the system"/"it"). Split overloaded requirements into several.
Pattern: [Where <feature>,] [While <state>,] [When <trigger> | If <trigger>, then] <system> shall <response>.
  e.g. "While the aircraft is in-flight, the control system shall maintain fuel flow above 5 lbs/sec."
  e.g. "When ignition is commanded, the control system shall switch on continuous ignition."
If a requirement resists clean EARS phrasing, write the closest 'shall' statement. Prioritize completeness over perfect phrasing.

Example JSON object:
{{
  "Name": "Torque Setpoint CAN Message",
  "Description": "When a new torque setpoint is computed, the control system shall transmit CAN message 0x3A2 containing the torque setpoint.",
  "VerificationMethod": "Test",
  "Tags": [],
  "RequirementType": "Interface Req.",
  "DocumentRequirementID": "SS_TO_SR-SOI_SYS_REQ-1"
}}

Now scan the full chunk end-to-end and extract EVERY qualifying statement.

Text Chunk:
{document_context}

VERIFICATION METHODS REFERENCE (read-only):
{verification_methods_context}

REQUIREMENT TYPE REFERENCE (read-only):
{requirement_type_context}

GENERAL CONTEXT (read-only, clarification only - never a requirement source):
{general_context}

JSON Output:
"""

SUMMARIZATION_PROMPT_TEMPLATE = """
You are an expert research assistant. Provide a concise summary of the following document.
Focus on the main points and key takeaways. The summary should be approximately 3-5 sentences long.

Document:
{document_text}

Summary:
"""

KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """
You are an expert research assistant. Analyze the following document and extract the top 5-10 most relevant keywords or key phrases.
Present them as a comma-separated list.

Document:
{document_text}

Keywords:
"""

CHUNK_CLASSIFICATION_PROMPT_TEMPLATE = """
You are an expert document analyst. Classify the following text chunk into one of two categories:
1.  **General Context**: Portions of a document that provide broad, high-level information. This includes introductions, overviews, and background information that help a reader understand the overall context of the document, but do not contain any requirements.
2.  **Requirements**: Portions of a document that contain specific requirements, specifications, or instructions. These are the technical details of a project, system, or process. A requirement should be a statement that can be verified or tested.

IMPORTANT: 
Requirements are not limited to statements containing "shall" or "should". 
Declarative statements using "is", "are", or "will" also qualify in addition to imperative instructions and parameter/limit specifications (including tables). 

Examples of "General Context" class:
"Introduction: This document specifies the technical system requirements for the All-Terrain Military Hybrid Vehicle (ATMHV), translating user needs (as defined in the URS) into detailed, implementable, and verifiable system requirements. The A-Spec outlines functional, performance, interface, design, safety, and verification criteria to guide system development, integration, and testing."

Examples of "Requirements" class:
"The interface will timeout and enter a fail-safe state if no message is received within 200 ms."
"The default baud rate for UART2 is 115200."
"The BOOT state is a transitional state under normal conditions. The system enters BOOT immediately after power-on reset and remains in BOOT until self-test completes."
"Power Supply Specifications
Input Voltage Range: 18V - 32V DC
Max Input Current: 15A
Operating Temperature: -40C to +71C
Overvoltage Protection Threshold: 36V"
"SS_TO_SR-SOI_SYS_REQ-1: The system shall support maximum speed limitation #1 as per mission profile and design objectives."
"SS_TO_SR-SOI_SYS_REQ-2: The system shall support environmental operating range #2 as per mission profile and design objectives."

QUALIFYING RULE (apply this to determine "Requirements", even without "shall"):
Classify a chunk as "Requirements" if it contains a statement that:
1. Names a specific system, component, module, state, or parameter, AND
2. States a concrete, verifiable fact, value, limit, condition, state, or behavior about that named item.

DISQUALIFYING RULE (apply this last, before finalizing your answer):
Classify as "General Context" if the chunk consists ONLY of requirement IDs paired with metadata ABOUT other requirements defined elsewhere — e.g. traceability links, justification references, type codes, or verification method names in a "field = value" / "ID, Attribute = value" format — with no actual specification, parameter value, state/behavior description, or instruction stated directly in the chunk itself.

Analyze the text chunk provided below and determine which category it belongs to.
Your response should be a single word: either "General Context" or "Requirements".

Text Chunk:
{chunk_text}

Classification:
"""

REQUIREMENT_TYPE_CONTEXT = """Assign exactly one of: Functional Req. | Performance Req. | Interface Req. | Constraint Req.

**Functional Req.** — Describes WHAT the system does: a computation, logic branch, state
transition, mapping, clamping, fault response, or conditional action.
  -> Keywords: compute, calculate, set, map, convert, clamp, detect, enable, disable, trigger
  -> Example: "The system shall scale the ADC value by 0.00488 to obtain voltage."

**Performance Req.** — Describes HOW WELL or HOW OFTEN: a timing bound, rate, frequency,
accuracy, latency, or throughput. Always contains a numeric value with a unit.
  -> Keywords: within N ms, at a rate of, every N ms, accuracy of, no more than, maximum
  -> Example: "The system shall execute the control loop within 1 ms."

**Interface Req.** — Describes interaction with an EXTERNAL ENTITY: CAN, SPI, I2C, UART,
GPIO, ADC, DMA, register, memory-mapped address, OS API, or another module.
  -> Keywords: transmit, receive, read from, write to, signal, frame, message, pin, register
  -> Example: "The system shall transmit CAN message 0x3A2 containing the torque setpoint."

**Constraint Req.** — Imposes a DESIGN RESTRICTION: a coding standard, architectural rule,
memory policy, or safety mandate — restricts the solution regardless of behavior.
  -> Keywords: shall comply with, shall not use, MISRA-C, ISO 26262, stack depth, watchdog policy
  -> Example: "The system shall not use dynamic memory allocation after initialization."

Decision rule: timing/accuracy -> Performance | external bus/signal -> Interface |
standard/policy -> Constraint | everything else -> Functional
"""

# ===========================================================================
# DROP-IN REPLACEMENT for core/config.py, from the "Hard defaults" block
# down to (and including) the RAG_API_KEY line -- roughly lines 180-208.
# Everything below that in the original file (config.local.json loading,
# _normalize_url, OLLAMA_URL, the mkdir loop) stays exactly as it is.
#
# Add these two lines to the imports at the TOP of config.py:
#     from dotenv import load_dotenv
#     load_dotenv()
# ===========================================================================

def _env_str(key, default):
    val = os.getenv(key)
    return val if val not in (None, "") else default

def _env_bool(key, default):
    val = os.getenv(key)
    if val in (None, ""):
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def _env_int(key, default):
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

# ========== Defaults (all overridable via .env) ==========
# Models
OLLAMA_EMBEDDING_MODEL_NAME = _env_str("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text:latest")
OLLAMA_LLM_NAME             = _env_str("OLLAMA_LLM_NAME", "qwen2.5:14b-instruct")
RERANKER_MODEL_NAME         = _env_str("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOKENIZER_MODEL_NAME        = _env_str("TOKENIZER_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_NUM_CTX              = _env_int("OLLAMA_NUM_CTX", 8192)

# Local HF cache + offline guardrails
MODEL_CACHE_DIR             = Path(_env_str("MODEL_CACHE_DIR", "./models"))
HF_LOCAL_FILES_ONLY         = _env_bool("HF_LOCAL_FILES_ONLY", True)

# Storage
BASE_STORE               = Path(_env_str("BASE_STORE", "./document_store"))
PDF_STORAGE_PATH         = str(BASE_STORE / "pdfs")
CONTEXT_PDF_STORAGE_PATH = str(BASE_STORE / "context_pdfs")
MEMORY_FILE_PATH         = str(BASE_STORE / "memory" / "context.json")
REQUIREMENTS_OUTPUT_PATH = str(BASE_STORE / "generated_requirements")

# Endpoints
# OLLAMA_BASE_URL is what the API wrapper uses to reach Ollama.
# RAG_API_BASE is what the Streamlit app uses to reach the wrapper.
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
RAG_API_BASE    = _env_str("RAG_API_BASE",    "http://127.0.0.1:8000")

# Wrapper switch + (optional) auth
USE_API_WRAPPER = _env_bool("USE_API_WRAPPER", True)
RAG_API_KEY     = _env_str("RAG_API_KEY", "")

_local_path = os.path.join(os.path.dirname(__file__), "config.local.json")
if os.path.exists(_local_path):
    try:
        _ov = json.load(open(_local_path, "r"))
        for k, v in _ov.items():
            if k in globals():
                globals()[k] = v
    except Exception:
        pass

# Recompute cache-dependent paths AFTER applying local overrides
def _as_path(val):
    return val if isinstance(val, Path) else Path(val)

MODEL_CACHE_DIR = _as_path(MODEL_CACHE_DIR)

def _repo_cache_dir(repo_id: str) -> Path:
    return MODEL_CACHE_DIR / repo_id.replace("/", "-")

RERANKER_LOCAL_PATH  = _repo_cache_dir(RERANKER_MODEL_NAME)
TOKENIZER_LOCAL_PATH = _repo_cache_dir(TOKENIZER_MODEL_NAME)

# ========== Utilities shared by API + App ==========
def _normalize_url(url: str, default_port: int) -> str:
    """Force IPv4 for 'localhost' and ensure a port is present."""
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or default_port
    if host == "localhost":
        host = "127.0.0.1"
    return urlunparse((u.scheme or "http", f"{host}:{port}", "", "", "", ""))

# Canonical, normalized URLs to use everywhere else
OLLAMA_URL = _normalize_url(OLLAMA_BASE_URL, 11434)  # your “ollama_url”
API_URL    = _normalize_url(RAG_API_BASE,    8000)

# Postman / debug proxy (set only when you want to capture traffic)
# export POSTMAN_PROXY=http://127.0.0.1:5559
POSTMAN_PROXY = os.getenv("POSTMAN_PROXY") 

# HF offline/caching env hints (harmless if already set)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if HF_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))

# --- Docling PDF pipeline (layout + OCR) artifacts ---
DOCLING_ARTIFACTS_PATH = MODEL_CACHE_DIR / "docling"
DOCLING_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
if HF_LOCAL_FILES_ONLY:
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(DOCLING_ARTIFACTS_PATH))

# Ensure directories exist
for p in (PDF_STORAGE_PATH, CONTEXT_PDF_STORAGE_PATH, Path(MEMORY_FILE_PATH).parent, REQUIREMENTS_OUTPUT_PATH, MODEL_CACHE_DIR):
    Path(p).mkdir(parents=True, exist_ok=True)
