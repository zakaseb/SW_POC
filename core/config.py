from __future__ import annotations
import os
from pathlib import Path
import json
from urllib.parse import urlparse, urlunparse


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
You are an expert system engineer specialized in requirement extraction. Your task is to analyze the provided text chunk and extract all the requirements CONTAINED IN THAT CHUNK.
For each requirement found in the text chunk, you must generate a single JSON object that follows the schema below.
If a chunk contains multiple requirements, generate a list of JSON objects.
If the chunk contains no requirements, return an empty list [].
Your response MUST be only the JSON data (a single object or a list of objects) and nothing else. Do not include any prefixes, suffixes, or explanations.

SCOPE: 
- The Text Chunk is the ONLY source of requirements. 
- VERIFICATION METHODS REFERENCE, REQUIREMENT TYPE REFERENCE, and GENERAL CONTEXT are READ-ONLY - use them only to fill fields and clarify terms, NEVER as a source of requirements (even if they contain 'shall'). 
- If the Text Chunk has no requirements, return [].

JSON Schema:
{{
  "Name": "string",
  "Description": "single EARS 'shall' statement, content drawn from the Text Chunk",
  "VerificationMethod": "Analysis | Inspection | Test | Demonstration (use VERIFICATION METHODS REFERENCE)",
  "Tags": ["TBD" if description has TBD/TBC; add "Updated" if rewritten/split per quality rules below],
  "RequirementType": "Functional Req. | Performance Req. | Interface Req. | Constraint Req. (use REQUIREMENT TYPE REFERENCE)",
  "DocumentRequirementID": "ID copied verbatim from the chunk if present (e.g. SS_TO_SR-SOI_SYS_REQ-1, SYS_REQ-12, REQ_014), else empty string - never invent one"
}}

QUALITY: Each requirement must be verifiable, autonomous, unambiguous, concise, and a single EARS 'shall' statement. If not, rewrite it to comply, tag 'Updated', and split one overloaded requirement into several if needed.

EARS SYNTAX (mandatory, keyword order Where -> While -> When/If-then -> "<system> shall <response>"):
- Ubiquitous: <system> shall <response>.  e.g. "The control system shall prevent engine overspeed."
- State-driven (WHILE): While <precondition>, <system> shall <response>.  e.g. "While the aircraft is in-flight, the control system shall maintain fuel flow above 5 lbs/sec."
- Event-driven (WHEN): When <trigger>, <system> shall <response>.  e.g. "When ignition is commanded, the control system shall switch on continuous ignition."
- Optional feature (WHERE): Where <feature included>, <system> shall <response>.  e.g. "Where overspeed protection is included, the control system shall test its availability prior to dispatch."
- Unwanted behaviour (IF/THEN): If <trigger>, then <system> shall <response>.  e.g. "If computed airspeed is unavailable, then the control system shall use modelled airspeed."
- Complex: combine the above, e.g. "While the aircraft is on the ground, when reverse thrust is commanded, the control system shall enable deployment of the thrust reverser."

Rules: name the specific system immediately before 'shall' (never vague 'the system'/'it'); active voice; exactly one 'shall'; at most one trigger; up to 3 preconditions (else split); one-or-more responses. If a requirement is a formula/graphic that can't be cleanly EARS-ified, write the closest 'shall' statement, keep the value, tag 'Updated'.

Examples of desired JSON objects:
{{
  "Name": "Torque Setpoint CAN Message",
  "Description": "When a new torque setpoint is computed, the control system shall transmit CAN message 0x3A2 containing the torque setpoint.",
  "VerificationMethod": "Test",
  "Tags": [],
  "RequirementType": "Interface Req.",
  "DocumentRequirementID": "SS_TO_SR-SOI_SYS_REQ-1"
}}
{{
  "Name": "ADC Voltage Scaling",
  "Description": "When an ADC sample is available, the control system shall scale the ADC value by TBD to obtain the measured voltage.",
  "VerificationMethod": "Analysis",
  "Tags": ["TBD"],
  "RequirementType": "Functional Req.",
  "DocumentRequirementID": ""
}}

Now, analyze the following Text Chunk and extract ONLY the requirements stated in it. Every Description you output MUST be a single EARS 'shall' statement drawn from this Text Chunk.

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
1.  **General Context**: Portions of a document that provide broad, high-level information. This includes introductions, overviews, and background information that help a reader understand the overall context of the document, but do not contain specific, detailed requirements.
2.  **Requirements**: Portions of a document that contain specific, detailed, and actionable requirements, specifications, or instructions. These are the granular details of a project, system, or process. A requirement should be a statement that can be verified or tested.

PRIORITY RULE (apply this first, it overrides everything else):
If the text chunk contains the word "shall" or "should" (in any casing, as a whole word), classify it as "Requirements" - even if the chunk also looks like introductory or overview text. Only when neither "shall" nor "should" appears should you weigh the chunk against the two category descriptions above.

I will provide below an example of both classes to get a better understanding of the required task. 
An example of the "General Context" class is:
"Introduction:
This document specifies the technical system requirements for the All-Terrain Military Hybrid Vehicle (ATMHV), translating user needs (as defined in the URS) into detailed, implementable, and verifiable system requirements. The A-Spec outlines functional, performance, interface, design, safety, and verification criteria to guide system development, integration, and testing.
System Overview:
The ATMHV is an 8-passenger hybrid-powered vehicle designed for military operations across challenging terrains including deserts, forests, swamps, and snow. The vehicle provides off-road mobility, tactical communication capabilities, protection against threats, and logistics support. It must operate reliably across extreme environmental conditions.
"
An example of the "Requirements" class is:
"       SS_TO_SR-SOI_SYS_REQ-1
Maximum Speed Limitation #1
The system shall support maximum speed limitation #1 as per mission profile and design objectives. 
Verification Method(s): Analysis
       SS_TO_SR-SOI_SYS_REQ-2
Environmental Operating Range #2
The system shall support environmental operating range #2 as per mission profile and design objectives. 
Verification Method(s): Inspection
       SS_TO_SR-SOI_SYS_REQ-3
Minimum Range Capability #3
The system shall support minimum range capability #3 as per mission profile and design objectives. 
Verification Method(s): Demonstration
       SS_TO_SR-SOI_SYS_REQ-4
Dual-Mode Powertrain #4
The system shall support dual-mode powertrain #4 as per mission profile and design objectives. 
Verification Method(s): Test
       SS_TO_SR-SOI_SYS_REQ-5
Dynamic Power Mode Switching #5
The system shall support dynamic power mode switching #5 as per mission profile and design objectives. 
Verification Method(s): Test
"
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

# ========== Hard defaults (no env required) ==========
# Models
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"
OLLAMA_LLM_NAME             = "mistral:7b"
RERANKER_MODEL_NAME         = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Local HF cache + offline guardrails
MODEL_CACHE_DIR          = Path("./models")
HF_LOCAL_FILES_ONLY      = True  # avoid network calls at runtime
TOKENIZER_MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"

# Storage
BASE_STORE               = Path("./document_store")
PDF_STORAGE_PATH         = str(BASE_STORE / "pdfs")
CONTEXT_PDF_STORAGE_PATH = str(BASE_STORE / "context_pdfs")
MEMORY_FILE_PATH         = str(BASE_STORE / "memory" / "context.json")
REQUIREMENTS_OUTPUT_PATH = str(BASE_STORE / "generated_requirements")

# Endpoints (local single-device defaults)
# OLLAMA_BASE_URL is what the API wrapper uses to reach Ollama.
# RAG_API_BASE is what the Streamlit app uses to reach your wrapper.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
RAG_API_BASE    = "http://127.0.0.1:8000"

# Wrapper switch + (optional) auth
USE_API_WRAPPER = True
RAG_API_KEY     = ""  # set token here if you add auth to api.py

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
