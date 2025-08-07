import os

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
PROMPT_TEMPLATE = """
You are an expert research assistant. Your task is to extract and present requirements from the provided document context.
Based on the document context, answer the user's query.
The answer must be based *only* on the information present in the document context. Do not add any information that is not explicitly stated in the context.
If the context does not contain the answer, state that the information is not available in the document.
Do not invent new requirements.

Persistent Memory (if any):
{persistent_memory}

Conversation History (if any):
{conversation_history}

Document Context:
{document_context}

Current Query: {user_query}
Answer:
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

# Model Names
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv(
    "OLLAMA_EMBEDDING_MODEL_NAME", "mistral:7b"
)
OLLAMA_LLM_NAME = os.getenv("OLLAMA_LLM_NAME", "mistral:7b")
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Paths and URLs
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "document_store/pdfs/")
CONTEXT_PDF_STORAGE_PATH = os.getenv(
    "CONTEXT_PDF_STORAGE_PATH", "document_store/context_pdfs/"
)
MEMORY_FILE_PATH = os.getenv("MEMORY_FILE_PATH", "document_store/memory/context.json")
# Fetch Ollama base URL from environment variable, with a default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


