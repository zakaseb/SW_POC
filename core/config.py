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
You are an expert research assistant. Use the provided document context, conversation history, and persistent memory to answer the current query.
If the query is a follow-up question, use the conversation history to understand the context.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

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

# Model Names
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv(
    "OLLAMA_EMBEDDING_MODEL_NAME", "llama3.3:latest"
)
OLLAMA_LLM_NAME = os.getenv("OLLAMA_LLM_NAME", "llama3.3:latest")
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Paths and URLs
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "document_store/pdfs/")
MEMORY_FILE_PATH = os.getenv("MEMORY_FILE_PATH", "document_store/memory/context.json")
# Fetch Ollama base URL from environment variable, with a default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


