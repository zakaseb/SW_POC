# DocuMind-AI: Detailed Explanation

**DocuMind-AI** is an AI-powered document assistant that allows users to upload **one or more documents** (PDF, DOCX, TXT), process their combined content, and interactively ask questions. Built with Streamlit and LangChain, it leverages embeddings and language models to provide concise, factual answers, summaries, and keyword extractions based on the documents' context. The application's backend logic has been modularized into a `core` package for better structure and maintainability.

## **1. App Styling with CSS**
```python
# In rag_deep.py
st.markdown(
    """
    <style>
    .stApp { ... }
    /* ... other styles ... */
    </style>
    """,
    unsafe_allow_html=True,
)
```
- Uses **custom CSS** injected via `st.markdown` in the main `rag_deep.py` script to style the Streamlit UI.
- Sets a **dark theme** and customizes chat input, messages, file uploader, and headings.

## **2. Global Configuration (`core/config.py`)**
Global constants, prompt templates, model names, and configurable paths are now primarily located in `core/config.py`.

```python
# In core/config.py
import os

# Global Application Constants
MAX_HISTORY_TURNS = 3
K_SEMANTIC = 5
# ... other search parameters ...

# Model Names (configurable via environment variables)
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "deepseek-r1:1.5b")
OLLAMA_LLM_NAME = os.getenv("OLLAMA_LLM_NAME", "deepseek-r1:1.5b")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Paths and URLs (configurable via environment variables)
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "document_store/pdfs/")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Prompt Templates
PROMPT_TEMPLATE = """..."""
SUMMARIZATION_PROMPT_TEMPLATE = """..."""
KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """..."""

# Logging Configuration (via environment variable)
# LOG_LEVEL is read in core/logger_config.py using os.getenv("LOG_LEVEL", "INFO")
```
- **Constants**: Defines application behavior (e.g., `MAX_HISTORY_TURNS`, search parameters).
- **Model Names & Paths**: Key configurations like Ollama URLs, model names (`OLLAMA_EMBEDDING_MODEL_NAME`, `OLLAMA_LLM_NAME`, `RERANKER_MODEL_NAME`), and `PDF_STORAGE_PATH` are configurable via **environment variables** with sensible defaults. This allows flexibility for different deployment environments.
- **Prompt Templates**: Contains the structures for querying the LLM for answers, summaries, and keywords.
- The `LOG_LEVEL` environment variable is used by the logging setup (see Section 9).

## **3. Model Loading (`core/model_loader.py`)**
Model loading functions are centralized in `core/model_loader.py`. These functions use `@st.cache_resource` for performance, loading each model only once.

```python
# In core/model_loader.py
from .config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL_NAME, OLLAMA_LLM_NAME, RERANKER_MODEL_NAME
# ... other imports ...

@st.cache_resource
def get_embedding_model():
    logger.info(f"Attempting to load Embedding Model: {OLLAMA_EMBEDDING_MODEL_NAME} from: {OLLAMA_BASE_URL}")
    try:
        model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        # ... (error handling for connection & other exceptions) ...
        return model
    # ...

# Similar functions for get_language_model() and get_reranker_model()
```
- `get_embedding_model()`: Loads the Ollama model specified by `OLLAMA_EMBEDDING_MODEL_NAME` from `core/config.py` for text embeddings.
- `get_language_model()`: Loads the Ollama model specified by `OLLAMA_LLM_NAME` from `core/config.py` for text generation.
- `get_reranker_model()`: Loads the CrossEncoder model specified by `RERANKER_MODEL_NAME` from `core/config.py` for re-ranking search results.
- In `rag_deep.py`, these models are initialized at startup:
  ```python
  EMBEDDING_MODEL = get_embedding_model()
  LANGUAGE_MODEL = get_language_model()
  RERANKER_MODEL = get_reranker_model()
  ```
- Enhanced error handling is included, especially for Ollama connection issues. If core models (`EMBEDDING_MODEL`, `LANGUAGE_MODEL`) fail to load, `rag_deep.py` will display an error and halt.

## **4. Session State Management (`core/session_manager.py`)**
Session state initialization and reset logic are managed by functions in `core/session_manager.py`.

```python
# In core/session_manager.py
def initialize_session_state():
    if "DOCUMENT_VECTOR_DB" not in st.session_state:
        st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    # ... other initializations for messages, document_processed, uploaded_filenames, etc. ...

def reset_document_states(clear_chat=True):
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    # ... resets document_processed, uploaded_filenames, raw_documents, summary, keywords, bm25_index, etc. ...
    if clear_chat:
        st.session_state.messages = []
```
- `initialize_session_state()`: Called at the start of `rag_deep.py` to ensure all necessary session state variables (`DOCUMENT_VECTOR_DB`, `messages`, `uploaded_filenames`, `raw_documents`, etc.) are initialized if they don't exist.
- `reset_document_states()`: Encapsulates the logic to clear all document-related data (vector store, raw documents, processed flags, analysis results) and optionally the chat history. This is used by the "Reset" button and when new files are uploaded.
- `reset_file_uploader()`: Increments a key for `st.file_uploader` to ensure it resets.

## **5. Core Logic Modules**
The application's main functionalities are broken down into specific modules within the `core` package.

### **5.1 Document Processing (`core/document_processing.py`)**
Handles the lifecycle of documents from upload to indexing.
- **`save_uploaded_file(uploaded_file)`**: Saves an uploaded file to the path specified by `PDF_STORAGE_PATH` (from `core/config.py`).
- **`load_document(file_path)`**: Loads text from PDF (using `PDFPlumberLoader`), DOCX (using `python-docx`), or TXT files. Includes specific error handling for each format. Returns a list of Langchain `Document` objects.
- **`chunk_documents(raw_documents)`**: Takes a list of Langchain `Document` objects (potentially from multiple files) and splits them into smaller chunks using `RecursiveCharacterTextSplitter`.
- **`index_documents(document_chunks)`**: Adds the processed text chunks to the in-memory vector store (`st.session_state.DOCUMENT_VECTOR_DB`). It also sets `st.session_state.document_processed = True` upon success.

### **5.2 Search Pipeline (`core/search_pipeline.py`)**
Manages the multi-stage process of retrieving and ranking relevant document chunks.
- **`find_related_documents(...)`**: Performs a hybrid search:
    - **Semantic Search**: Uses the vector store (`st.session_state.DOCUMENT_VECTOR_DB`) for similarity search.
    - **BM25 Search**: Uses `st.session_state.bm25_index` for keyword-based search.
    - Returns a dictionary of results from both methods.
- **`combine_results_rrf(search_results_dict)`**: Merges results from semantic and BM25 searches using Reciprocal Rank Fusion (RRF), returning a de-duplicated, re-scored list of document chunks.
- **`rerank_documents(query, documents, model, top_n)`**: Uses the `CrossEncoder` model (`RERANKER_MODEL`) to further re-rank the combined search results based on relevance to the query.

### **5.3 Generation (`core/generation.py`)**
Responsible for generating text using the language model.
- **`generate_answer(language_model, user_query, context_documents, conversation_history)`**: Constructs a prompt from the user's query, retrieved context documents, and conversation history, then invokes the `LANGUAGE_MODEL` to generate an answer.
- **`generate_summary(language_model, full_document_text)`**: Takes the combined text from all uploaded documents and uses the `LANGUAGE_MODEL` with `SUMMARIZATION_PROMPT_TEMPLATE` (from `core/config.py`) to produce a summary.
- **`generate_keywords(language_model, full_document_text)`**: Similar to summary generation, but uses `KEYWORD_EXTRACTION_PROMPT_TEMPLATE` to extract keywords from the combined document text.

## **6. User Interface (`rag_deep.py`)**
The main `rag_deep.py` script orchestrates the UI and calls functions from the `core` modules.

#### **6.1 Sidebar Controls**
- **Clear Chat History**: Clears `st.session_state.messages`.
- **Reset All Documents & Chat**: Calls `reset_document_states(clear_chat=True)` and `reset_file_uploader()` from `core.session_manager`.
- **Summarize Uploaded Content**: (Visible after documents are processed)
    - Concatenates `page_content` from all documents in `st.session_state.raw_documents`.
    - Calls `generate_summary()` from `core.generation` with the combined text.
    - Displays the result.
- **Extract Keywords from Content**: (Visible after documents are processed)
    - Similar to summarization, calls `generate_keywords()` from `core.generation` with the combined text.

#### **6.2 File Upload and Processing**
```python
# In rag_deep.py
uploaded_files = st.file_uploader(
    "Upload Research Documents (PDF, DOCX, TXT)",
    accept_multiple_files=True, ...
)

if uploaded_files:
    current_uploaded_file_names = sorted([f.name for f in uploaded_files])
    if set(current_uploaded_file_names) != set(st.session_state.get('uploaded_filenames', [])):
        reset_document_states(clear_chat=True) # From core.session_manager
        # ...
        for uploaded_file_obj in uploaded_files:
            saved_path = save_uploaded_file(uploaded_file_obj) # From core.document_processing
            if saved_path:
                raw_docs_from_file = load_document(saved_path) # From core.document_processing
                # ... accumulate raw_docs_from_file ...

        # If documents were successfully loaded:
        st.session_state.raw_documents = all_raw_docs_for_session
        processed_chunks = chunk_documents(st.session_state.raw_documents) # From core.document_processing
        if processed_chunks:
            index_documents(processed_chunks) # From core.document_processing
            # ... setup BM25 index ...
```
- **Multi-Document Upload**: `st.file_uploader` is configured with `accept_multiple_files=True`.
- **Processing Loop**:
  1. If a new set of files is uploaded, `reset_document_states()` is called.
  2. Each uploaded file is processed individually:
     - Saved using `save_uploaded_file()`.
     - Loaded using `load_document()`. Text content from all successfully loaded files is aggregated.
  3. The aggregated raw documents are stored in `st.session_state.raw_documents`.
  4. This aggregated content is then chunked using `chunk_documents()`.
  5. The resulting chunks (from all documents) are indexed together using `index_documents()` for semantic search and for BM25 index creation.
- A list of successfully processed filenames is displayed in the UI.

#### **6.3 Chat Input and Interaction**
- If documents are processed, the chat input is displayed.
- When a user submits a query:
  1. Recent chat history is formatted.
  2. `find_related_documents()` (from `core.search_pipeline`) is called, passing relevant session state items like the vector DB and BM25 index.
  3. Results are combined with `combine_results_rrf()`.
  4. Top results are re-ranked with `rerank_documents()`.
  5. The final context documents are passed to `generate_answer()` (from `core.generation`).
  6. The response is displayed in the chat.

## **7. Multi-Document Handling**
The application is designed to handle multiple documents seamlessly:
- **Upload**: Users can upload several files at once.
- **Processing**: All uploaded documents are processed together; their text is extracted, chunked, and indexed into a unified vector store and BM25 index.
- **Search**: Queries search across the combined content of all processed documents.
- **Summarization & Keywords**: These features operate on the aggregated text content of all uploaded documents, providing a holistic view.
- **UI**: Clearly indicates when actions apply to "all documents" or "uploaded content."

## **8. Logging (`core/logger_config.py`)**
A new logging system has been implemented for improved diagnostics:
- **Setup**: `core/logger_config.py` contains `setup_logging()`, called once at application startup in `rag_deep.py`.
- **Configuration**:
    - Uses the standard Python `logging` module.
    - Creates a logger named "rag_app" (and child loggers like "rag_app.core.module_name").
    - Log level is configurable via the `LOG_LEVEL` environment variable (defaults to `INFO`).
    - Logs are formatted to include timestamp, level, logger name, and message, and are output to `sys.stdout`.
- **Usage**:
    - `print()` statements previously used for debugging have been replaced with `logger.info()`, `logger.debug()`, etc., in all modules.
    - `logger.error()` or `logger.exception()` are used to log detailed error information (including stack traces) when exceptions occur, often supplementing user-facing `st.error()` messages.

## **9. Error Handling**
- **Specific Error Catching**: Functions in `core.document_processing` catch specific errors related to file types (e.g., `PDFSyntaxError`, `docx.opc.exceptions.PackageNotFoundError`).
- **Model Loading**: `core.model_loader` now specifically catches `requests.exceptions.ConnectionError` for Ollama models and provides user-friendly `st.error` messages. The main `rag_deep.py` script checks if core models (embedding, language) loaded successfully and halts with an error if not.
- **Graceful Degradation**: If the reranker model fails to load, re-ranking is disabled, and the application continues with the results from the hybrid search.
- **User Feedback**: `st.error`, `st.warning`, and `st.info` are used to provide clear feedback to the user, while more detailed logs are captured in the backend.

## **Summary of Key Features (Updated)**
- **Upload and process multiple PDF, DOCX, and TXT documents simultaneously.**
- **Modular codebase with logic organized into a `core` package.**
- **Configuration via Environment Variables** for Ollama settings, model names, storage paths, and log levels.
- **Efficient text extraction and chunking of combined document content.**
- **Advanced Retrieval Pipeline**:
  - **Hybrid Search**: Semantic (vector) + keyword (BM25) search across all documents.
  - **Reciprocal Rank Fusion (RRF)**: Merges hybrid search results.
  - **Re-ranking**: Refines relevance using a CrossEncoder model.
- **Query combined document content using natural language, with conversation history.**
- **Generate summaries and keywords from the aggregated text of all documents.**
- **Structured Logging System** for improved diagnostics.
- **Enhanced Error Handling** with more specific messages and critical failure checks.
- **Chat interface for interaction.**
- **Controls for clearing chat and resetting all documents.**
- **Performance optimization with model caching.**

This refactored application provides a robust, configurable, and maintainable platform for AI-powered document analysis.
