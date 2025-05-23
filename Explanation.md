This Streamlit application, **DocuMind-AI**, is designed to process various document formats (PDF, DOCX, TXT), extract text, index content in an in-memory vector store, and allow users to query the document using **natural language**. Additionally, it offers features like document summarization and keyword extraction. It utilizes **LangChain** and **Ollama** models (cached for performance) for embeddings, language generation, summarization, and keyword extraction.

## **Detailed Explanation**

### **1. App Styling with CSS**
```python
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    ...
    </style>
    """,
    unsafe_allow_html=True,
)
```
- Uses **custom CSS** to style the Streamlit UI.
- Sets a **dark theme** with a black background (`#0E1117`).
- Customizes **chat input, messages, file uploader**, and **headings** for a modern, professional look.

### **2. Global Configuration**
```python
MAX_HISTORY_TURNS = 3 # Number of recent user/assistant turn pairs to include in history
K_SEMANTIC = 5 # Number of results for semantic search
K_BM25 = 5     # Number of results for BM25 search
K_RRF_PARAM = 60 # Constant for Reciprocal Rank Fusion (RRF)
TOP_K_FOR_RERANKER = 10 # Number of docs from hybrid search to pass to reranker
FINAL_TOP_N_FOR_CONTEXT = 3 # Number of docs reranker should return for LLM context

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided document context and conversation history to answer the current query.
If the query is a follow-up question, use the conversation history to understand the context.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Conversation History (if any):
{conversation_history}

Document Context:
{document_context}

Current Query: {user_query}
Answer:
"""
```
- Defines a **prompt template** (`PROMPT_TEMPLATE`) that structures how the AI should respond. It now includes placeholders for `conversation_history`, `document_context`, and `user_query`.
- `MAX_HISTORY_TURNS` controls how many pairs of user/assistant messages are included in the history.

```python
PDF_STORAGE_PATH = "document_store/pdfs/" # This path is used for all uploaded file types.
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
```
- Sets a **directory** where uploaded documents will be stored.
- Ensures that the directory exists before saving files.

```python
# Cached functions to load models
@st.cache_resource
def get_embedding_model():
    # ... (OllamaEmbeddings setup)
    return OllamaEmbeddings(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)

@st.cache_resource
def get_language_model():
    # ... (OllamaLLM setup)
    return OllamaLLM(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)

@st.cache_resource
def get_reranker_model(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    # ... (CrossEncoder setup)
    return CrossEncoder(model_name)

EMBEDDING_MODEL = get_embedding_model()
LANGUAGE_MODEL = get_language_model()
RERANKER_MODEL = get_reranker_model() # RERANKER_MODEL_NAME is used by default
```
- Loads **DeepSeek** models for embeddings and language generation, and a **CrossEncoder model** (`ms-marco-MiniLM-L-6-v2`) for re-ranking. All models are loaded using `@st.cache_resource` for performance.
  - `OllamaEmbeddings` → Converts text into vector embeddings.
  - `OllamaLLM` → Generates responses, summaries, and keywords.
  - `CrossEncoder` → Re-ranks search results for improved relevance.

```python
# Session state variables like DOCUMENT_VECTOR_DB, messages, document_processed, 
# raw_documents, document_summary, document_keywords, bm25_index, bm25_corpus_chunks are initialized if not present.
if "DOCUMENT_VECTOR_DB" not in st.session_state:
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# ... (other session state initializations)
```
- Uses **Streamlit session state** extensively to store:
  - Vector database (`DOCUMENT_VECTOR_DB`), BM25 index (`bm25_index`), and related corpus chunks (`bm25_corpus_chunks`).
  - Chat messages (`messages`).
  - Status flags (`document_processed`, `uploaded_file_key`).
  - Loaded document content (`raw_documents`, `uploaded_filename`).
  - Generated summary (`document_summary`) and keywords (`document_keywords`).
  - This ensures data persistence and state management across user interactions and reruns.

### **3. Utility Functions**
Error handling in these functions has been significantly improved to provide more specific feedback to the user.

#### **3.1 Save Uploaded File**
```python
def save_uploaded_file(uploaded_file):
    # ... (error handling for IO and other exceptions)
```
- **Saves** an uploaded file (PDF, DOCX, TXT) to disk under `PDF_STORAGE_PATH`.
- **Handles errors** (e.g., I/O errors, permission issues) if saving fails.

#### **3.2 Load Document (Formerly Load PDF Document)**
```python
def load_document(file_path):
    # ... (logic to handle .pdf, .docx, .txt using PDFPlumberLoader, python-docx, and standard file reading)
    # ... (specific error handling for each file type, e.g., PDFSyntaxError, PackageNotFoundError, UnicodeDecodeError, MemoryError)
```
- **Loads** text from PDF, DOCX, or TXT files. The function name was changed from `load_pdf_documents`.
- Uses **PDFPlumberLoader** for PDFs.
- Uses **python-docx** (imported as `docx`) for DOCX files.
- Uses standard file reading for TXT files (UTF-8 encoding).
- Includes robust error handling for corrupted files, unsupported types, encoding issues, and large files causing memory errors.

#### **3.3 Split Documents into Chunks**
```python
def chunk_documents(raw_documents):
    # ... (handles empty raw_documents, improved warnings for no processable chunks)
    # ... (uses RecursiveCharacterTextSplitter)
```
- Uses **RecursiveCharacterTextSplitter** to break the document into chunks.
  - **Chunk Size:** 1000 characters.
  - **Overlap:** 200 characters → Helps retain context between chunks.
- **Prepares the text** for vector storage. Handles cases where no processable chunks are found.

#### **3.4 Index Documents in the Vector Store**
```python
def index_documents(document_chunks):
    # ... (handles empty document_chunks, improved error messages)
```
- **Adds** the processed text chunks to the **in-memory vector store** (`st.session_state.DOCUMENT_VECTOR_DB`) for semantic search.
- Alongside vector indexing, a **BM25 index** is also created from the same document chunks (`st.session_state.bm25_index` and `st.session_state.bm25_corpus_chunks` are populated). This allows for keyword-based search.

#### **3.5 Perform Hybrid Search (Retrieval)**
```python
def find_related_documents(query):
    # ... (performs semantic search using DOCUMENT_VECTOR_DB.similarity_search)
    # ... (performs BM25 search using bm25_index.get_scores if bm25_index is available)
```
- This function, formerly just "Perform Similarity Search," now executes a **hybrid search strategy**:
  - It performs a **semantic search** using the vector store (FAISS via `InMemoryVectorStore`) to find `K_SEMANTIC` relevant chunks based on contextual meaning.
  - It also performs a **keyword-based search** using the BM25 index to find `K_BM25` relevant chunks based on term frequency and inverse document frequency.
- Returns a dictionary containing lists of results from both search methods (`{"semantic_results": ..., "bm25_results": ...}`).

#### **3.6 Combine Search Results (Reciprocal Rank Fusion)**
```python
def combine_results_rrf(search_results_dict, k_param=K_RRF_PARAM):
    # ... (calculates RRF scores and sorts documents)
```
- This function takes the results from `find_related_documents`.
- It combines the document lists from semantic and BM25 searches using **Reciprocal Rank Fusion (RRF)**.
- RRF assigns a score to each document based on its rank in each result list (score = 1 / (k_param + rank)). Scores for the same document from different lists are summed.
- The final list of documents is de-duplicated and sorted by the combined RRF score.

#### **3.7 Re-rank Documents (CrossEncoder)**
```python
def rerank_documents(query: str, documents: list, model: CrossEncoder, top_n: int = 3):
    # ... (uses model.predict to get scores and re-sorts documents)
```
- This new function takes the user query and the list of documents (typically the output of `combine_results_rrf`).
- It uses the pre-loaded `CrossEncoder` model (`RERANKER_MODEL`) to predict a relevance score for each (query, document_content) pair.
- The documents are then re-sorted based on these new scores, and the top `FINAL_TOP_N_FOR_CONTEXT` documents are returned.
- This step aims to further refine the relevance of documents before they are used as context for the LLM.
- Includes fallbacks if the model is not loaded or if prediction fails.

#### **3.8 Generate an Answer**
```python
def generate_answer(user_query, context_documents, conversation_history=""):
    # ... (handles empty query/context, empty LLM response, improved error messages)
```
- **Constructs** context from retrieved document chunks.
- **Formats** the prompt using `PROMPT_TEMPLATE`, now including `conversation_history`, `document_context`, and `user_query`.
- **Generates a response** using the cached `LANGUAGE_MODEL`.
- The `conversation_history` parameter allows the LLM to consider recent turns for better follow-up question understanding.

#### **3.7 Generate Summary**
```python
SUMMARIZATION_PROMPT_TEMPLATE = """...""" # Defines how to ask the LLM for a summary
def generate_summary(full_document_text):
    # ... (handles empty document text, empty LLM response, improved error messages)
```
- Takes the full text of the document (from `st.session_state.raw_documents`).
- Uses `SUMMARIZATION_PROMPT_TEMPLATE` and the cached `LANGUAGE_MODEL` to produce a concise summary.
- The result is stored in `st.session_state.document_summary`.

#### **3.8 Generate Keywords**
```python
KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """...""" # Defines how to ask the LLM for keywords
def generate_keywords(full_document_text):
    # ... (handles empty document text, empty LLM response, improved error messages)
```
- Takes the full text of the document.
- Uses `KEYWORD_EXTRACTION_PROMPT_TEMPLATE` and the cached `LANGUAGE_MODEL` to extract key phrases.
- The result is stored in `st.session_state.document_keywords`.

### **4. User Interface**
The UI is built using Streamlit components.

#### **4.1 App Title and Description**
Standard title and markdown for introduction.

#### **4.2 Sidebar Controls**
```python
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History", key="clear_chat"): ...
    if st.button("Reset Document", key="reset_document"): ...
    if st.session_state.document_processed:
        if st.button("Summarize Document", key="summarize_doc_button"): ...
        if st.button("Extract Keywords", key="extract_keywords_button"): ...
    # Logic to display summary and keywords from session state
```
- **Sidebar**: Contains controls for managing the session and accessing document analysis features.
  - **Clear Chat History**: Clears `st.session_state.messages`.
  - **Reset Document**: Resets the vector store (using the cached embedding model), clears all document-related session state (`document_processed`, `raw_documents`, `document_summary`, `document_keywords`, `messages`), and increments `uploaded_file_key` to reset the file uploader.
  - **Summarize Document**: (Visible after document processing) Triggers `generate_summary` using `st.session_state.raw_documents` and displays the result.
  - **Extract Keywords**: (Visible after document processing) Triggers `generate_keywords` using `st.session_state.raw_documents` and displays the result.
- The summary and keywords are displayed in the sidebar using `st.sidebar.info` and `st.sidebar.text_area` respectively if they are successfully generated. Error messages from generation functions are typically shown via `st.error` by the functions themselves.

#### **4.3 File Upload Section**
```python
uploaded_file = st.file_uploader( # Renamed from uploaded_pdf
    "Upload Research Document (PDF, DOCX, TXT)", # Updated help text
    type=["pdf", "docx", "txt"], # Accepts new types
    # ...
)
```
- Allows users to **upload PDF, DOCX, or TXT documents**.
- Uses a key based on `st.session_state.uploaded_file_key` to allow programmatic reset of the uploader.

#### **4.4 Process Uploaded Document**
```python
if uploaded_file:
    # Logic to reset state if a new file is uploaded (vector store, session states for summary, keywords etc.)
    if not st.session_state.document_processed:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            saved_path = save_uploaded_file(uploaded_file)
            if saved_path:
                raw_docs = load_document(saved_path) # Uses the updated load_document
                if raw_docs:
                    st.session_state.raw_documents = raw_docs # Store for summarization/keywords
                    processed_chunks = chunk_documents(raw_docs)
                    if processed_chunks:
                        index_documents(processed_chunks)
                        # ... success/error messages reflecting improved error handling
```
- If a **document is uploaded**:
  1. **Resets state** if it's a new document or if "Reset Document" was used. This includes clearing the vector store and session state for summary, keywords, etc.
  2. **Saves** the file.
  3. **Loads** text from the document using the updated `load_document` function (handles errors).
  4. Stores the raw loaded documents in `st.session_state.raw_documents` for use by summarization and keyword extraction features.
  5. **Splits** the text into chunks.
  6. **Indexes** the text into the vector store.
  7. Provides comprehensive success or error messages at each step, leveraging the improved error handling.

#### **4.5 Chat Input and Interaction**
```python
if st.session_state.document_processed:
    user_input = st.chat_input(...)
    if user_input:
        # ... (logic to format recent st.session_state.messages into formatted_history)
        retrieved_results_dict = find_related_documents(user_input)
        hybrid_search_docs = combine_results_rrf(retrieved_results_dict)
        if hybrid_search_docs:
            docs_for_reranking = hybrid_search_docs[:TOP_K_FOR_RERANKER]
            if RERANKER_MODEL:
                final_context_docs = rerank_documents(user_input, docs_for_reranking, RERANKER_MODEL, top_n=FINAL_TOP_N_FOR_CONTEXT)
            else: # Fallback
                final_context_docs = docs_for_reranking[:FINAL_TOP_N_FOR_CONTEXT]
        else:
            final_context_docs = []
        # ... (generate_answer(context_documents=final_context_docs, ...))
        # ... (display user and assistant messages)
else:
    st.info("Please upload a PDF, DOCX, or TXT document to begin your session.") # Updated info message
```
- Chat input is available only if a document has been successfully processed.
- If a **user asks a question**:
  1. **Formats recent chat history**.
  2. Retrieves results from both semantic and BM25 search using `find_related_documents`.
  3. Combines these results using `combine_results_rrf` to get a ranked list of unique document chunks (`hybrid_search_docs`).
  4. The top `TOP_K_FOR_RERANKER` documents from this list are then passed to `rerank_documents`.
  5. `rerank_documents` uses the CrossEncoder model to re-score and sort these documents, returning the top `FINAL_TOP_N_FOR_CONTEXT`.
  6. This final, re-ranked list is used as context for `generate_answer`.
  7. Displays the user's question and the assistant's response.
- An initial informational message prompts the user to upload a document of any supported type.

## **Summary**
### **Key Features:**
- **Upload and process PDF, DOCX, and TXT documents.**
- **Efficient text extraction and chunking.**
- **Advanced Retrieval Pipeline**:
  - **Hybrid Search**: Combines semantic (vector) search with keyword-based (BM25) search.
  - **Reciprocal Rank Fusion (RRF)**: Merges results from hybrid search.
  - **Re-ranking**: Further refines relevance using a CrossEncoder model (`ms-marco-MiniLM-L-6-v2`).
- **Query documents using natural language, with conversation history for contextual follow-up questions.**
- **Generate concise document summaries.**
- **Extract relevant keywords.**
- **Chat interface for interaction.**
- **Controls for clearing chat and resetting documents in the sidebar.**
- **Improved error handling and user feedback throughout the application.**
- **Performance optimization with model caching (`@st.cache_resource` for Ollama, Embedding, and Re-ranker models).**

This app is a **powerful research assistant** that leverages **LangChain, Ollama (with DeepSeek models), Sentence Transformers, and Streamlit** to provide an **interactive, AI-powered document analysis** experience for multiple file formats.
The screenshots in this document may not reflect the latest UI with sidebar controls, but the textual descriptions are updated.
