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
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
```
- Defines a **prompt template** that structures how the AI should respond.

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
    # print("Loading Embedding Model...") # For debugging/verifying cache
    return OllamaEmbeddings(model="deepseek-r1:1.5b")

@st.cache_resource
def get_language_model():
    # print("Loading Language Model...") # For debugging/verifying cache
    return OllamaLLM(model="deepseek-r1:1.5b")

EMBEDDING_MODEL = get_embedding_model()
LANGUAGE_MODEL = get_language_model()
```
- Loads **DeepSeek** models using `@st.cache_resource` for performance. This means models are loaded once and reused across sessions/reruns.
  - `OllamaEmbeddings` → Converts text into vector embeddings for semantic search.
  - `OllamaLLM` → Generates responses to user queries, summaries, and keywords.

```python
# Session state variables like DOCUMENT_VECTOR_DB, messages, document_processed, 
# raw_documents, document_summary, document_keywords are initialized if not present.
if "DOCUMENT_VECTOR_DB" not in st.session_state:
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# ... (other session state initializations for messages, document_processed, raw_documents, etc.)
```
- Uses **Streamlit session state** extensively to store:
  - The vector database (`DOCUMENT_VECTOR_DB`).
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
- **Adds** the processed text chunks to the **in-memory vector store** (`st.session_state.DOCUMENT_VECTOR_DB`).
- Allows efficient **semantic search**.

#### **3.5 Perform Similarity Search**
```python
def find_related_documents(query):
    # ... (handles empty query, improved error messages)
```
- Searches for **related document chunks** based on **semantic similarity** to the user's query.

#### **3.6 Generate an Answer**
```python
def generate_answer(user_query, context_documents):
    # ... (handles empty query/context, empty LLM response, improved error messages)
```
- **Constructs** context from retrieved document chunks.
- **Formats** the prompt using `PROMPT_TEMPLATE`.
- **Generates a response** using the cached `LANGUAGE_MODEL`.

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
        # ... (append to messages, get relevant_docs, generate_answer)
        # ... (display user and assistant messages, with generate_answer handling its own errors/warnings)
else:
    st.info("Please upload a PDF, DOCX, or TXT document to begin your session.") # Updated info message
```
- Chat input is available only if a document has been successfully processed.
- If a **user asks a question**:
  1. Retrieves relevant document chunks using `find_related_documents`.
  2. Generates an answer using `generate_answer`.
  3. Displays the user's question and the assistant's response (which might be an answer or an error/warning string from `generate_answer`).
- An initial informational message prompts the user to upload a document of any supported type.

## **Summary**
### **Key Features:**
- **Upload and process PDF, DOCX, and TXT documents.**
- **Extract and index text efficiently.**
- **Query documents using natural language.**
- **Generate concise document summaries.**
- **Extract relevant keywords.**
- **Chat interface for interaction.**
- **Controls for clearing chat and resetting documents in the sidebar.**
- **Improved error handling and user feedback throughout the application.**
- **Performance optimization with model caching (`@st.cache_resource` for Ollama models).**

This app is a **powerful research assistant** that leverages **LangChain, Ollama (with DeepSeek models), and Streamlit** to provide an **interactive, AI-powered document analysis** experience for multiple file formats.
The screenshots in this document may not reflect the latest UI with sidebar controls, but the textual descriptions are updated.
