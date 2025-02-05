This Streamlit application, **DocuMind-AI**, is designed to process PDF documents, extract text from them, index the content in an in-memory vector store, and allow users to query the document using **natural language**. It utilizes **LangChain** and **Ollama** models for embeddings and answering questions.

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
PDF_STORAGE_PATH = "document_store/pdfs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
```
- Sets a **directory** where uploaded PDFs will be stored.
- Ensures that the directory exists before saving files.

```python
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
```
- Loads **DeepSeek** models:
  - `OllamaEmbeddings` → Converts text into vector embeddings for semantic search.
  - `OllamaLLM` → Generates responses to user queries.

```python
if "DOCUMENT_VECTOR_DB" not in st.session_state:
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
```
- Uses **Streamlit session state** to store the vector database so that indexed documents persist across interactions.

### **3. Utility Functions**
#### **3.1 Save Uploaded File**
```python
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None
```
- **Saves** an uploaded PDF file to disk.
- **Handles errors** if saving fails.

#### **3.2 Load PDF Document**
```python
def load_pdf_documents(file_path):
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []
```
- Uses **PDFPlumberLoader** to extract text from the uploaded PDF.

#### **3.3 Split Documents into Chunks**
```python
def chunk_documents(raw_documents):
    try:
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        return text_processor.split_documents(raw_documents)
    except Exception as e:
        st.error(f"Error chunking documents: {e}")
        return []
```
- Uses **RecursiveCharacterTextSplitter** to break the document into chunks:
  - **Chunk Size:** 1000 characters.
  - **Overlap:** 200 characters → Helps retain context between chunks.
- **Prepares the text** for vector storage.

#### **3.4 Index Documents in the Vector Store**
```python
def index_documents(document_chunks):
    try:
        st.session_state.DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
```
- **Adds** the processed text chunks to the **in-memory vector store**.
- Allows efficient **semantic search** based on user queries.

#### **3.5 Perform Similarity Search**
```python
def find_related_documents(query):
    try:
        return st.session_state.DOCUMENT_VECTOR_DB.similarity_search(query)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return []
```
- Searches for **related document chunks** based on **semantic similarity**.
- Ensures the **most relevant sections** of the document are retrieved.

#### **3.6 Generate an Answer**
```python
def generate_answer(user_query, context_documents):
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I'm sorry, but I encountered an error while processing your request."
```
- **Constructs** the context from retrieved document chunks.
- **Formats** the final prompt using the defined `PROMPT_TEMPLATE`.
- **Generates a response** using `OllamaLLM`.

### **4. User Interface**
#### **4.1 App Title and Description**
```python
st.title("📘 DocuMind-AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")
```
- Sets up the app's **title and description**.

#### **4.2 File Upload Section**
```python
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False,
)
```
- Allows users to **upload a PDF document**.

#### **4.3 Process Uploaded PDF**
```python
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    if saved_path:
        with st.spinner("Processing document..."):
            raw_docs = load_pdf_documents(saved_path)
            if raw_docs:
                processed_chunks = chunk_documents(raw_docs)
                if processed_chunks:
                    index_documents(processed_chunks)
                    st.success("✅ Document processed successfully! Ask your questions below.")
                else:
                    st.error("No document chunks were created.")
            else:
                st.error("No documents were loaded from the PDF.")
```
- If a **PDF is uploaded**, the app:
  1. **Saves** the file.
  2. **Extracts** text from the PDF.
  3. **Splits** the text into chunks.
  4. **Indexes** the text into the vector store.

#### **4.4 Chat Input**
```python
user_input = st.chat_input("Enter your question about the document...")
```
- Accepts **user queries** about the document.

#### **4.5 Process User Query**
```python
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
```
- If a **user asks a question**:
  1. It **retrieves relevant document chunks**.
  2. It **generates an answer** using the LLM.
  3. The assistant **responds** in a chat message.

## **Summary**
### **Key Features:**
- **Upload and process PDFs**  
- **Extract and index text**  
- **Query the document using natural language**  
- **Retrieve relevant sections using vector search**  
- **Generate concise, fact-based responses**  

This app is a **powerful research assistant** that leverages **LangChain, Ollama, and Streamlit** to provide an **interactive, AI-powered document analysis** experience.
