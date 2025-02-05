import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ---------------------------------
# App Styling with CSS
# ---------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Global Configuration
# ---------------------------------
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = "document_store/pdfs/"

# Ensure the PDF storage directory exists
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Initialize models
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Initialize or retrieve the vector store from session state for persistence across interactions
if "DOCUMENT_VECTOR_DB" not in st.session_state:
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# ---------------------------------
# Utility Functions
# ---------------------------------
def save_uploaded_file(uploaded_file):
    """
    Save the uploaded PDF file to disk and return the file path.
    """
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def load_pdf_documents(file_path):
    """
    Load PDF documents using PDFPlumberLoader.
    """
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def chunk_documents(raw_documents):
    """
    Split raw documents into manageable chunks using RecursiveCharacterTextSplitter.
    """
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

def index_documents(document_chunks):
    """
    Add document chunks to the in-memory vector store.
    """
    try:
        st.session_state.DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    except Exception as e:
        st.error(f"Error indexing documents: {e}")

def find_related_documents(query):
    """
    Perform a similarity search in the vector store for the given query.
    """
    try:
        return st.session_state.DOCUMENT_VECTOR_DB.similarity_search(query)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return []

def generate_answer(user_query, context_documents):
    """
    Generate an answer based on the user query and context documents.
    """
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I'm sorry, but I encountered an error while processing your request."

# ---------------------------------
# User Interface
# ---------------------------------
st.title("📘 DocuMind-AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False,
)

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

    # Chat Input Section
    user_input = st.chat_input("Enter your question about the document...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
