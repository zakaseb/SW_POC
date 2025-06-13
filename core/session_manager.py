import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from .model_loader import get_embedding_model
from .logger_config import get_logger

logger = get_logger(__name__)

def initialize_session_state():
    """
    Initializes the session state variables if they don't exist.
    """
    if "DOCUMENT_VECTOR_DB" not in st.session_state:
        st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = 0
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
    if "raw_documents" not in st.session_state:
        st.session_state.raw_documents = []
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
    if "document_keywords" not in st.session_state:
        st.session_state.document_keywords = None
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "bm25_corpus_chunks" not in st.session_state:
        st.session_state.bm25_corpus_chunks = []

def reset_document_states(clear_chat=True):
    """
    Resets all document-related session state variables.
    Optionally clears chat history.
    """
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    st.session_state.document_processed = False
    if clear_chat:
        st.session_state.messages = []
    # uploaded_file_key is typically incremented where this is called, if needed for uploader reset
    st.session_state.uploaded_filenames = []
    st.session_state.raw_documents = []
    st.session_state.document_summary = None
    st.session_state.document_keywords = None
    st.session_state.bm25_index = None
    st.session_state.bm25_corpus_chunks = []
    logger.info("Document states reset.")

def reset_file_uploader():
    """Increments the key for the file uploader to reset it."""
    st.session_state.uploaded_file_key += 1
