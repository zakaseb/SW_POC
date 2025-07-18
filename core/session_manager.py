import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from .model_loader import get_embedding_model
from .logger_config import get_logger
import os
import json
from .config import MEMORY_FILE_PATH

logger = get_logger(__name__)


def load_persistent_memory():
    """Loads the persistent memory from the file into the session state."""
    if os.path.exists(MEMORY_FILE_PATH):
        with open(MEMORY_FILE_PATH, "r") as f:
            st.session_state.memory = json.load(f)
    else:
        st.session_state.memory = []


def save_persistent_memory():
    """Saves the session state's memory to the file."""
    os.makedirs(os.path.dirname(MEMORY_FILE_PATH), exist_ok=True)
    with open(MEMORY_FILE_PATH, "w") as f:
        json.dump(st.session_state.memory, f)


def purge_persistent_memory():
    """Purges the persistent memory file."""
    if os.path.exists(MEMORY_FILE_PATH):
        os.remove(MEMORY_FILE_PATH)
    st.session_state.memory = []


def initialize_session_state():
    """
    Initializes the session state variables if they don't exist.
    """
    if "DOCUMENT_VECTOR_DB" not in st.session_state:
        st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        load_persistent_memory()
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
