import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from .model_loader import get_embedding_model
from .logger_config import get_logger
from .database import delete_session
import os
from .document_processing import load_document, chunk_documents, index_documents
from .config import CONTEXT_PDF_STORAGE_PATH

logger = get_logger(__name__)


def purge_persistent_memory():
    """
    Purges all persisted data for the current user from the database and resets
    the current session state completely.
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        logger.warning("Attempted to purge memory without a user_id in session.")
        st.error("Cannot purge memory: User not identified.")
        return

    # 1. Delete the session from the persistent database
    try:
        delete_session(user_id)
        logger.info(f"Successfully purged persistent session data for user_id: {user_id}.")
    except Exception as e:
        logger.exception(f"Failed to purge session for user_id: {user_id}. Error: {e}")
        st.error("Failed to purge persistent memory from the database.")
        return

    # 2. Reset the current in-memory session state completely

    # Reset document-related state
    reset_document_states(clear_chat=True)

    # Reset context-document-related state
    st.session_state.CONTEXT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    st.session_state.context_document_loaded = False
    st.session_state.processed_context_file_info = None
    st.session_state.context_chunks = []

    # Reset chat memory
    st.session_state.memory = []

    logger.info(f"In-memory session state completely reset for user_id: {user_id}.")


def initialize_session_state():
    """
    Initializes the session state variables if they don't exist.
    """
    if "CONTEXT_VECTOR_DB" not in st.session_state:
        st.session_state.CONTEXT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "DOCUMENT_VECTOR_DB" not in st.session_state:
        st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "context_document_loaded" not in st.session_state:
        st.session_state.context_document_loaded = False
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = 0
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
    if "raw_documents" not in st.session_state:
        st.session_state.raw_documents = []
    if "processed_files_info" not in st.session_state:
        st.session_state.processed_files_info = {}
    if "processed_context_file_info" not in st.session_state:
        st.session_state.processed_context_file_info = None
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
    # Re-initialize vector stores
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())

    # Reset file processing info
    st.session_state.document_processed = False
    st.session_state.processed_files_info = {}
    st.session_state.uploaded_filenames = []
    st.session_state.raw_documents = []

    # Reset generated content
    st.session_state.document_summary = None
    st.session_state.document_keywords = None
    st.session_state.generated_requirements = None
    st.session_state.excel_file_data = None

    # Reset chunks and search indices
    st.session_state.general_context_chunks = []
    st.session_state.requirements_chunks = []
    st.session_state.bm25_index = None
    st.session_state.bm25_corpus_chunks = []

    # Optionally clear chat history
    if clear_chat:
        st.session_state.messages = []

    logger.info("Document states reset.")


def reset_file_uploader():
    """Increments the key for the file uploader to reset it."""
    st.session_state.uploaded_file_key += 1
