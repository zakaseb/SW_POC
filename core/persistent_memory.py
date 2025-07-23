import os
import pickle
from typing import List
from langchain_core.documents import Document as LangchainDocument

CONTEXT_MEMORY_PATH = os.path.join(".persistent_memory", "context_memory.pkl")

os.makedirs(os.path.dirname(CONTEXT_MEMORY_PATH), exist_ok=True)

def save_context_memory(docs: List[LangchainDocument]) -> None:
    existing_docs = load_context_memory()
    combined = existing_docs + docs
    with open(CONTEXT_MEMORY_PATH, "wb") as f:
        pickle.dump(combined, f)

def load_context_memory() -> List[LangchainDocument]:
    if not os.path.exists(CONTEXT_MEMORY_PATH):
        return []
    with open(CONTEXT_MEMORY_PATH, "rb") as f:
        return pickle.load(f)

def purge_context_memory() -> None:
    if os.path.exists(CONTEXT_MEMORY_PATH):
        os.remove(CONTEXT_MEMORY_PATH)

# FILE: document_processing.py (modification)

# ADD AT THE TOP
from .persistent_memory import save_context_memory, load_context_memory, purge_context_memory

# MODIFY index_documents to support persistent context

def index_documents(document_chunks, is_context=False):
    if not document_chunks:
        logger.warning("index_documents called with no chunks to index.")
        st.warning("No document chunks available to index.")
        return

    logger.info(f"Indexing {len(document_chunks)} document chunks. Context={is_context}")

    try:
        if is_context:
            save_context_memory(document_chunks)
            logger.info("Context documents saved to persistent memory.")
        else:
            st.session_state.DOCUMENT_VECTOR_DB.add_documents(document_chunks)
            logger.info("Input document chunks indexed into vector store.")
        st.session_state.document_processed = True
    except Exception as e:
        user_message = "An error occurred while indexing document chunks."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        st.session_state.document_processed = False
