import streamlit as st
from .logger_config import get_logger

logger = get_logger(__name__)

def get_persistent_context(context_vector_db, general_context_chunks):
    """
    Retrieves all documents from the context vector store and general context chunks.
    """
    persistent_context = []

    # Retrieve all documents from the context vector store
    if context_vector_db:
        try:
            context_docs = context_vector_db.similarity_search("", k=10000)
            persistent_context.extend(context_docs)
            logger.info(f"Retrieved {len(context_docs)} documents from the context vector store.")
        except Exception as e:
            user_message = "An error occurred while retrieving documents from the context vector store."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    # Add the general context chunks
    if general_context_chunks:
        persistent_context.extend(general_context_chunks)
        logger.info(f"Added {len(general_context_chunks)} general context chunks to the persistent context.")

    return persistent_context

def get_requirements_chunks(document_vector_db):
    """
    Retrieves all requirements chunks. It prioritizes chunks stored in the session state
    and falls back to querying the vector store.
    """
    # Prioritize session state as the source of truth after a session is loaded.
    if "requirements_chunks" in st.session_state and st.session_state["requirements_chunks"]:
        logger.info(
            f"Returning {len(st.session_state['requirements_chunks'])} requirements chunks directly from session state."
        )
        return st.session_state["requirements_chunks"]

    # Fallback to querying the vector store if session state is empty.
    logger.warning(
        "Requirements chunks not found in session state. Attempting to retrieve from vector store."
    )
    requirements_chunks = []
    if document_vector_db:
        try:
            retrieved_chunks = document_vector_db.similarity_search("", k=10000)
            # Filter out potential empty/dummy chunks from an uninitialized vector store.
            for chunk in retrieved_chunks:
                if chunk.page_content:
                    requirements_chunks.append(chunk)
            logger.info(
                f"Retrieved {len(requirements_chunks)} non-empty documents from the document vector store."
            )
        except Exception as e:
            user_message = "An error occurred while retrieving documents from the document vector store."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    return requirements_chunks
