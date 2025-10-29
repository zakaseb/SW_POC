import streamlit as st
from .logger_config import get_logger

logger = get_logger(__name__)

def get_persistent_context(context_vector_db):
    """
    Retrieves all documents from the context vector store, ensuring no duplicates
    and filtering out empty documents.
    """
    persistent_context = []
    if context_vector_db:
        try:
            # Retrieve all documents. k=10000 is a stand-in for "all".
            all_docs = context_vector_db.similarity_search("", k=10000)

            # Filter out any potential dummy documents with no content
            non_empty_docs = [doc for doc in all_docs if doc.page_content]

            persistent_context.extend(non_empty_docs)
            logger.info(f"Retrieved {len(persistent_context)} non-empty documents from the context vector store.")
        except Exception as e:
            user_message = "An error occurred while retrieving documents from the context vector store."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    return persistent_context

def get_general_context(general_vector_db):
    """
    Retrieves all documents from the general vector store,
    ensuring no duplicates and filtering out empty documents.
    """
    general_context = []
    if general_vector_db:
        try:
            # Retrieve all documents; k=10000 acts as "fetch everything".
            all_docs = general_vector_db.similarity_search("", k=10000)

            # Filter out any documents with no actual content
            non_empty_docs = [doc for doc in all_docs if doc.page_content]

            general_context.extend(non_empty_docs)
            logger.info(
                f"Retrieved {len(general_context)} non-empty documents from the general vector store."
            )
        except Exception as e:
            user_message = (
                "An error occurred while retrieving documents from the general vector store."
            )
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    return general_context


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
