import streamlit as st
from .logger_config import get_logger

logger = get_logger(__name__)

def get_all_documents(document_vector_db, context_vector_db, general_context_chunks):
    """
    Retrieves all documents from the document and context vector stores.
    """
    all_docs = []

    # Retrieve all documents from the document vector store
    if document_vector_db:
        try:
            # The InMemoryVectorStore does not have a "get all" method, so we do a similarity search with a wildcard
            # and a high k value to retrieve all documents.
            all_docs.extend(document_vector_db.similarity_search("", k=10000))
            logger.info(f"Retrieved {len(all_docs)} documents from the document vector store.")
        except Exception as e:
            user_message = "An error occurred while retrieving documents from the document vector store."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    # Retrieve all documents from the context vector store
    if context_vector_db:
        try:
            context_docs = context_vector_db.similarity_search("", k=10000)
            all_docs.extend(context_docs)
            logger.info(f"Retrieved {len(context_docs)} documents from the context vector store.")
        except Exception as e:
            user_message = "An error occurred while retrieving documents from the context vector store."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")

    # Add the general context chunks
    if general_context_chunks:
        all_docs.extend(general_context_chunks)
        logger.info(f"Added {len(general_context_chunks)} general context chunks to the document list.")

    return all_docs
