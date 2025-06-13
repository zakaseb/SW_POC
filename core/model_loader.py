import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import CrossEncoder
from .config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL_NAME, OLLAMA_LLM_NAME, RERANKER_MODEL_NAME
from .logger_config import get_logger # Import the logger
import requests

logger = get_logger(__name__) # Initialize logger for this module

# Cached functions to load models
@st.cache_resource
def get_embedding_model():
    """Loads and caches the Ollama embedding model."""
    logger.info(f"Attempting to load Embedding Model: {OLLAMA_EMBEDDING_MODEL_NAME} from: {OLLAMA_BASE_URL}")
    try:
        model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        logger.info(f"Embedding Model {OLLAMA_EMBEDDING_MODEL_NAME} loaded successfully.")
        # Attempt a simple operation to check connectivity, if available and cheap.
        # For OllamaEmbeddings, actual connection might be deferred.
        # If not, error will be caught on first use in the main script.
        return model
    except requests.exceptions.ConnectionError as conn_err:
        user_message = f"Failed to connect to Ollama at {OLLAMA_BASE_URL} for Embedding Model. Please ensure Ollama is running and accessible."
        logger.error(f"{user_message} Details: {conn_err}")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = f"An unexpected error occurred while loading the Embedding Model ({OLLAMA_EMBEDDING_MODEL_NAME})."
        logger.exception(f"{user_message} Details: {e}") # Use logger.exception to include stack trace
        st.error(f"{user_message} Check logs for details.")
        return None

@st.cache_resource
def get_language_model():
    """Loads and caches the Ollama language model."""
    logger.info(f"Attempting to load Language Model: {OLLAMA_LLM_NAME} from: {OLLAMA_BASE_URL}")
    try:
        model = OllamaLLM(model=OLLAMA_LLM_NAME, base_url=OLLAMA_BASE_URL)
        logger.info(f"Language Model {OLLAMA_LLM_NAME} loaded successfully.")
        # Similar to embedding model, actual connection test might be deferred.
        return model
    except requests.exceptions.ConnectionError as conn_err:
        user_message = f"Failed to connect to Ollama at {OLLAMA_BASE_URL} for Language Model. Please ensure Ollama is running and accessible."
        logger.error(f"{user_message} Details: {conn_err}")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = f"An unexpected error occurred while loading the Language Model ({OLLAMA_LLM_NAME})."
        logger.exception(f"{user_message} Details: {e}") # Use logger.exception to include stack trace
        st.error(f"{user_message} Check logs for details.")
        return None

@st.cache_resource
def get_reranker_model(): # model_name parameter removed, uses RERANKER_MODEL_NAME from config
    """
    Loads and caches the CrossEncoder model for re-ranking.
    Uses RERANKER_MODEL_NAME from config.
    """
    logger.info(f"Attempting to load CrossEncoder model: {RERANKER_MODEL_NAME}")
    try:
        model = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info(f"CrossEncoder model {RERANKER_MODEL_NAME} loaded successfully.")
        return model
    except Exception as e:
        user_message = f"Error loading CrossEncoder model '{RERANKER_MODEL_NAME}'. Re-ranking will be disabled."
        logger.exception(f"{user_message} Details: {e}") # Use logger.exception to include stack trace
        st.error(f"{user_message} Check logs for details.")
        return None
