# core/model_loader.py  (LM Studio only)

import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .logger_config import get_logger
from .config import (
    # Keep your legacy names for convenience, but these should be LM Studio IDs
    OLLAMA_LLM_NAME,
    OLLAMA_EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    # LM Studio OpenAI-compatible server
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
)

logger = get_logger(__name__)

@st.cache_resource
def get_embedding_model():
    """Embeddings via LM Studio (OpenAI-compatible)."""
    logger.info(
        f"Loading Embedding Model via LM Studio: {OLLAMA_EMBEDDING_MODEL_NAME} @ {OPENAI_BASE_URL}"
    )
    try:
        return OpenAIEmbeddings(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            check_embedding_ctx_length=False,
        )
    except Exception as e:
        logger.exception(
            f"Failed to load Embedding Model '{OLLAMA_EMBEDDING_MODEL_NAME}'. Details: {e}"
        )
        st.error(
            f"Failed to load Embedding Model '{OLLAMA_EMBEDDING_MODEL_NAME}'. "
            "Check LM Studio (Developer → Local Server) and that the model is started."
        )
        return None

@st.cache_resource
def get_language_model():
    """Chat LLM via LM Studio (OpenAI-compatible)."""
    logger.info(
        f"Loading LLM via LM Studio: {OLLAMA_LLM_NAME} @ {OPENAI_BASE_URL}"
    )
    try:
        return ChatOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=OLLAMA_LLM_NAME,
            temperature=0.2,
        )
    except Exception as e:
        logger.exception(
            f"Failed to load Language Model '{OLLAMA_LLM_NAME}'. Details: {e}"
        )
        st.error(
            f"Failed to load Language Model '{OLLAMA_LLM_NAME}'. "
            "Check LM Studio (Developer → Local Server) and that the model is started."
        )
        return None


@st.cache_resource
def get_reranker_model():  # model_name parameter removed, uses RERANKER_MODEL_NAME from config
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
        logger.exception(
            f"{user_message} Details: {e}"
        )  # Use logger.exception to include stack trace
        st.error(f"{user_message} Check logs for details.")
        return None
