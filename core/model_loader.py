import requests
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import CrossEncoder

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL_NAME,
    OLLAMA_LLM_NAME,
    RERANKER_MODEL_NAME,
    RERANKER_LOCAL_PATH,
    MODEL_CACHE_DIR,
    HF_LOCAL_FILES_ONLY,
)
from .logger_config import get_logger

logger = get_logger(__name__)


def _reranker_source() -> str:
    """Prefer a locally cached reranker checkpoint when available."""
    if RERANKER_LOCAL_PATH and RERANKER_LOCAL_PATH.exists():
        return str(RERANKER_LOCAL_PATH)
    return RERANKER_MODEL_NAME


def _ollama_up(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.ok
    except Exception as e:
        logger.error(f"Ollama health check failed at {base_url}: {e}")
        return False


@st.cache_resource
def get_embedding_model():
    """Loads and caches the Ollama embedding model (same API/shape as before)."""
    logger.info(
        f"Attempting to load Embedding Model: {OLLAMA_EMBEDDING_MODEL_NAME} from: {OLLAMA_BASE_URL}"
    )
    try:
        if not _ollama_up(OLLAMA_BASE_URL):
            msg = f"Ollama not reachable at {OLLAMA_BASE_URL} for embeddings."
            logger.error(msg)
            st.error(msg)
            return None

        # ---- probe: does this model actually support embeddings?
        probe = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBEDDING_MODEL_NAME, "prompt": "probe"},
            timeout=10,
        )
        if not probe.ok:
            warn = (
                f"Embedding endpoint not supported by model '{OLLAMA_EMBEDDING_MODEL_NAME}'. "
                f"Falling back to no-embeddings (BM25 + optional reranker)."
            )
            logger.warning(warn)
            st.warning(warn)
            return None

        model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL
        )
        logger.info(f"Embedding Model {OLLAMA_EMBEDDING_MODEL_NAME} loaded successfully.")
        return model

    except requests.exceptions.ConnectionError as conn_err:
        user_message = f"Failed to connect to Ollama at {OLLAMA_BASE_URL} for Embedding Model."
        logger.error(f"{user_message} Details: {conn_err}")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = (
            f"An unexpected error occurred while loading the Embedding Model "
            f"({OLLAMA_EMBEDDING_MODEL_NAME})."
        )
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return None


@st.cache_resource
def get_language_model():
    """Loads and caches the Ollama language model (same API/shape as before)."""
    logger.info(
        f"Attempting to load Language Model: {OLLAMA_LLM_NAME} from: {OLLAMA_BASE_URL}"
    )
    try:
        if not _ollama_up(OLLAMA_BASE_URL):
            msg = f"Failed to reach Ollama at {OLLAMA_BASE_URL} for Language Model."
            logger.error(msg)
            st.error(msg)
            return None

        # ---- Option B: allow same model name for LLM and embeddings, just warn
        if OLLAMA_LLM_NAME == OLLAMA_EMBEDDING_MODEL_NAME:
            msg = (
                f"Warning: LLM and Embedding model both set to '{OLLAMA_LLM_NAME}'. "
                f"Make sure this model supports BOTH /api/generate and /api/embeddings."
            )
            logger.warning(msg)
            # no st.error, no return — we proceed like old days

        model = OllamaLLM(model=OLLAMA_LLM_NAME, base_url=OLLAMA_BASE_URL)
        logger.info(f"Language Model {OLLAMA_LLM_NAME} loaded successfully.")
        return model

    except requests.exceptions.ConnectionError as conn_err:
        user_message = f"Failed to connect to Ollama at {OLLAMA_BASE_URL} for Language Model."
        logger.error(f"{user_message} Details: {conn_err}")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = (
            f"An unexpected error occurred while loading the Language Model "
            f"({OLLAMA_LLM_NAME})."
        )
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return None


@st.cache_resource
def get_reranker_model():
    """
    Loads and caches the CrossEncoder model for re-ranking.
    Uses RERANKER_MODEL_NAME from config.
    """
    logger.info(f"Attempting to load CrossEncoder model: {RERANKER_MODEL_NAME}")
    try:
        source = _reranker_source()
        model = CrossEncoder(
            source,
            local_files_only=HF_LOCAL_FILES_ONLY,
            cache_folder=str(MODEL_CACHE_DIR),
        )
        logger.info(f"CrossEncoder model loaded successfully from: {source}")
        return model
    except Exception as e:
        user_message = (
            f"Error loading CrossEncoder model '{_reranker_source()}'. Re-ranking will be disabled."
        )
        if HF_LOCAL_FILES_ONLY:
            user_message += " Ensure required Hugging Face assets are pre-downloaded via pre_download_model.py."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return None
