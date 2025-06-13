import pytest
import os
from unittest.mock import patch
import importlib

# Import the module to be tested by referencing its path relative to the project root
# This assumes that tests are run from the project root directory.
# For robust imports, especially if using a test runner that might change sys.path,
# consider adding the project root to sys.path or using relative imports if tests are part of the package.
# For now, let's assume `core` is in PYTHONPATH or the test runner handles it.
from core import config  # Initial import

# --- Tests for Environment Variable Configurations ---


@pytest.fixture(autouse=True)
def manage_os_environ():
    """Fixture to save and restore os.environ for each test."""
    original_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_environ)


def test_ollama_base_url_default():
    # Ensure env var is not set for this test by clearing it specifically for this context
    with patch.dict(
        os.environ, {}, clear=True
    ):  # Clears all, then use fixture to restore
        # Need to remove the specific key if it was set by the system/shell
        if "OLLAMA_BASE_URL" in os.environ:
            del os.environ["OLLAMA_BASE_URL"]
        importlib.reload(config)
        assert config.OLLAMA_BASE_URL == "http://localhost:11434"


def test_ollama_base_url_env_override():
    test_url = "http://customhost:12345"
    with patch.dict(os.environ, {"OLLAMA_BASE_URL": test_url}):
        importlib.reload(config)
        assert config.OLLAMA_BASE_URL == test_url


def test_ollama_embedding_model_name_default():
    with patch.dict(os.environ, {}, clear=True):
        if "OLLAMA_EMBEDDING_MODEL_NAME" in os.environ:
            del os.environ["OLLAMA_EMBEDDING_MODEL_NAME"]
        importlib.reload(config)
        assert config.OLLAMA_EMBEDDING_MODEL_NAME == "deepseek-r1:1.5b"


def test_ollama_embedding_model_name_env_override():
    test_model = "custom_embed_model"
    with patch.dict(os.environ, {"OLLAMA_EMBEDDING_MODEL_NAME": test_model}):
        importlib.reload(config)
        assert config.OLLAMA_EMBEDDING_MODEL_NAME == test_model


def test_ollama_llm_name_default():
    with patch.dict(os.environ, {}, clear=True):
        if "OLLAMA_LLM_NAME" in os.environ:
            del os.environ["OLLAMA_LLM_NAME"]
        importlib.reload(config)
        assert config.OLLAMA_LLM_NAME == "deepseek-r1:1.5b"


def test_ollama_llm_name_env_override():
    test_model = "custom_llm_name"
    with patch.dict(os.environ, {"OLLAMA_LLM_NAME": test_model}):
        importlib.reload(config)
        assert config.OLLAMA_LLM_NAME == test_model


def test_reranker_model_name_default():
    with patch.dict(os.environ, {}, clear=True):
        if "RERANKER_MODEL_NAME" in os.environ:
            del os.environ["RERANKER_MODEL_NAME"]
        importlib.reload(config)
        assert config.RERANKER_MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_reranker_model_name_env_override():
    test_model = "custom_reranker"
    with patch.dict(os.environ, {"RERANKER_MODEL_NAME": test_model}):
        importlib.reload(config)
        assert config.RERANKER_MODEL_NAME == test_model


def test_pdf_storage_path_default():
    with patch.dict(os.environ, {}, clear=True):
        if "PDF_STORAGE_PATH" in os.environ:
            del os.environ["PDF_STORAGE_PATH"]
        importlib.reload(config)
        assert config.PDF_STORAGE_PATH == "document_store/pdfs/"


def test_pdf_storage_path_env_override():
    test_path = "/mnt/custom_storage/"
    with patch.dict(os.environ, {"PDF_STORAGE_PATH": test_path}):
        importlib.reload(config)
        assert config.PDF_STORAGE_PATH == test_path


# --- Tests for Hardcoded Constants (Lower Priority) ---


def test_max_history_turns_value():
    importlib.reload(
        config
    )  # Ensure we have the module's current state if other tests modified it via env
    assert config.MAX_HISTORY_TURNS == 3


def test_k_semantic_value():
    importlib.reload(config)
    assert config.K_SEMANTIC == 5


def test_k_bm25_value():
    importlib.reload(config)
    assert config.K_BM25 == 5


def test_k_rrf_param_value():
    importlib.reload(config)
    assert config.K_RRF_PARAM == 60


def test_top_k_for_reranker_value():
    importlib.reload(config)
    assert config.TOP_K_FOR_RERANKER == 10


def test_final_top_n_for_context_value():
    importlib.reload(config)
    assert config.FINAL_TOP_N_FOR_CONTEXT == 3


def test_prompt_template_exists_and_type():
    importlib.reload(config)
    assert isinstance(config.PROMPT_TEMPLATE, str)
    assert "{user_query}" in config.PROMPT_TEMPLATE  # Basic check for content


def test_summarization_prompt_template_exists_and_type():
    importlib.reload(config)
    assert isinstance(config.SUMMARIZATION_PROMPT_TEMPLATE, str)
    assert "{document_text}" in config.SUMMARIZATION_PROMPT_TEMPLATE


def test_keyword_extraction_prompt_template_exists_and_type():
    importlib.reload(config)
    assert isinstance(config.KEYWORD_EXTRACTION_PROMPT_TEMPLATE, str)
    assert "{document_text}" in config.KEYWORD_EXTRACTION_PROMPT_TEMPLATE


# Note: LOG_LEVEL is tested in test_logger_config.py as it's directly used there.
# The pattern in config.py (os.getenv) is tested by the variables above.
