import pytest
from unittest.mock import patch, MagicMock, ANY

# Modules to test
from core.model_loader import (
    get_embedding_model,
    get_language_model,
    get_reranker_model,
)

# To allow modifying config values for tests
from core import config

# For mocking requests.exceptions.ConnectionError
import requests

# Paths for mocking external dependencies as they are seen by 'core.model_loader'
OLLAMA_EMBEDDINGS_PATH = "core.model_loader.OllamaEmbeddings"
OLLAMA_LLM_PATH = "core.model_loader.OllamaLLM"
CROSS_ENCODER_PATH = "core.model_loader.CrossEncoder"
STREAMLIT_ERROR_PATH = (
    "core.model_loader.st.error"  # st.error is used directly in model_loader
)
MODEL_LOADER_LOGGER_PATH = (
    "core.model_loader.logger"  # Path to the logger instance in model_loader.py
)


# Fixture to automatically clear caches for model loader functions after each test.
# This is important because @st.cache_resource memoizes results.
@pytest.fixture(autouse=True)
def clear_model_loader_caches():
    yield  # Run the test
    # Check if .clear_cache attribute exists before calling,
    # as it's added by @st.cache_resource at runtime.
    if hasattr(get_embedding_model, "clear_cache"):
        get_embedding_model.clear_cache()
    if hasattr(get_language_model, "clear_cache"):
        get_language_model.clear_cache()
    if hasattr(get_reranker_model, "clear_cache"):
        get_reranker_model.clear_cache()


# --- Test for get_embedding_model ---


@patch(OLLAMA_EMBEDDINGS_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_embedding_model_success(
    mock_logger, mock_st_error, mock_ollama_embeddings_class
):
    mock_embedding_instance = MagicMock()
    mock_ollama_embeddings_class.return_value = mock_embedding_instance

    with patch.object(
        config, "OLLAMA_EMBEDDING_MODEL_NAME", "test-embed-model"
    ), patch.object(config, "OLLAMA_BASE_URL", "http://test-host:11434"):

        if hasattr(get_embedding_model, "clear_cache"):
            get_embedding_model.clear_cache()
        model = get_embedding_model()

    mock_ollama_embeddings_class.assert_called_once_with(
        model="test-embed-model", base_url="http://test-host:11434"
    )
    assert model == mock_embedding_instance
    mock_st_error.assert_not_called()
    mock_logger.info.assert_any_call(
        "Attempting to load Embedding Model: test-embed-model from: http://test-host:11434"
    )
    mock_logger.info.assert_any_call(
        "Embedding Model test-embed-model loaded successfully."
    )


@patch(OLLAMA_EMBEDDINGS_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_embedding_model_connection_error(
    mock_logger, mock_st_error, mock_ollama_embeddings_class
):
    mock_ollama_embeddings_class.side_effect = requests.exceptions.ConnectionError(
        "Test connection error"
    )
    if hasattr(get_embedding_model, "clear_cache"):
        get_embedding_model.clear_cache()

    with patch.object(
        config, "OLLAMA_BASE_URL", "http://nonexistent:11434"
    ), patch.object(config, "OLLAMA_EMBEDDING_MODEL_NAME", "test-embed-model"):
        model = get_embedding_model()

    assert model is None
    mock_st_error.assert_called_once()
    args, _ = mock_st_error.call_args
    assert "Failed to connect to Ollama" in args[0]
    assert "http://nonexistent:11434" in args[0]
    mock_logger.error.assert_called_once_with(ANY)


@patch(OLLAMA_EMBEDDINGS_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_embedding_model_other_exception(
    mock_logger, mock_st_error, mock_ollama_embeddings_class
):
    mock_ollama_embeddings_class.side_effect = Exception("Some other error")
    if hasattr(get_embedding_model, "clear_cache"):
        get_embedding_model.clear_cache()

    with patch.object(config, "OLLAMA_EMBEDDING_MODEL_NAME", "test-embed-model"):
        model = get_embedding_model()

    assert model is None
    mock_st_error.assert_called_once()
    args, _ = mock_st_error.call_args
    assert (
        "An unexpected error occurred while loading the Embedding Model (test-embed-model)"
        in args[0]
    )
    mock_logger.exception.assert_called_once_with(ANY)


# --- Test for get_language_model ---


@patch(OLLAMA_LLM_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_language_model_success(mock_logger, mock_st_error, mock_ollama_llm_class):
    mock_llm_instance = MagicMock()
    mock_ollama_llm_class.return_value = mock_llm_instance
    if hasattr(get_language_model, "clear_cache"):
        get_language_model.clear_cache()

    with patch.object(config, "OLLAMA_LLM_NAME", "test-llm-model"), patch.object(
        config, "OLLAMA_BASE_URL", "http://test-llm-host:11434"
    ):

        model = get_language_model()

    mock_ollama_llm_class.assert_called_once_with(
        model="test-llm-model", base_url="http://test-llm-host:11434"
    )
    assert model == mock_llm_instance
    mock_st_error.assert_not_called()
    mock_logger.info.assert_any_call(
        "Attempting to load Language Model: test-llm-model from: http://test-llm-host:11434"
    )
    mock_logger.info.assert_any_call(
        "Language Model test-llm-model loaded successfully."
    )


@patch(OLLAMA_LLM_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_language_model_connection_error(
    mock_logger, mock_st_error, mock_ollama_llm_class
):
    mock_ollama_llm_class.side_effect = requests.exceptions.ConnectionError(
        "Test LLM connection error"
    )
    if hasattr(get_language_model, "clear_cache"):
        get_language_model.clear_cache()

    with patch.object(
        config, "OLLAMA_BASE_URL", "http://nonexistent-llm:11434"
    ), patch.object(config, "OLLAMA_LLM_NAME", "test-llm-model"):
        model = get_language_model()

    assert model is None
    mock_st_error.assert_called_once()
    args, _ = mock_st_error.call_args
    assert "Failed to connect to Ollama" in args[0]
    assert "http://nonexistent-llm:11434" in args[0]
    mock_logger.error.assert_called_once_with(ANY)


@patch(OLLAMA_LLM_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_language_model_other_exception(
    mock_logger, mock_st_error, mock_ollama_llm_class
):
    mock_ollama_llm_class.side_effect = Exception("Some other LLM error")
    if hasattr(get_language_model, "clear_cache"):
        get_language_model.clear_cache()

    with patch.object(config, "OLLAMA_LLM_NAME", "test-llm-model"):
        model = get_language_model()

    assert model is None
    mock_st_error.assert_called_once()
    args, _ = mock_st_error.call_args
    assert (
        "An unexpected error occurred while loading the Language Model (test-llm-model)"
        in args[0]
    )
    mock_logger.exception.assert_called_once_with(ANY)


# --- Test for get_reranker_model ---


@patch(CROSS_ENCODER_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_reranker_model_success(
    mock_logger, mock_st_error, mock_cross_encoder_class
):
    mock_reranker_instance = MagicMock()
    mock_cross_encoder_class.return_value = mock_reranker_instance
    if hasattr(get_reranker_model, "clear_cache"):
        get_reranker_model.clear_cache()

    with patch.object(config, "RERANKER_MODEL_NAME", "test-reranker-model"):
        model = get_reranker_model()

    mock_cross_encoder_class.assert_called_once_with("test-reranker-model")
    assert model == mock_reranker_instance
    mock_st_error.assert_not_called()
    mock_logger.info.assert_any_call(
        "Attempting to load CrossEncoder model: test-reranker-model"
    )
    mock_logger.info.assert_any_call(
        "CrossEncoder model test-reranker-model loaded successfully."
    )


@patch(CROSS_ENCODER_PATH)
@patch(STREAMLIT_ERROR_PATH)
@patch(MODEL_LOADER_LOGGER_PATH)
def test_get_reranker_model_exception(
    mock_logger, mock_st_error, mock_cross_encoder_class
):
    mock_cross_encoder_class.side_effect = Exception("Reranker load error")
    if hasattr(get_reranker_model, "clear_cache"):
        get_reranker_model.clear_cache()

    with patch.object(config, "RERANKER_MODEL_NAME", "failing-reranker-model"):
        model = get_reranker_model()

    assert model is None
    mock_st_error.assert_called_once()
    args, _ = mock_st_error.call_args
    assert "Error loading CrossEncoder model 'failing-reranker-model'" in args[0]
    mock_logger.exception.assert_called_once_with(ANY)
