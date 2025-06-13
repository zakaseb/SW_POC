import pytest
from unittest.mock import patch, MagicMock

# Import Streamlit and patch its session_state for testing
import streamlit as st

# Modules to test
from core.session_manager import (
    initialize_session_state,
    reset_document_states,
    reset_file_uploader
)

# To mock get_embedding_model call
import core.model_loader

# Path to the logger instance in session_manager.py
SESSION_MANAGER_LOGGER_PATH = 'core.session_manager.logger'
IN_MEMORY_VECTOR_STORE_PATH = 'core.session_manager.InMemoryVectorStore'
GET_EMBEDDING_MODEL_PATH = 'core.session_manager.get_embedding_model'


@pytest.fixture
def mock_embedding_instance():
    return MagicMock(name="MockEmbeddingModelInstance")

@pytest.fixture(autouse=True)
def mock_logger_fixture():
    """Automatically mock the logger for all tests in this file."""
    with patch(SESSION_MANAGER_LOGGER_PATH) as mock_log:
        yield mock_log

# --- Tests for initialize_session_state ---

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_initialize_session_state_initial_call(mock_vector_store_class, mock_get_embedding, mock_embedding_instance, mock_logger_fixture):
    mock_get_embedding.return_value = mock_embedding_instance
    mock_vs_instance = MagicMock(name="MockVectorStoreInstance")
    mock_vector_store_class.return_value = mock_vs_instance

    # Use a fresh dictionary for session_state in each test
    mock_session_state_dict = {}
    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        initialize_session_state()

        assert "DOCUMENT_VECTOR_DB" in st.session_state
        assert st.session_state.DOCUMENT_VECTOR_DB == mock_vs_instance
        mock_vector_store_class.assert_called_once_with(mock_embedding_instance)

        assert "messages" in st.session_state
        assert st.session_state.messages == []

        assert "document_processed" in st.session_state
        assert st.session_state.document_processed is False

        assert "uploaded_file_key" in st.session_state
        assert st.session_state.uploaded_file_key == 0

        assert "uploaded_filenames" in st.session_state
        assert st.session_state.uploaded_filenames == []

        assert "raw_documents" in st.session_state
        assert st.session_state.raw_documents == []

        assert "document_summary" in st.session_state
        assert st.session_state.document_summary is None

        assert "document_keywords" in st.session_state
        assert st.session_state.document_keywords is None

        assert "bm25_index" in st.session_state
        assert st.session_state.bm25_index is None

        assert "bm25_corpus_chunks" in st.session_state
        assert st.session_state.bm25_corpus_chunks == []

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_initialize_session_state_idempotency(mock_vector_store_class, mock_get_embedding, mock_embedding_instance, mock_logger_fixture):
    mock_get_embedding.return_value = mock_embedding_instance

    # Pre-populate session state
    initial_messages = [{"role": "user", "content": "Hello"}]
    initial_db = MagicMock(name="OriginalDB")

    mock_session_state_dict = {
        "DOCUMENT_VECTOR_DB": initial_db,
        "messages": initial_messages,
        "document_processed": True, # Non-default
        "uploaded_file_key": 5,     # Non-default
        "uploaded_filenames": ["file1.pdf"],
        "raw_documents": [MagicMock()],
        "document_summary": "A summary",
        "document_keywords": "keywords",
        "bm25_index": MagicMock(),
        "bm25_corpus_chunks": [MagicMock()]
    }

    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        initialize_session_state() # Call again

        # Assert that existing values were NOT overwritten
        assert st.session_state.DOCUMENT_VECTOR_DB == initial_db
        assert st.session_state.messages == initial_messages
        assert st.session_state.document_processed is True
        assert st.session_state.uploaded_file_key == 5
        assert st.session_state.uploaded_filenames == ["file1.pdf"]
        assert len(st.session_state.raw_documents) == 1
        assert st.session_state.document_summary == "A summary"
        # Ensure get_embedding_model and InMemoryVectorStore constructor were NOT called again
        mock_get_embedding.assert_not_called()
        mock_vector_store_class.assert_not_called()


# --- Tests for reset_document_states ---

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_reset_document_states_clears_all_with_chat(mock_vector_store_class, mock_get_embedding, mock_embedding_instance, mock_logger_fixture):
    mock_get_embedding.return_value = mock_embedding_instance
    new_mock_vs_instance = MagicMock(name="NewMockVectorStoreInstance")
    mock_vector_store_class.return_value = new_mock_vs_instance

    mock_session_state_dict = {
        "DOCUMENT_VECTOR_DB": MagicMock(name="OldDB"),
        "messages": [{"role": "user", "content": "Old message"}],
        "document_processed": True,
        "uploaded_filenames": ["old_file.txt"],
        "raw_documents": [MagicMock()],
        "document_summary": "Old summary",
        "document_keywords": "Old keywords",
        "bm25_index": MagicMock(name="OldBM25Index"),
        "bm25_corpus_chunks": [MagicMock()]
    }

    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        reset_document_states(clear_chat=True)

        assert st.session_state.DOCUMENT_VECTOR_DB == new_mock_vs_instance
        mock_vector_store_class.assert_called_with(mock_embedding_instance) # Called to create new DB
        assert st.session_state.document_processed is False
        assert st.session_state.messages == [] # Chat cleared
        assert st.session_state.uploaded_filenames == []
        assert st.session_state.raw_documents == []
        assert st.session_state.document_summary is None
        assert st.session_state.document_keywords is None
        assert st.session_state.bm25_index is None
        assert st.session_state.bm25_corpus_chunks == []
        mock_logger_fixture.info.assert_called_with("Document states reset.")


@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_reset_document_states_preserves_chat(mock_vector_store_class, mock_get_embedding, mock_embedding_instance, mock_logger_fixture):
    mock_get_embedding.return_value = mock_embedding_instance
    new_mock_vs_instance = MagicMock(name="NewMockVectorStoreInstance")
    mock_vector_store_class.return_value = new_mock_vs_instance

    initial_messages = [{"role": "user", "content": "Existing message"}]
    mock_session_state_dict = {
        "DOCUMENT_VECTOR_DB": MagicMock(name="OldDB"),
        "messages": initial_messages.copy(), # Ensure we compare against the original
        "document_processed": True,
        "uploaded_filenames": ["old_file.txt"],
        # ... other states ...
    }

    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        reset_document_states(clear_chat=False)

        assert st.session_state.DOCUMENT_VECTOR_DB == new_mock_vs_instance
        assert st.session_state.document_processed is False
        assert st.session_state.messages == initial_messages # Chat preserved
        assert st.session_state.uploaded_filenames == [] # Other states reset
        mock_logger_fixture.info.assert_called_with("Document states reset.")


# --- Tests for reset_file_uploader ---

def test_reset_file_uploader(mock_logger_fixture): # mock_logger_fixture is auto-used
    mock_session_state_dict = {"uploaded_file_key": 10}
    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        reset_file_uploader()
        assert st.session_state.uploaded_file_key == 11
    # No specific log for reset_file_uploader in the function itself, so not checking logger here.

def test_reset_file_uploader_initializes_key(mock_logger_fixture):
    mock_session_state_dict = {} # Key doesn't exist
    with patch('core.session_manager.st.session_state', mock_session_state_dict):
        # This scenario implies uploaded_file_key should be initialized by initialize_session_state first.
        # If reset_file_uploader were called before initialize_session_state, it would error.
        # Assuming initialize_session_state has run:
        st.session_state.uploaded_file_key = 0 # Simulate prior initialization
        reset_file_uploader()
        assert st.session_state.uploaded_file_key == 1
