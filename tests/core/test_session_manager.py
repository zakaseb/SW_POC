import pytest
from unittest.mock import patch, MagicMock

# Modules to test
from core.session_manager import (
    initialize_session_state,
    reset_document_states,
    reset_file_uploader
)

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
def test_initialize_session_state_initial_call(
    mock_vector_store_class,
    mock_get_embedding,
    mock_embedding_instance,
    mock_logger_fixture # autouse=True, so available
):
    mock_get_embedding.return_value = mock_embedding_instance
    mock_vs_instance = MagicMock(name="MockVectorStoreInstance")
    mock_vector_store_class.return_value = mock_vs_instance

    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        # For a truly initial call, these attributes wouldn't exist.
        # MagicMock by default allows creating attributes on first access.
        # The 'key not in st.session_state' check translates to 'not hasattr(mock_session_state, key)'
        # or using a side_effect for __contains__ if we want to be very explicit.
        # For simplicity, we can assume a fresh MagicMock won't have these attributes.
        # To be absolutely certain for `key not in session_state`:
        def mock_contains(key):
            return False # Simulate no keys exist initially
        mock_session_state.__contains__.side_effect = mock_contains

        initialize_session_state()

        # Check attributes are set
        assert mock_session_state.DOCUMENT_VECTOR_DB == mock_vs_instance
        mock_vector_store_class.assert_called_once_with(mock_embedding_instance)

        assert mock_session_state.messages == []
        assert mock_session_state.document_processed is False
        assert mock_session_state.uploaded_file_key == 0
        assert mock_session_state.uploaded_filenames == []
        assert mock_session_state.raw_documents == []
        assert mock_session_state.document_summary is None
        assert mock_session_state.document_keywords is None
        assert mock_session_state.bm25_index is None
        assert mock_session_state.bm25_corpus_chunks == []

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_initialize_session_state_idempotency(
    mock_vector_store_class,
    mock_get_embedding,
    mock_embedding_instance,
    mock_logger_fixture
):
    mock_get_embedding.return_value = mock_embedding_instance

    initial_messages = [{"role": "user", "content": "Hello"}]
    initial_db_mock = MagicMock(name="OriginalDB")
    mock_raw_docs_list = [MagicMock(name="RawDoc")]
    mock_bm25_index_obj = MagicMock(name="OriginalBM25Index")
    mock_bm25_chunks_list = [MagicMock(name="OriginalBM25Chunk")]

    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        # Pre-populate the mock_session_state
        mock_session_state.DOCUMENT_VECTOR_DB = initial_db_mock
        mock_session_state.messages = initial_messages
        mock_session_state.document_processed = True
        mock_session_state.uploaded_file_key = 5
        mock_session_state.uploaded_filenames = ["file1.pdf"]
        mock_session_state.raw_documents = mock_raw_docs_list
        mock_session_state.document_summary = "A summary"
        mock_session_state.document_keywords = "keywords"
        mock_session_state.bm25_index = mock_bm25_index_obj
        mock_session_state.bm25_corpus_chunks = mock_bm25_chunks_list

        # Make __contains__ reflect that these attributes are now set
        def mock_contains(key):
            return hasattr(mock_session_state, key)
        mock_session_state.__contains__.side_effect = mock_contains

        initialize_session_state()  # Call again

        assert mock_session_state.DOCUMENT_VECTOR_DB == initial_db_mock
        assert mock_session_state.messages == initial_messages
        assert mock_session_state.document_processed is True
        assert mock_session_state.uploaded_file_key == 5
        assert mock_session_state.uploaded_filenames == ["file1.pdf"]
        assert mock_session_state.raw_documents == mock_raw_docs_list
        assert mock_session_state.document_summary == "A summary"
        assert mock_session_state.document_keywords == "keywords"
        assert mock_session_state.bm25_index == mock_bm25_index_obj
        assert mock_session_state.bm25_corpus_chunks == mock_bm25_chunks_list

        mock_get_embedding.assert_not_called()
        mock_vector_store_class.assert_not_called()

# --- Tests for reset_document_states ---

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_reset_document_states_clears_all_with_chat(
    mock_vector_store_class,
    mock_get_embedding,
    mock_embedding_instance,
    mock_logger_fixture
):
    mock_get_embedding.return_value = mock_embedding_instance
    new_mock_vs_instance = MagicMock(name="NewMockVectorStoreInstance")
    mock_vector_store_class.return_value = new_mock_vs_instance

    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        # Pre-populate with some old values
        mock_session_state.DOCUMENT_VECTOR_DB = MagicMock(name="OldDB")
        mock_session_state.messages = [{"role": "user", "content": "Old message"}]
        mock_session_state.document_processed = True
        mock_session_state.uploaded_filenames = ["old_file.txt"]
        mock_session_state.raw_documents = [MagicMock()]
        mock_session_state.document_summary = "Old summary"
        mock_session_state.document_keywords = "Old keywords"
        mock_session_state.bm25_index = MagicMock(name="OldBM25Index")
        mock_session_state.bm25_corpus_chunks = [MagicMock()]

        reset_document_states(clear_chat=True)

        assert mock_session_state.DOCUMENT_VECTOR_DB == new_mock_vs_instance
        mock_vector_store_class.assert_called_with(mock_embedding_instance)
        assert mock_session_state.document_processed is False
        assert mock_session_state.messages == []
        assert mock_session_state.uploaded_filenames == []
        assert mock_session_state.raw_documents == []
        assert mock_session_state.document_summary is None
        assert mock_session_state.document_keywords is None
        assert mock_session_state.bm25_index is None
        assert mock_session_state.bm25_corpus_chunks == []
        mock_logger_fixture.info.assert_called_with("Document states reset.")

@patch(GET_EMBEDDING_MODEL_PATH)
@patch(IN_MEMORY_VECTOR_STORE_PATH)
def test_reset_document_states_preserves_chat(
    mock_vector_store_class,
    mock_get_embedding,
    mock_embedding_instance,
    mock_logger_fixture
):
    mock_get_embedding.return_value = mock_embedding_instance
    new_mock_vs_instance = MagicMock(name="NewMockVectorStoreInstance")
    mock_vector_store_class.return_value = new_mock_vs_instance

    initial_messages = [{"role": "user", "content": "Existing message"}]

    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        mock_session_state.DOCUMENT_VECTOR_DB = MagicMock(name="OldDB")
        mock_session_state.messages = initial_messages.copy()
        mock_session_state.document_processed = True
        mock_session_state.uploaded_filenames = ["old_file.txt"]
        # ... other states can be pre-populated ...

        reset_document_states(clear_chat=False)

        assert mock_session_state.DOCUMENT_VECTOR_DB == new_mock_vs_instance
        assert mock_session_state.document_processed is False
        assert mock_session_state.messages == initial_messages  # Chat preserved
        assert mock_session_state.uploaded_filenames == []  # Other states reset
        mock_logger_fixture.info.assert_called_with("Document states reset.")

# --- Tests for reset_file_uploader ---

def test_reset_file_uploader(mock_logger_fixture):
    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        mock_session_state.uploaded_file_key = 10
        reset_file_uploader()
        assert mock_session_state.uploaded_file_key == 11

def test_reset_file_uploader_from_initial_state(mock_logger_fixture):
    with patch("core.session_manager.st.session_state", new_callable=MagicMock) as mock_session_state:
        # Simulate state after initialize_session_state would have run
        mock_session_state.uploaded_file_key = 0
        reset_file_uploader()
        assert mock_session_state.uploaded_file_key == 1
