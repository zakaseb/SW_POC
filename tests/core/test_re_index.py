import pytest
from unittest.mock import patch, MagicMock
from core.document_processing import re_index_documents_from_session
from langchain_core.documents import Document as LangchainDocument

@patch("core.document_processing.st")
@patch("core.document_processing.index_documents")
def test_re_index_documents_from_session_with_chunks(mock_index_documents, mock_st):
    """
    Tests if re_index_documents_from_session calls index_documents for both
    general context and requirements chunks when they exist in the session state.
    """
    # Arrange
    mock_st.session_state.get.return_value = True
    mock_st.session_state.general_context_chunks = [LangchainDocument(page_content="general chunk")]
    mock_st.session_state.requirements_chunks = [LangchainDocument(page_content="requirement chunk")]
    mock_st.session_state.CONTEXT_VECTOR_DB = MagicMock()
    mock_st.session_state.DOCUMENT_VECTOR_DB = MagicMock()

    # Act
    re_index_documents_from_session()

    # Assert
    assert mock_index_documents.call_count == 2
    mock_index_documents.assert_any_call(mock_st.session_state.general_context_chunks, vector_db=mock_st.session_state.CONTEXT_VECTOR_DB)
    mock_index_documents.assert_any_call(mock_st.session_state.requirements_chunks, vector_db=mock_st.session_state.DOCUMENT_VECTOR_DB)

@patch("core.document_processing.st")
@patch("core.document_processing.index_documents")
def test_re_index_documents_from_session_no_chunks(mock_index_documents, mock_st):
    """
    Tests that re_index_documents_from_session does not call index_documents
    if there are no chunks in the session state.
    """
    # Arrange
    mock_st.session_state.get.return_value = True
    mock_st.session_state.general_context_chunks = []
    mock_st.session_state.requirements_chunks = []

    # Act
    re_index_documents_from_session()

    # Assert
    mock_index_documents.assert_not_called()

@patch("core.document_processing.st")
@patch("core.document_processing.index_documents")
def test_re_index_documents_from_session_only_general_chunks(mock_index_documents, mock_st):
    """
    Tests if re_index_documents_from_session calls index_documents only for
    general context chunks when only they exist in the session state.
    """
    # Arrange
    mock_st.session_state.get.return_value = True
    mock_st.session_state.general_context_chunks = [LangchainDocument(page_content="general chunk")]
    mock_st.session_state.requirements_chunks = []
    mock_st.session_state.CONTEXT_VECTOR_DB = MagicMock()

    # Act
    re_index_documents_from_session()

    # Assert
    mock_index_documents.assert_called_once_with(mock_st.session_state.general_context_chunks, vector_db=mock_st.session_state.CONTEXT_VECTOR_DB)
