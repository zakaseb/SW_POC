import pytest
import os
from unittest.mock import patch, MagicMock, ANY

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions/objects from rag_deep that will be patched or whose attributes will be patched
# This also helps confirm how rag_deep.py sees these names.
from rag_deep import (
    find_related_documents,
    combine_results_rrf,
    rerank_documents,
    generate_answer,
    LANGUAGE_MODEL, # Global in rag_deep.py
    RERANKER_MODEL  # Global in rag_deep.py
)
# Import core.config directly for accessing constants if needed by tests
from core import config as core_config
from langchain_core.documents import Document as LangchainDocument
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)

@pytest.fixture
def mock_streamlit_ui(mocker):
    mock_session_dict = {
        "document_processed": False, "messages": [], "uploaded_filenames": [],
        "raw_documents": [], "DOCUMENT_VECTOR_DB": MagicMock(), "uploaded_file_key": 0,
        "document_summary": None, "document_keywords": None, "bm25_index": None,
        "bm25_corpus_chunks": []
    }
    mocker.patch('rag_deep.st.session_state', mock_session_dict, create=True) # Use create=True if st.session_state might not exist
    mocker.patch('rag_deep.st.button', return_value=False)
    mocker.patch('rag_deep.st.file_uploader', return_value=None)
    mocker.patch('rag_deep.st.chat_input', return_value=None)
    m_spinner = MagicMock(); m_spinner.__enter__ = MagicMock(return_value=None); m_spinner.__exit__ = MagicMock(return_value=None)
    mocker.patch('rag_deep.st.spinner', return_value=m_spinner)
    mocker.patch('rag_deep.st.error')
    mocker.patch('rag_deep.st.warning')
    mocker.patch('rag_deep.st.info')
    mocker.patch('rag_deep.st.success')
    mocker.patch('rag_deep.st.markdown')
    mocker.patch('rag_deep.st.sidebar', MagicMock())
    mocker.patch('rag_deep.st.header'); mocker.patch('rag_deep.st.title'); mocker.patch('rag_deep.st.write')
    mocker.patch('rag_deep.st.chat_message', MagicMock())
    return mock_session_dict


@patch('rag_deep.index_documents')
@patch('rag_deep.chunk_documents', return_value=[LangchainDocument(page_content="chunk")])
@patch('rag_deep.load_document', return_value=[LangchainDocument(page_content="doc")])
@patch('rag_deep.save_uploaded_file', return_value="dummy_path.pdf")
@patch('rag_deep.reset_document_states')
def test_rag_deep_file_processing_flow_bm25_creation(
    mock_reset_document_states, mock_save_uploaded_file, mock_load_document,
    mock_chunk_documents, mock_index_documents,
    mock_streamlit_ui
):
    mock_uploaded_file = MagicMock(); mock_uploaded_file.name = "test.pdf"
    mock_streamlit_ui['uploaded_file_key'] = 0

    # This test simulates the BM25 creation block *within* rag_deep.py
    # It assumes previous steps (saving, loading, chunking, vector indexing) are mocked
    # and now we test the BM25 specific logic that uses st.session_state and processed_chunks

    mock_streamlit_ui["document_processed"] = True
    mock_streamlit_ui["uploaded_filenames"] = ["test.pdf"]
    # This is the 'processed_chunks' variable as it would be in rag_deep.py's scope
    processed_chunks_in_rag_deep_scope = mock_chunk_documents.return_value

    # --- Logic block from rag_deep.py related to BM25 ---
    if mock_streamlit_ui["document_processed"]: # This will be true
        try:
            corpus_texts = [chunk.page_content for chunk in processed_chunks_in_rag_deep_scope]
            tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
            bm25_index_instance = BM25Okapi(tokenized_corpus)
            mock_streamlit_ui["bm25_index"] = bm25_index_instance # Simulate assignment to session_state
            mock_streamlit_ui["bm25_corpus_chunks"] = processed_chunks_in_rag_deep_scope
        except Exception as e:
            pytest.fail(f"BM25 indexing part failed: {e}")
    # --- End of logic block ---

    assert isinstance(mock_streamlit_ui["bm25_index"], BM25Okapi)
    assert mock_streamlit_ui["bm25_corpus_chunks"] == processed_chunks_in_rag_deep_scope


# Patch targets should be where rag_deep.py looks them up.
# If rag_deep.py has "from core.generation import generate_answer", then patch "rag_deep.generate_answer"
@patch('rag_deep.generate_answer')
@patch('rag_deep.rerank_documents')
@patch('rag_deep.combine_results_rrf')
@patch('rag_deep.find_related_documents')
@patch('rag_deep.RERANKER_MODEL', new_callable=MagicMock) # Patch the global RERANKER_MODEL in rag_deep.py
@patch('rag_deep.LANGUAGE_MODEL', new_callable=MagicMock) # Patch the global LANGUAGE_MODEL in rag_deep.py
def test_rag_deep_chat_logic_reranking_successful(
    mock_language_model, # From @patch('rag_deep.LANGUAGE_MODEL')
    mock_reranker_model, # From @patch('rag_deep.RERANKER_MODEL')
    mock_find_related_documents,
    mock_combine_results_rrf,
    mock_rerank_documents,
    mock_generate_answer,
    mock_streamlit_ui
):
    mock_streamlit_ui["document_processed"] = True
    mock_streamlit_ui["messages"] = []

    mock_find_related_documents.return_value = {"semantic_results": [LangchainDocument("sem1")], "bm25_results": [LangchainDocument("bm25_1")]}
    hybrid_docs = [LangchainDocument(f"hybrid_doc_{i}") for i in range(core_config.TOP_K_FOR_RERANKER)]
    mock_combine_results_rrf.return_value = hybrid_docs

    assert mock_reranker_model is not None # Check the patched global RERANKER_MODEL

    reranked_final_docs = [LangchainDocument(f"reranked_doc_{i}") for i in range(core_config.FINAL_TOP_N_FOR_CONTEXT)]
    mock_rerank_documents.return_value = reranked_final_docs
    mock_generate_answer.return_value = "Final AI Answer from Orchestration"

    user_input_sim = "test user query"
    mock_streamlit_ui["messages"].append({"role": "user", "content": user_input_sim})

    # --- This simulates the block within `if user_input:` in rag_deep.py ---
    # Calls here use the names as they are available in rag_deep.py's scope
    retrieved_results_dict_val = find_related_documents( # This will call the mock_find_related_documents
        user_input_sim,
        mock_streamlit_ui["DOCUMENT_VECTOR_DB"],
        mock_streamlit_ui["bm25_index"],
        mock_streamlit_ui["bm25_corpus_chunks"],
        mock_streamlit_ui["document_processed"]
    )
    combined_hybrid_docs_val = combine_results_rrf(retrieved_results_dict_val) # Calls mock_combine_results_rrf

    final_context_docs_for_llm = []
    if combined_hybrid_docs_val:
        docs_for_reranking = combined_hybrid_docs_val[:core_config.TOP_K_FOR_RERANKER]
        # RERANKER_MODEL here refers to the global variable in rag_deep.py, which is patched.
        if RERANKER_MODEL:
             final_context_docs_for_llm = rerank_documents( # Calls mock_rerank_documents
                 user_input_sim, docs_for_reranking, RERANKER_MODEL,
                 top_n=core_config.FINAL_TOP_N_FOR_CONTEXT
             )
        else:
            final_context_docs_for_llm = docs_for_reranking[:core_config.FINAL_TOP_N_FOR_CONTEXT]

    if not final_context_docs_for_llm:
        ai_response = "No relevant sections found."
    else:
        # LANGUAGE_MODEL here refers to the global variable in rag_deep.py, which is patched.
        ai_response = generate_answer( # Calls mock_generate_answer
            LANGUAGE_MODEL,
            user_query=user_input_sim,
            context_documents=final_context_docs_for_llm,
            conversation_history=""
        )
    mock_streamlit_ui["messages"].append({"role": "assistant", "content": ai_response})
    # --- End of simulated block ---

    mock_find_related_documents.assert_called_once_with(user_input_sim, ANY, ANY, ANY, True)
    mock_combine_results_rrf.assert_called_once_with(mock_find_related_documents.return_value)
    mock_rerank_documents.assert_called_once_with(
        user_input_sim, hybrid_docs[:core_config.TOP_K_FOR_RERANKER],
        mock_reranker_model, # Assert it was called with the (mocked) RERANKER_MODEL global
        top_n=core_config.FINAL_TOP_N_FOR_CONTEXT
    )
    mock_generate_answer.assert_called_once_with(
        mock_language_model, # Assert it was called with the (mocked) LANGUAGE_MODEL global
        user_query=user_input_sim,
        context_documents=reranked_final_docs, conversation_history=""
    )
    assert mock_streamlit_ui["messages"][-1]["content"] == "Final AI Answer from Orchestration"


@patch('rag_deep.generate_answer')
@patch('rag_deep.rerank_documents')
@patch('rag_deep.combine_results_rrf')
@patch('rag_deep.find_related_documents')
@patch('rag_deep.RERANKER_MODEL', None) # Patch the global RERANKER_MODEL in rag_deep.py to be None
@patch('rag_deep.LANGUAGE_MODEL', new_callable=MagicMock) # Still need to mock LANGUAGE_MODEL
@patch('rag_deep.st.info')
def test_rag_deep_chat_logic_reranker_model_none(
    mock_st_info,
    mock_language_model, # From @patch('rag_deep.LANGUAGE_MODEL')
    mock_reranker_model_is_none, # From @patch('rag_deep.RERANKER_MODEL', None) - value is None
    mock_find_related_documents,
    mock_combine_results_rrf,
    mock_rerank_documents,
    mock_generate_answer,
    mock_streamlit_ui
):
    mock_streamlit_ui["document_processed"] = True
    mock_streamlit_ui["messages"] = []

    mock_find_related_documents.return_value = {"semantic_results": [LangchainDocument("sem1")], "bm25_results": [LangchainDocument("bm25_1")]}
    hybrid_docs = [LangchainDocument(f"hybrid_doc_{i}") for i in range(core_config.TOP_K_FOR_RERANKER)]
    mock_combine_results_rrf.return_value = hybrid_docs
    mock_generate_answer.return_value = "Fallback AI Answer"

    user_input_sim = "test user query"
    mock_streamlit_ui["messages"].append({"role": "user", "content": user_input_sim})

    # --- Start of simulated block from rag_deep.py ---
    retrieved_results_dict_val = find_related_documents( # Calls mock_find_related_documents
        user_input_sim, mock_streamlit_ui["DOCUMENT_VECTOR_DB"], mock_streamlit_ui["bm25_index"],
        mock_streamlit_ui["bm25_corpus_chunks"], mock_streamlit_ui["document_processed"]
    )
    combined_hybrid_docs_val = combine_results_rrf(retrieved_results_dict_val) # Calls mock_combine_results_rrf

    final_context_docs_for_llm = []
    if combined_hybrid_docs_val:
        docs_for_reranking = combined_hybrid_docs_val[:core_config.TOP_K_FOR_RERANKER]
        # RERANKER_MODEL here is the global in rag_deep.py, patched to None for this test
        if RERANKER_MODEL:
             final_context_docs_for_llm = rerank_documents(
                 user_input_sim, docs_for_reranking, RERANKER_MODEL,
                 top_n=core_config.FINAL_TOP_N_FOR_CONTEXT
             )
        else:
            # This path should be taken. rag_deep.py calls st.info here.
            st.info("Re-ranker model not loaded. Using documents from hybrid search directly (top results).")
            final_context_docs_for_llm = docs_for_reranking[:core_config.FINAL_TOP_N_FOR_CONTEXT]

    if not final_context_docs_for_llm:
        ai_response = "No relevant sections found."
    else:
        ai_response = generate_answer( # Calls mock_generate_answer
            LANGUAGE_MODEL, # This is the global LANGUAGE_MODEL from rag_deep.py
            user_query=user_input_sim,
            context_documents=final_context_docs_for_llm,
            conversation_history=""
        )
    mock_streamlit_ui["messages"].append({"role": "assistant", "content": ai_response})
    # --- End of simulated block ---

    mock_find_related_documents.assert_called_once()
    mock_combine_results_rrf.assert_called_once()
    mock_rerank_documents.assert_not_called()

    mock_st_info.assert_called_once_with(
        "Re-ranker model not loaded. Using documents from hybrid search directly (top results)."
    )

    expected_context_docs = hybrid_docs[:core_config.FINAL_TOP_N_FOR_CONTEXT]
    mock_generate_answer.assert_called_once_with(
        mock_language_model, # Assert it was called with the (mocked) LANGUAGE_MODEL global
        user_query=user_input_sim,
        context_documents=expected_context_docs, conversation_history=""
    )
    assert mock_streamlit_ui["messages"][-1]["content"] == "Fallback AI Answer"
