import pytest
from unittest.mock import patch, MagicMock, ANY

# Import LangchainDocument for creating test data
from langchain_core.documents import Document as LangchainDocument

# Import for type hinting if needed, though mocks replace actual instances
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# Modules to test
from core.search_pipeline import (
    find_related_documents,
    combine_results_rrf,
    rerank_documents,
)

# Import config to use/mock its values, and logger for mocking
from core import config

# Paths for mocking
STREAMLIT_ERROR_PATH = "core.search_pipeline.st.error"
STREAMLIT_WARNING_PATH = "core.search_pipeline.st.warning"
SEARCH_PIPELINE_LOGGER_PATH = "core.search_pipeline.logger"

# --- Fixtures ---


@pytest.fixture
def mock_doc_factory():  # Renamed to avoid conflict if mock_doc is used as a specific instance
    return lambda content, source="test_source": LangchainDocument(
        page_content=content, metadata={"source": source}
    )


@pytest.fixture
def mock_vector_db():
    db = MagicMock()
    db.similarity_search.return_value = []
    return db


@pytest.fixture
def mock_bm25_index():
    index = MagicMock(spec=BM25Okapi)  # Use spec for better mocking
    index.get_scores.return_value = []
    return index


@pytest.fixture
def mock_cross_encoder():  # Renamed for clarity
    model = MagicMock(spec=CrossEncoder)
    model.predict.return_value = []
    return model


@pytest.fixture(autouse=True)
def mock_logger_fixture():
    """Automatically mock the logger for all tests in this file."""
    with patch(SEARCH_PIPELINE_LOGGER_PATH) as mock_log:
        yield mock_log


# --- Tests for find_related_documents (incorporating tests from test_rag_deep.py) ---


def test_find_related_documents_empty_query(
    mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    with patch(STREAMLIT_WARNING_PATH) as mock_st_warning:
        results = find_related_documents("", mock_vector_db, mock_bm25_index, [], True)
    assert results == {"semantic_results": [], "bm25_results": []}
    mock_st_warning.assert_called_once_with(
        "Search query is empty. Please enter a query to find related documents."
    )
    mock_logger_fixture.warning.assert_called_with(
        "find_related_documents called with empty query."
    )


def test_find_related_documents_document_not_processed(
    mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    with patch(STREAMLIT_WARNING_PATH) as mock_st_warning:
        results = find_related_documents(
            "query", mock_vector_db, mock_bm25_index, [], False
        )
    assert results == {"semantic_results": [], "bm25_results": []}
    mock_st_warning.assert_called_once_with(
        "No document has been processed yet. Please upload and process a document before searching."
    )
    mock_logger_fixture.warning.assert_called_with(
        "find_related_documents called but no document processed."
    )


def test_find_related_documents_success_both_searches(
    mock_doc_factory, mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    semantic_result = [mock_doc_factory("semantic content")]
    bm25_result_doc = mock_doc_factory("bm25 content")

    mock_vector_db.similarity_search.return_value = semantic_result
    mock_bm25_index.get_scores.return_value = [0.5]
    bm25_corpus_chunks = [bm25_result_doc]

    with patch.object(config, "K_SEMANTIC", 1), patch.object(config, "K_BM25", 1):
        results = find_related_documents(
            "query", mock_vector_db, mock_bm25_index, bm25_corpus_chunks, True
        )

    mock_vector_db.similarity_search.assert_called_once_with("query", k=1)
    mock_bm25_index.get_scores.assert_called_once_with(
        ["query"]
    )  # Assuming query is tokenized to ["query"]
    assert results["semantic_results"] == semantic_result
    assert results["bm25_results"] == [bm25_result_doc]
    mock_logger_fixture.info.assert_any_call("Semantic search found 1 results.")
    mock_logger_fixture.info.assert_any_call(
        "BM25 search found 1 results with positive scores."
    )


def test_find_related_documents_only_semantic(
    mock_doc_factory, mock_vector_db, mock_logger_fixture
):
    semantic_doc1 = mock_doc_factory("semantic only")
    mock_vector_db.similarity_search.return_value = [semantic_doc1]

    # BM25 not available (bm25_index is None)
    results = find_related_documents("test query", mock_vector_db, None, [], True)
    assert len(results["semantic_results"]) == 1
    assert results["semantic_results"][0].page_content == "semantic only"
    assert len(results["bm25_results"]) == 0
    mock_logger_fixture.info.assert_any_call(
        "BM25 index not available. Skipping BM25 search."
    )


def test_find_related_documents_only_bm25(
    mock_doc_factory, mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    mock_vector_db.similarity_search.return_value = []  # No semantic results

    bm25_doc1 = mock_doc_factory("bm25 only")
    mock_bm25_index.get_scores.return_value = [0.9]
    bm25_corpus_chunks = [bm25_doc1]

    results = find_related_documents(
        "test query", mock_vector_db, mock_bm25_index, bm25_corpus_chunks, True
    )
    assert len(results["semantic_results"]) == 0
    assert len(results["bm25_results"]) == 1
    assert results["bm25_results"][0].page_content == "bm25 only"


def test_find_related_documents_no_results(
    mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    mock_vector_db.similarity_search.return_value = []
    mock_bm25_index.get_scores.return_value = []
    bm25_corpus_chunks = [mock_doc_factory("doc1"), mock_doc_factory("doc2")]

    results = find_related_documents(
        "query for no results",
        mock_vector_db,
        mock_bm25_index,
        bm25_corpus_chunks,
        True,
    )
    assert len(results["semantic_results"]) == 0
    assert len(results["bm25_results"]) == 0


def test_find_related_documents_semantic_error(
    mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    mock_vector_db.similarity_search.side_effect = Exception("Semantic DB error")
    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        results = find_related_documents(
            "query", mock_vector_db, mock_bm25_index, [], True
        )
    assert results["semantic_results"] == []
    mock_st_error.assert_called_once()
    mock_logger_fixture.exception.assert_called_once()


def test_find_related_documents_bm25_error(
    mock_vector_db, mock_bm25_index, mock_logger_fixture
):
    mock_bm25_index.get_scores.side_effect = Exception("BM25 index error")
    bm25_corpus_chunks = [MagicMock()]
    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        results = find_related_documents(
            "query", mock_vector_db, mock_bm25_index, bm25_corpus_chunks, True
        )
    assert results["bm25_results"] == []
    mock_st_error.assert_called_once()
    mock_logger_fixture.exception.assert_called_once()


# --- Tests for combine_results_rrf (incorporating tests from test_rag_deep.py) ---


def test_combine_results_rrf_no_overlap(mock_doc_factory, mock_logger_fixture):
    docS1 = mock_doc_factory("semantic doc 1")
    docS2 = mock_doc_factory("semantic doc 2")
    docB1 = mock_doc_factory("bm25 doc 1")
    docB2 = mock_doc_factory("bm25 doc 2")

    search_results = {
        "semantic_results": [docS1, docS2],
        "bm25_results": [docB1, docB2],
    }

    with patch.object(
        config, "K_RRF_PARAM", 60
    ):  # Use actual default or a testable one
        combined = combine_results_rrf(search_results)

    assert len(combined) == 4
    combined_contents = [doc.page_content for doc in combined]
    # Exact order for equal initial scores depends on stability and original list order.
    # Just check presence for this simple case.
    assert docS1.page_content in combined_contents
    assert docS2.page_content in combined_contents
    assert docB1.page_content in combined_contents
    assert docB2.page_content in combined_contents
    mock_logger_fixture.info.assert_called_once()


def test_combine_results_rrf_full_overlap(mock_doc_factory, mock_logger_fixture):
    doc1 = mock_doc_factory("doc content 1")
    doc2 = mock_doc_factory("doc content 2")

    search_results_deterministic = {
        "semantic_results": [doc1, doc2],
        "bm25_results": [doc1, doc2],
    }
    with patch.object(config, "K_RRF_PARAM", 60):
        combined_det = combine_results_rrf(search_results_deterministic)
    assert len(combined_det) == 2
    assert combined_det[0] == doc1
    assert combined_det[1] == doc2


def test_combine_results_rrf_partial_overlap(mock_doc_factory, mock_logger_fixture):
    doc1 = mock_doc_factory("doc1 common")
    docS2 = mock_doc_factory("docS2 semantic only")
    docB2 = mock_doc_factory("docB2 bm25 only")

    search_results = {"semantic_results": [doc1, docS2], "bm25_results": [doc1, docB2]}
    with patch.object(config, "K_RRF_PARAM", 1):  # For predictable scoring
        combined = combine_results_rrf(search_results)

    assert len(combined) == 3
    assert combined[0] == doc1
    assert docS2 in combined[1:]
    assert docB2 in combined[1:]


def test_combine_results_rrf_empty_lists(mock_doc_factory, mock_logger_fixture):
    docS1 = mock_doc_factory("semantic doc 1")
    docB1 = mock_doc_factory("bm25 doc 1")

    combined_both_empty = combine_results_rrf(
        {"semantic_results": [], "bm25_results": []}
    )
    assert len(combined_both_empty) == 0

    combined_sem_empty = combine_results_rrf(
        {"semantic_results": [], "bm25_results": [docB1]}
    )
    assert len(combined_sem_empty) == 1 and combined_sem_empty[0] == docB1

    combined_bm25_empty = combine_results_rrf(
        {"semantic_results": [docS1], "bm25_results": []}
    )
    assert len(combined_bm25_empty) == 1 and combined_bm25_empty[0] == docS1


def test_combine_results_rrf_varying_ranks_and_k(mock_doc_factory, mock_logger_fixture):
    docA, docB, docC, docD = (
        mock_doc_factory("A"),
        mock_doc_factory("B"),
        mock_doc_factory("C"),
        mock_doc_factory("D"),
    )
    search_results = {
        "semantic_results": [docA, docB, docC],
        "bm25_results": [docB, docC, docA, docD],
    }

    # Patch K_RRF_PARAM directly in the config module for this test
    with patch.object(config, "K_RRF_PARAM", 2):
        combined = combine_results_rrf(search_results)

    assert len(combined) == 4
    assert combined[0] == docB
    assert combined[1] == docA
    assert combined[2] == docC
    assert combined[3] == docD


# --- Tests for rerank_documents (incorporating tests from test_rag_deep.py) ---


@pytest.mark.parametrize(
    "scores, initial_contents, expected_order_indices, top_n_config_key",
    [
        (
            [0.1, 0.9, 0.5],
            ["doc1", "doc2", "doc3"],
            [1, 2, 0],
            "FINAL_TOP_N_FOR_CONTEXT",
        ),
        (
            [0.9, 0.5, 0.1],
            ["doc1", "doc2", "doc3"],
            [0, 1, 2],
            "FINAL_TOP_N_FOR_CONTEXT",
        ),
        ([0.1, 0.9, 0.5], ["doc1", "doc2", "doc3"], [1, 2], "FINAL_TOP_N_FOR_CONTEXT"),
    ],
)
def test_rerank_documents_predict_scenarios(
    mock_cross_encoder,
    mock_doc_factory,
    scores,
    initial_contents,
    expected_order_indices,
    top_n_config_key,
    mock_logger_fixture,
):
    mock_cross_encoder.predict.return_value = scores
    docs_to_rerank = [mock_doc_factory(content) for content in initial_contents]

    # Use a test-specific top_n, or ensure config.FINAL_TOP_N_FOR_CONTEXT is patched if used directly
    # The function signature uses top_n, so we pass it directly.
    # The original test_rag_deep used FINAL_TOP_N_FOR_CONTEXT, so we simulate that.
    top_n_value = (
        2 if len(expected_order_indices) == 2 else 3
    )  # Adjust based on expected output length

    reranked = rerank_documents(
        "test query", docs_to_rerank, mock_cross_encoder, top_n=top_n_value
    )

    assert len(reranked) == min(top_n_value, len(initial_contents))
    expected_reranked_contents = [
        initial_contents[i] for i in expected_order_indices[:top_n_value]
    ]
    assert [doc.page_content for doc in reranked] == expected_reranked_contents
    mock_cross_encoder.predict.assert_called_once()
    mock_logger_fixture.info.assert_any_call(
        "Document re-ranking prediction successful."
    )


def test_rerank_documents_model_none(mock_doc_factory, mock_logger_fixture):
    docs_to_rerank = [mock_doc_factory(f"doc{i}") for i in range(5)]
    with patch(STREAMLIT_WARNING_PATH) as mock_st_warning:
        reranked = rerank_documents("test query", docs_to_rerank, model=None, top_n=3)

    assert len(reranked) == 3
    assert [doc.page_content for doc in reranked] == ["doc0", "doc1", "doc2"]
    mock_st_warning.assert_called_once()
    mock_logger_fixture.warning.assert_called_with(
        "Re-ranker model not available. Skipping re-ranking."
    )


def test_rerank_documents_prediction_error(
    mock_cross_encoder, mock_doc_factory, mock_logger_fixture
):
    mock_cross_encoder.predict.side_effect = Exception("Prediction failed")
    docs_to_rerank = [mock_doc_factory(f"doc{i}") for i in range(5)]

    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        reranked = rerank_documents(
            "test query", docs_to_rerank, mock_cross_encoder, top_n=3
        )

    assert len(reranked) == 3
    assert [doc.page_content for doc in reranked] == ["doc0", "doc1", "doc2"]
    mock_st_error.assert_called_once()
    mock_logger_fixture.exception.assert_called_once()


def test_rerank_documents_empty_input(mock_cross_encoder, mock_logger_fixture):
    reranked = rerank_documents("test query", [], mock_cross_encoder, top_n=3)
    assert reranked == []
    mock_logger_fixture.debug.assert_called_with(
        "rerank_documents called with no documents."
    )
