import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from .config import K_SEMANTIC, K_BM25, K_RRF_PARAM
from .logger_config import get_logger

logger = get_logger(__name__)

def find_related_documents(query, document_vector_db, bm25_index, bm25_corpus_chunks, document_processed_flag):
    """
    Perform both semantic and BM25 search to find related document chunks.
    Returns a dictionary with 'semantic_results' and 'bm25_results'.
    """
    semantic_docs = []
    bm25_retrieved_chunks = []

    if not query or not query.strip():
        logger.warning("find_related_documents called with empty query.")
        st.warning("Search query is empty. Please enter a query to find related documents.")
        return {"semantic_results": semantic_docs, "bm25_results": bm25_retrieved_chunks}

    if not document_processed_flag:
        logger.warning("find_related_documents called but no document processed.")
        st.warning("No document has been processed yet. Please upload and process a document before searching.")
        return {"semantic_results": semantic_docs, "bm25_results": bm25_retrieved_chunks}

    # 1. Semantic Search (Vector Search)
    logger.debug(f"Performing semantic search for query: '{query[:50]}...'")
    try:
        semantic_docs = document_vector_db.similarity_search(query, k=K_SEMANTIC)
        logger.info(f"Semantic search found {len(semantic_docs)} results.")
    except Exception as e:
        user_message = "An error occurred during semantic search."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        semantic_docs = []

    # 2. BM25 Search
    if bm25_index and bm25_corpus_chunks:
        logger.debug(f"Performing BM25 search for query: '{query[:50]}...'")
        try:
            tokenized_query = query.lower().split(" ")
            all_doc_scores = bm25_index.get_scores(tokenized_query)

            num_bm25_chunks = len(bm25_corpus_chunks)
            num_docs_to_consider = min(K_BM25, num_bm25_chunks)

            top_n_indices = sorted(
                [i for i, score in enumerate(all_doc_scores) if score > 0],
                key=lambda i: all_doc_scores[i],
                reverse=True
            )[:num_docs_to_consider]

            bm25_retrieved_chunks = [bm25_corpus_chunks[i] for i in top_n_indices]
            logger.info(f"BM25 search found {len(bm25_retrieved_chunks)} results with positive scores.")
        except Exception as e:
            user_message = "An error occurred during BM25 search."
            logger.exception(f"{user_message} Details: {e}")
            st.error(f"{user_message} Check logs for details.")
            bm25_retrieved_chunks = []
    else:
        logger.info("BM25 index not available. Skipping BM25 search.")
        bm25_retrieved_chunks = []

    return {"semantic_results": semantic_docs, "bm25_results": bm25_retrieved_chunks}

def combine_results_rrf(search_results_dict):
    """
    Combines search results from different methods using Reciprocal Rank Fusion (RRF).
    """
    doc_to_score = {}
    doc_objects = {}

    semantic_results = search_results_dict.get("semantic_results", [])
    for i, doc in enumerate(semantic_results):
        doc_id = doc.page_content
        if doc_id not in doc_objects:
            doc_objects[doc_id] = doc
        score = 1.0 / (K_RRF_PARAM + i + 1)
        doc_to_score[doc_id] = doc_to_score.get(doc_id, 0) + score

    bm25_results = search_results_dict.get("bm25_results", [])
    for i, doc in enumerate(bm25_results):
        doc_id = doc.page_content
        if doc_id not in doc_objects:
            doc_objects[doc_id] = doc
        score = 1.0 / (K_RRF_PARAM + i + 1)
        doc_to_score[doc_id] = doc_to_score.get(doc_id, 0) + score

    sorted_doc_ids = sorted(doc_to_score.keys(), key=lambda x: doc_to_score[x], reverse=True)
    final_combined_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    logger.info(f"RRF combined {len(semantic_results)} semantic and {len(bm25_results)} BM25 results into {len(final_combined_docs)} unique docs.")
    return final_combined_docs

def rerank_documents(query: str, documents: list, model: CrossEncoder, top_n: int):
    """
    Re-ranks a list of documents based on their relevance to a query using a CrossEncoder model.
    """
    if not documents:
        logger.debug("rerank_documents called with no documents.")
        return []
    if model is None:
        logger.warning("Re-ranker model not available. Skipping re-ranking.")
        st.warning("Re-ranker model not available. Returning documents without re-ranking.") # User feedback fine
        return documents[:top_n]

    logger.debug(f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'")
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
        logger.info("Document re-ranking prediction successful.")
    except Exception as e:
        user_message = "Error during document re-ranking."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return documents[:top_n]

    scored_documents = list(zip(scores, documents))
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for score, doc in scored_documents[:top_n]]
    return reranked_docs
