import os
import streamlit as st
from rank_bm25 import BM25Okapi

# Configure logging first
from core.logger_config import setup_logging, get_logger
setup_logging() # Initialize logging system
logger = get_logger(__name__)


# Import from new core modules
from core.config import (
    MAX_HISTORY_TURNS, K_SEMANTIC, K_BM25, K_RRF_PARAM, TOP_K_FOR_RERANKER, FINAL_TOP_N_FOR_CONTEXT,
    PDF_STORAGE_PATH
)
from core.model_loader import get_embedding_model, get_language_model, get_reranker_model
from core.document_processing import (
    save_uploaded_file, load_document, chunk_documents, index_documents
)
from core.search_pipeline import (
    find_related_documents, combine_results_rrf, rerank_documents
)
from core.generation import (
    generate_answer, generate_summary, generate_keywords
)
from core.session_manager import initialize_session_state, reset_document_states, reset_file_uploader

# ---------------------------------
# App Styling with CSS
# ---------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Initialize Session State & Models
# ---------------------------------
logger.info("Initializing session state.")
initialize_session_state()

logger.info("Loading core models.")
EMBEDDING_MODEL = get_embedding_model()
LANGUAGE_MODEL = get_language_model()
RERANKER_MODEL = get_reranker_model()

if not EMBEDDING_MODEL or not LANGUAGE_MODEL:
    critical_error_message = "Core models (Embedding or Language Model) failed to load. Application cannot continue. Please check Ollama connection and settings."
    logger.critical(critical_error_message)
    st.error(critical_error_message)
    st.stop()

logger.info("Core models loaded successfully (or Reranker gracefully disabled).")

os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
logger.info(f"Ensured PDF storage directory exists: {PDF_STORAGE_PATH}")


# ---------------------------------
# User Interface
# ---------------------------------

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")

    if st.button("Reset All Documents & Chat", key="reset_doc_chat_button"):
        logger.info("Resetting all documents and chat.")
        reset_document_states(clear_chat=True)
        reset_file_uploader()
        st.success("All documents and chat reset. You can now upload new documents.")
        logger.info("All documents and chat successfully reset.")
        st.rerun()

    if st.session_state.document_processed:
        if st.button("Summarize Uploaded Content", key="summarize_content_button"):
            if st.session_state.raw_documents:
                with st.spinner("Generating summary for all documents... This might take a few moments."):
                    full_text = "\n\n".join([doc.page_content for doc in st.session_state.raw_documents])
                    if not full_text.strip():
                        st.sidebar.warning("Cannot generate summary: Combined content of documents is effectively empty.")
                        logger.warning("Summarization attempt on empty combined content.")
                        st.session_state.document_summary = None
                    else:
                        logger.info("Generating combined content summary.")
                        summary_text = generate_summary(LANGUAGE_MODEL, full_text)
                        st.session_state.document_summary = summary_text
                        if "Failed to generate summary" in (summary_text or ""):
                             logger.error(f"Summary generation failed: {summary_text}")
                        else:
                             logger.info("Summary generated successfully.")
            else:
                st.sidebar.error("Cannot generate summary: Document content not loaded or available.")
                logger.warning("Summarization attempt with no raw documents loaded.")
        
        if st.button("Extract Keywords from Content", key="extract_keywords_content_button"):
            if st.session_state.raw_documents:
                with st.spinner("Extracting keywords from all documents..."):
                    full_text = "\n\n".join([doc.page_content for doc in st.session_state.raw_documents])
                    if not full_text.strip():
                        st.sidebar.warning("Cannot extract keywords: Combined content of documents is effectively empty.")
                        logger.warning("Keyword extraction attempt on empty combined content.")
                        st.session_state.document_keywords = None
                    else:
                        logger.info("Extracting keywords from combined content.")
                        keywords_text = generate_keywords(LANGUAGE_MODEL, full_text)
                        st.session_state.document_keywords = keywords_text
                        if "Failed to extract keywords" in (keywords_text or ""):
                            logger.error(f"Keyword extraction failed: {keywords_text}")
                        else:
                            logger.info("Keywords extracted successfully.")
            else:
                st.sidebar.error("Cannot extract keywords: Document content not loaded or available.")
                logger.warning("Keyword extraction attempt with no raw documents loaded.")
    
    if st.session_state.document_summary and "Failed to generate summary" not in st.session_state.document_summary:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Combined Content Summary")
        st.sidebar.info(st.session_state.document_summary)

    if st.session_state.document_keywords and "Failed to extract keywords" not in st.session_state.document_keywords:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 Extracted Keywords (Combined)")
        st.sidebar.text_area("Keywords:", st.session_state.document_keywords, height=100, disabled=True)

st.title("📘 DocuMind-AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload Research Documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    help="Select one or more PDF, DOCX, or TXT documents for analysis. Processing will begin upon upload. Re-uploading or changing files will reset the session.",
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploaded_file_key}"
)

if uploaded_files:
    current_uploaded_file_names = sorted([f.name for f in uploaded_files])
    logger.info(f"Files uploaded: {current_uploaded_file_names}")

    if set(current_uploaded_file_names) != set(st.session_state.get('uploaded_filenames', [])):
        logger.info("New set of files detected. Resetting document states.")
        reset_document_states(clear_chat=True)

        all_raw_docs_for_session = []
        successfully_loaded_filenames = []

        for uploaded_file_obj in uploaded_files:
            filename = uploaded_file_obj.name
            logger.debug(f"Processing uploaded file: {filename}")
            with st.spinner(f"Processing '{filename}'... This may take a moment."):
                saved_path = save_uploaded_file(uploaded_file_obj)
                if saved_path:
                    logger.info(f"File '{filename}' saved to '{saved_path}'")
                    raw_docs_from_file = load_document(saved_path)
                    if raw_docs_from_file:
                        all_raw_docs_for_session.extend(raw_docs_from_file)
                        successfully_loaded_filenames.append(filename)
                        logger.info(f"Successfully loaded and parsed: {filename}")
                    else:
                        st.error(f"Could not load document from '{filename}'. It might be empty, corrupted, or an unsupported type.")
                        logger.error(f"Failed to load document from '{filename}'.")
                else:
                    st.error(f"Failed to save '{filename}'. It will be skipped.")
                    logger.error(f"Failed to save '{filename}'.")

        st.session_state.uploaded_filenames = successfully_loaded_filenames

        if all_raw_docs_for_session:
            st.session_state.raw_documents = all_raw_docs_for_session
            logger.info(f"Total {len(all_raw_docs_for_session)} raw documents collected from {len(successfully_loaded_filenames)} files.")

            with st.spinner(f"Chunking and indexing {len(st.session_state.uploaded_filenames)} document(s)..."):
                logger.debug("Starting document chunking.")
                processed_chunks = chunk_documents(st.session_state.raw_documents)
                if processed_chunks:
                    logger.info(f"{len(processed_chunks)} chunks created.")
                    logger.debug("Starting document indexing.")
                    index_documents(processed_chunks) # Modifies st.session_state

                    if st.session_state.document_processed:
                        logger.info("Vector indexing successful.")
                        try:
                            logger.debug("Starting BM25 indexing.")
                            corpus_texts = [chunk.page_content for chunk in processed_chunks]
                            tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
                            st.session_state.bm25_index = BM25Okapi(tokenized_corpus)
                            st.session_state.bm25_corpus_chunks = processed_chunks
                            display_filenames = ", ".join(st.session_state.uploaded_filenames)
                            logger.info(f"BM25 index created for documents: {display_filenames}")
                            st.success(f"✅ Documents ({display_filenames}) processed and indexed successfully!")
                        except Exception as e:
                            display_filenames = ", ".join(st.session_state.uploaded_filenames)
                            logger.exception(f"Failed to create BM25 index for documents ({display_filenames}).")
                            st.error(f"Failed to create BM25 index for documents ({display_filenames}). Vector indexing may still be active. Details: {e}")
                    else:
                        display_filenames = ", ".join(st.session_state.uploaded_filenames)
                        logger.error(f"Vector indexing failed for documents ({display_filenames}). BM25 indexing skipped.")
                        st.error(f"Documents ({display_filenames}) loaded but failed during vector indexing. BM25 indexing also skipped.")
                else:
                    logger.warning("No processable chunks generated from documents. Indexing skipped.")
                    st.warning("No processable content found after loading all uploaded documents. Indexing skipped.")
        elif not st.session_state.uploaded_filenames and uploaded_files:
             logger.warning("Files were uploaded, but none could be successfully processed.")
             st.warning("Although files were uploaded, none could be successfully processed. Please check file formats and content.")

if st.session_state.get('uploaded_filenames') and st.session_state.get('document_processed'):
    st.markdown("---")
    st.markdown(f"**Successfully processed document(s):**")
    for name in st.session_state.uploaded_filenames:
        st.markdown(f"- _{name}_")
    st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.write(message["content"])

if st.session_state.document_processed:
    if len(st.session_state.uploaded_filenames) > 1:
        chat_placeholder = f"Ask a question about the {len(st.session_state.uploaded_filenames)} loaded documents..."
    elif len(st.session_state.uploaded_filenames) == 1:
        chat_placeholder = f"Ask a question about '{st.session_state.uploaded_filenames[0]}'..."
    else:
        chat_placeholder = "Ask a question about the loaded document(s)..."

    user_input = st.chat_input(chat_placeholder)
    if user_input:
        logger.info(f"User query: {user_input}")
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                num_messages_to_take = MAX_HISTORY_TURNS * 2
                chat_log_for_prompt = st.session_state.messages[:-1]
                chat_log_for_prompt = chat_log_for_prompt[-num_messages_to_take:]
                history_lines = [f"{('User' if msg['role'] == 'user' else 'Assistant')}: {msg['content']}" for msg in chat_log_for_prompt]
                formatted_history = "\n".join(history_lines)
                logger.debug(f"Formatted history for prompt: {formatted_history}")

                retrieved_results_dict = find_related_documents(
                    user_input,
                    st.session_state.DOCUMENT_VECTOR_DB,
                    st.session_state.bm25_index,
                    st.session_state.bm25_corpus_chunks,
                    st.session_state.document_processed
                )
                logger.info(f"Retrieved {len(retrieved_results_dict.get('semantic_results',[]))} semantic and {len(retrieved_results_dict.get('bm25_results',[]))} BM25 results.")

                hybrid_search_docs = combine_results_rrf(retrieved_results_dict)
                logger.info(f"Combined RRF results: {len(hybrid_search_docs)} documents.")

                final_context_docs = []
                if hybrid_search_docs:
                    docs_for_reranking = hybrid_search_docs[:TOP_K_FOR_RERANKER]
                    if RERANKER_MODEL:
                        logger.debug(f"Re-ranking top {len(docs_for_reranking)} documents.")
                        with st.spinner(f"Re-ranking top {len(docs_for_reranking)} documents..."):
                            final_context_docs = rerank_documents(user_input, docs_for_reranking, RERANKER_MODEL, top_n=FINAL_TOP_N_FOR_CONTEXT)
                        logger.info(f"Re-ranked results: {len(final_context_docs)} documents.")
                    else:
                        logger.info("Re-ranker model not loaded. Using documents from hybrid search directly.")
                        st.info("Re-ranker model not loaded. Using documents from hybrid search directly (top results).")
                        final_context_docs = docs_for_reranking[:FINAL_TOP_N_FOR_CONTEXT]

                    if not final_context_docs:
                        logger.warning("No relevant sections found after re-ranking (or hybrid search if reranker disabled).")
                        ai_response = "After re-ranking, no relevant sections were found in the loaded documents to answer your query."
                    else:
                        logger.debug("Generating answer with final context documents.")
                        ai_response = generate_answer(
                            LANGUAGE_MODEL,
                            user_query=user_input,
                            context_documents=final_context_docs,
                            conversation_history=formatted_history
                        )
                else:
                    logger.warning("No relevant sections found from hybrid search.")
                    ai_response = "I could not find relevant sections in the loaded documents to answer your query. Please ensure the documents contain information related to your query or try rephrasing."
            
            logger.info(f"AI Response: {ai_response[:100]}...") # Log snippet of AI response
            st.session_state.messages.append({"role": "assistant", "content": ai_response, "avatar": "🤖"})
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
else:
    st.info("Please upload one or more PDF, DOCX, or TXT documents to begin your session and ask questions.")
