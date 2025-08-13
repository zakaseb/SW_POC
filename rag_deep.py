import os
import streamlit as st
from rank_bm25 import BM25Okapi
import pandas as pd
import json
import io
import re
import base64

# Configure logging first
from core.logger_config import setup_logging, get_logger

setup_logging()  # Initialize logging system
logger = get_logger(__name__)


# Import from new core modules
from core.config import (
    MAX_HISTORY_TURNS,
    # K_SEMANTIC, # Removed as it's used in core.search_pipeline
    # K_BM25, # Removed as it's used in core.search_pipeline
    # K_RRF_PARAM, # Removed as it's used in core.search_pipeline
    TOP_K_FOR_RERANKER,  # Still used directly in rag_deep.py for slicing
    FINAL_TOP_N_FOR_CONTEXT,  # Still used directly in rag_deep.py for rerank_documents call
    PDF_STORAGE_PATH,
    CONTEXT_PDF_STORAGE_PATH,
)
from core.model_loader import (
    get_embedding_model,
    get_language_model,
    get_reranker_model,
)
from core.document_processing import (
    save_uploaded_file,
    load_document,
    chunk_documents,
    index_documents,
)
from core.search_pipeline import get_persistent_context, get_requirements_chunks
from core.generation import generate_answer, generate_summary, generate_keywords, generate_requirements_json
from core.session_manager import (
    initialize_session_state,
    reset_document_states,
    reset_file_uploader,
    purge_persistent_memory,
    save_persistent_memory,
)

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


def generate_excel_file(requirements_json_list):
    """
    Parses a list of JSON strings, cleans them, and generates an Excel file in memory.
    """
    all_requirements = []

    for json_str in requirements_json_list:
        # Clean the string: remove markdown and other non-JSON artifacts
        # This regex looks for content between ```json and ``` or just `{` and `}` or `[` and `]`
        match = re.search(r"```json\s*([\s\S]*?)\s*```|([\s\S]*)", json_str)
        if match:
            cleaned_str = match.group(1) if match.group(1) is not None else match.group(2)
            cleaned_str = cleaned_str.strip()

            try:
                # Try to parse the cleaned string
                data = json.loads(cleaned_str)
                if isinstance(data, list):
                    all_requirements.extend(data)
                elif isinstance(data, dict):
                    all_requirements.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from string: {cleaned_str}")
                continue # Skip this string if it's not valid JSON

    if not all_requirements:
        return None

    # Define the columns based on the JSON schema to ensure order and handle missing keys
    columns = [
        "Name",
        "Description",
        "VerificationMethod",
        "Tags",
        "RequirementType",
        "DocumentRequirementID"
    ]

    # Create a DataFrame
    df = pd.DataFrame(all_requirements)

    # Ensure all columns are present, fill missing ones with empty strings
    for col in columns:
        if col not in df.columns:
            df[col] = ''

    # Reorder columns to match the desired schema and select only them
    df = df[columns]

    # Convert list-like columns (e.g., Tags) to a string representation
    if 'Tags' in df.columns:
        df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Create an in-memory Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Requirements')

    processed_data = output.getvalue()
    return processed_data


def get_excel_download_link(excel_data):
    """
    Generates a link to download the given excel data.
    """
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="generated_requirements.xlsx" id="downloadLink" style="display:none">Download Excel</a>'
    script = """
    <script>
        window.setTimeout(function() {
            var link = document.getElementById('downloadLink');
            if (link) {
                link.click();
            }
        }, 200);
    </script>
    """
    return href + script


# ---------------------------------
# User Interface
# ---------------------------------

with st.sidebar:
    st.header("Controls")

    st.header("Context Document")
    context_uploaded_file = st.file_uploader(
        "Upload Context Document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key="context_file_uploader",
    )
    if context_uploaded_file is not None:
        context_file_info = {
            "name": context_uploaded_file.name,
            "size": context_uploaded_file.size,
        }
        if context_file_info != st.session_state.get("processed_context_file_info"):
            # Save the context document to the designated folder
            saved_path = save_uploaded_file(
                context_uploaded_file, CONTEXT_PDF_STORAGE_PATH
            )
            if saved_path:
                # Process and index the context document
                raw_docs = load_document(saved_path)
                if raw_docs:
                    _, _, all_chunks = chunk_documents(raw_docs, CONTEXT_PDF_STORAGE_PATH, classify=False)
                    if all_chunks:
                        index_documents(
                            all_chunks, vector_db=st.session_state.CONTEXT_VECTOR_DB
                        )
                        st.session_state.processed_context_file_info = context_file_info
                        st.success("Context document successfully uploaded!")
                    else:
                        st.error("Failed to generate chunks from the context document.")
                else:
                    st.error("Failed to load the context document.")
            else:
                st.error("Failed to save the context document.")

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")

    if st.button("Purge Memory", key="purge_memory"):
        purge_persistent_memory()
        logger.info("Persistent memory purged by user.")
        st.success("Persistent memory has been purged.")

    if st.button("Reset All Documents & Chat", key="reset_doc_chat_button"):
        logger.info("Resetting all documents and chat.")
        reset_document_states(clear_chat=True)
        reset_file_uploader()
        st.success("All documents and chat reset. You can now upload new documents.")
        logger.info("All documents and chat successfully reset.")
        st.rerun()

    if st.session_state.document_processed:
        if st.button("Generate Requirements", key="generate_requirements_button"):
            with st.spinner("Generating requirements... This might take a moment."):
                st.session_state.generated_requirements = None
                st.session_state.excel_file = None
                requirements_chunks = get_requirements_chunks(
                    document_vector_db=st.session_state.DOCUMENT_VECTOR_DB,
                )
                if not requirements_chunks:
                    st.sidebar.warning("No requirements chunks found to process.")
                else:
                    # Get the combined persistent and general context to be used for all requirement generations
                    common_context_docs = get_persistent_context(
                        context_vector_db=st.session_state.CONTEXT_VECTOR_DB,
                        general_context_chunks=st.session_state.get("general_context_chunks", []),
                    )

                    all_requirements = []
                    for req_chunk in requirements_chunks:
                        json_response = generate_requirements_json(
                            LANGUAGE_MODEL,
                            req_chunk,
                            context_documents=common_context_docs,
                        )
                        all_requirements.append(json_response)
                    st.session_state.generated_requirements = all_requirements

                    excel_data = generate_excel_file(all_requirements)
                    if excel_data:
                        download_link = get_excel_download_link(excel_data)
                        st.session_state.download_trigger = download_link
                        st.sidebar.success("Requirements generated and Excel file is downloading!")
                    else:
                        st.sidebar.warning("Requirements generated, but no valid data was found to create an Excel file.")

        if st.button(
            "Extract Keywords from Content", key="extract_keywords_content_button"
        ):
            if st.session_state.raw_documents:
                with st.spinner("Extracting keywords from all documents..."):
                    full_text = "\n\n".join(
                        [doc.page_content for doc in st.session_state.raw_documents]
                    )
                    if not full_text.strip():
                        st.sidebar.warning(
                            "Cannot extract keywords: Combined content of documents is effectively empty."
                        )
                        logger.warning(
                            "Keyword extraction attempt on empty combined content."
                        )
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
                st.sidebar.error(
                    "Cannot extract keywords: Document content not loaded or available."
                )
                logger.warning(
                    "Keyword extraction attempt with no raw documents loaded."
                )

    if st.session_state.get("generated_requirements"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Generated Requirements")
        # Display each requirement in a separate text area
        for i, req in enumerate(st.session_state.generated_requirements):
            st.sidebar.text_area(
                f"Requirement {i+1}", value=req, height=200, disabled=True
            )

        if st.session_state.get("excel_file"):
            # The download will now be triggered automatically.
            pass

    if (
        st.session_state.document_keywords
        and "Failed to extract keywords" not in st.session_state.document_keywords
    ):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 Extracted Keywords (Combined)")
        st.sidebar.text_area(
            "Keywords:", st.session_state.document_keywords, height=100, disabled=True
        )

st.title("📘 MBSE: Requirement Generator")
st.markdown("### Your AI Document Assistant")
st.markdown("---")

if st.session_state.get("context_document_loaded"):
    st.info("A context document is loaded and will be used in the session.")

uploaded_files = st.file_uploader(
    "Upload Research Documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    help="Select one or more PDF, DOCX, or TXT documents for analysis. Processing will begin upon upload. Re-uploading or changing files will reset the session.",
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploaded_file_key}",
)

if uploaded_files:
    current_uploaded_files_info = {f.name: f.size for f in uploaded_files}
    logger.info(f"Files uploaded: {list(current_uploaded_files_info.keys())}")

    # Check if the uploaded files are the same as the ones already processed
    if current_uploaded_files_info != st.session_state.get("processed_files_info", {}):
        logger.info("New set of files detected. Resetting document states and file uploader.")
        reset_document_states(clear_chat=True)
        # It's important to call reset_file_uploader to ensure the UI component updates
        reset_file_uploader()

        all_raw_docs_for_session = []
        successfully_loaded_filenames = []
        processed_files_info = {}

        for uploaded_file_obj in uploaded_files:
            filename = uploaded_file_obj.name
            file_size = uploaded_file_obj.size
            logger.debug(f"Processing uploaded file: {filename}")

            with st.spinner(f"Processing '{filename}'... This may take a moment."):
                saved_path = save_uploaded_file(uploaded_file_obj, PDF_STORAGE_PATH)
                if saved_path:
                    logger.info(f"File '{filename}' saved to '{saved_path}'")
                    raw_docs_from_file = load_document(saved_path)
                    if raw_docs_from_file:
                        all_raw_docs_for_session.extend(raw_docs_from_file)
                        successfully_loaded_filenames.append(filename)
                        processed_files_info[filename] = file_size
                        logger.info(f"Successfully loaded and parsed: {filename}")
                    else:
                        st.error(f"Could not load document from '{filename}'. It might be empty, corrupted, or an unsupported type.")
                        logger.error(f"Failed to load document from '{filename}'.")
                else:
                    st.error(f"Failed to save '{filename}'. It will be skipped.")
                    logger.error(f"Failed to save '{filename}'.")

        st.session_state.uploaded_filenames = successfully_loaded_filenames
        st.session_state.processed_files_info = processed_files_info

        if all_raw_docs_for_session:
            st.session_state.raw_documents = all_raw_docs_for_session
            logger.info(f"Total {len(all_raw_docs_for_session)} raw documents collected from {len(successfully_loaded_filenames)} files.")

            with st.spinner(f"Chunking, classifying, and indexing {len(st.session_state.uploaded_filenames)} document(s)..."):
                logger.debug("Starting document chunking and classification.")
                general_context_chunks, requirements_chunks, _ = chunk_documents(st.session_state.raw_documents, classify=True)

                total_chunks = len(general_context_chunks) + len(requirements_chunks)
                if total_chunks > 0:
                    st.success(f"Document chunking and classification complete. Found {len(general_context_chunks)} general context chunks and {len(requirements_chunks)} requirements chunks.")
                    logger.info(f"{total_chunks} total chunks created.")

                    # Store general context chunks in session state, but do not index into the persistent context vector DB
                    if general_context_chunks:
                        st.session_state.general_context_chunks = general_context_chunks
                        logger.info(f"{len(general_context_chunks)} general context chunks stored in session state.")
                        st.info(f"ℹ️ {len(general_context_chunks)} general context chunks will be used for this session.")

                    # Index requirements chunks into the main document vector DB
                    if requirements_chunks:
                        logger.debug("Starting indexing of requirements chunks.")
                        index_documents(requirements_chunks, vector_db=st.session_state.DOCUMENT_VECTOR_DB)
                        logger.info(f"{len(requirements_chunks)} requirements chunks indexed.")

                    # Set document_processed flag to true if any chunk was processed
                    st.session_state.document_processed = True

                    if st.session_state.document_processed:
                        logger.info("Vector indexing successful for one or both chunk types.")
                        try:
                            logger.debug("Starting BM25 indexing on requirements chunks.")
                            corpus_texts = [chunk.page_content for chunk in requirements_chunks]
                            if corpus_texts:
                                tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
                                st.session_state.bm25_index = BM25Okapi(tokenized_corpus)
                                st.session_state.bm25_corpus_chunks = requirements_chunks
                                display_filenames = ", ".join(st.session_state.uploaded_filenames)
                                logger.info(f"BM25 index created for documents: {display_filenames}")
                                st.success(f"✅ Documents ({display_filenames}) processed and indexed successfully!")
                            else:
                                logger.info("No requirements chunks to index for BM25.")
                                st.success("✅ Documents processed. No specific requirements chunks found for keyword search indexing.")
                            st.rerun()

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
    else:
        logger.info("Uploaded files are the same as the ones already processed. Skipping reprocessing.")

if st.session_state.get("uploaded_filenames") and st.session_state.get(
    "document_processed"
):
    st.markdown("---")
    st.markdown(f"**Successfully processed document(s):**")
    for name in st.session_state.uploaded_filenames:
        st.markdown(f"- _{name}_")
    st.markdown("---")

# Check if a download should be triggered
if "download_trigger" in st.session_state and st.session_state.download_trigger:
    st.components.v1.html(st.session_state.download_trigger, height=0)
    st.session_state.download_trigger = None # Clear the trigger

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.write(message["content"])

if st.session_state.document_processed:
    if len(st.session_state.uploaded_filenames) > 1:
        chat_placeholder = f"Ask a question about the {len(st.session_state.uploaded_filenames)} loaded documents..."
    elif len(st.session_state.uploaded_filenames) == 1:
        chat_placeholder = (
            f"Ask a question about '{st.session_state.uploaded_filenames[0]}'..."
        )
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
                st.session_state.memory.append({"role": "user", "content": user_input})
                num_messages_to_take = MAX_HISTORY_TURNS * 2
                chat_log_for_prompt = st.session_state.messages[:-1]
                chat_log_for_prompt = chat_log_for_prompt[-num_messages_to_take:]
                history_lines = [
                    f"{('User' if msg['role'] == 'user' else 'Assistant')}: {msg['content']}"
                    for msg in chat_log_for_prompt
                ]
                formatted_history = "\n".join(history_lines)
                logger.debug(f"Formatted history for prompt: {formatted_history}")

                # Get all chunks for context
                persistent_context = get_persistent_context(
                    context_vector_db=st.session_state.CONTEXT_VECTOR_DB,
                    general_context_chunks=st.session_state.get("general_context_chunks", []),
                )
                requirements_chunks = get_requirements_chunks(
                    document_vector_db=st.session_state.DOCUMENT_VECTOR_DB,
                )
                all_context_docs = persistent_context + requirements_chunks

                if not all_context_docs:
                    logger.warning("No documents found in any vector store.")
                    ai_response = "No documents have been processed yet. Please upload a document to begin."
                else:
                    logger.debug("Generating answer with all available documents.")
                    persistent_memory_str = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory]
                    )
                    ai_response = generate_answer(
                        LANGUAGE_MODEL,
                        user_query=user_input,
                        context_documents=all_context_docs,
                        conversation_history=formatted_history,
                        persistent_memory=persistent_memory_str,
                    )

            logger.info(
                f"AI Response: {ai_response[:100]}..."
            )  # Log snippet of AI response
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response, "avatar": "🤖"}
            )
            st.session_state.memory.append({"role": "assistant", "content": ai_response})
            save_persistent_memory()
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
else:
    st.info(
        "Please upload one or more PDF, DOCX, or TXT documents to begin your session and ask questions."
    )
