import os

# Avoid inotify watcher exhaustion on systems with low watch limits
os.environ.setdefault("STREAMLIT_WATCHER_TYPE", "poll")

import streamlit as st
from rank_bm25 import BM25Okapi
import pandas as pd
import json
import re

# Configure logging first
from core.logger_config import setup_logging, get_logger

setup_logging()  # Initialize logging system
logger = get_logger(__name__)

from pathlib import Path
import shutil

from core.auth import show_login_form
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
from core.search_pipeline import get_persistent_context, get_general_context, get_requirements_chunks
from core.generation import generate_answer
from core.session_manager import (
    initialize_session_state,
    reset_document_states,
    reset_file_uploader,
    purge_persistent_memory,
)
from core.database import save_session
from core.session_utils import package_session_for_storage
from core.requirement_jobs import (
    submit_requirement_generation_job,
    get_latest_requirement_job,
    get_requirement_job,
    list_requirement_jobs,
    load_job_excel_bytes,
    load_job_requirements,
)

from core.config import USE_API_WRAPPER, API_URL

import requests  

def log_ui_event_to_api(event: str, metadata: dict | None = None):
    """
    Send a UI event to the FastAPI wrapper so it shows up in API logs.
    """
    if not USE_API_WRAPPER:
        # In DIRECT mode you might skip this, or still send if you like
        return

    payload = {
        "event": event,
        "user_id": st.session_state.get("user_id"),
        "metadata": metadata or {},
    }

    try:
        requests.post(
            f"{API_URL}/ui-event",
            json=payload,
            timeout=2,
        )
    except Exception as e:
        # Don't break the UI if logging fails, just warn
        logger.warning(f"Failed to send UI event to API: {e}")

# Use your existing config var
CONTEXT_DIR = Path(CONTEXT_PDF_STORAGE_PATH).resolve()

# Make TEMPLATE_DOC absolute (safer for Streamlit reruns)
APP_ROOT = Path(__file__).resolve().parent
TEMPLATE_DOC = (APP_ROOT / "Verification Methods.docx").resolve()


def ensure_global_context_bootstrap() -> Path:
    """Ensure global context dir exists and contains the default doc.
    Returns the absolute path to the context file."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CONTEXT_DIR / TEMPLATE_DOC.name

    if not TEMPLATE_DOC.exists():
        logger.error(f"Template not found: {TEMPLATE_DOC}")
        st.warning(f"Default context template not found at {TEMPLATE_DOC}")
        return dest  # path where it would live

    if not dest.exists():
        shutil.copyfile(TEMPLATE_DOC, dest)
        logger.info(f"Global default context copied to: {dest}")
    else:
        logger.info(f"Global default context already present: {dest}")

    return dest


def index_global_context_once(ctx_file: Path):
    """Chunk + index the global context once per session or when the file changes."""
    # do nothing if user purged this session
    if not st.session_state.get("allow_global_context", False):
        return
    if not ctx_file.exists():
        return

    mtime = ctx_file.stat().st_mtime
    if st.session_state.get("global_ctx_indexed_mtime") == mtime:
        # already indexed this exact content
        return

    raw = load_document(str(ctx_file))
    if not raw:
        return
    _, _, chunks = chunk_documents(raw, str(CONTEXT_DIR), classify=False)
    if chunks:
        index_documents(chunks, vector_db=st.session_state.CONTEXT_VECTOR_DB)
        # Persistent backup (DO NOT purge this in your reset functions)
        index_documents(chunks, vector_db=st.session_state.PERSISTENT_VECTOR_DB)
        logger.info(f"Global context indexed: {len(chunks)} chunks.")
        st.session_state.global_ctx_indexed_mtime = mtime
        st.session_state.context_document_loaded = True


def _split_paras(text: str, max_chars: int = 900, overlap: int = 120):
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out = []
    for p in paras:
        if len(p) <= max_chars:
            out.append(p)
        else:
            i = 0
            while i < len(p):
                j = min(i + max_chars, len(p))
                out.append(p[i:j].strip())
                i = max(j - overlap, j)
    return out


def refresh_requirement_job_state():
    """Load the latest requirement job status and artifacts for the logged-in user."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    latest_job = get_latest_requirement_job(user_id)
    st.session_state.latest_requirement_job = latest_job

    if not latest_job:
        st.session_state.pop("latest_requirement_job_error", None)
        return

    status = latest_job.get("status")
    if status == "completed":
        excel_bytes = load_job_excel_bytes(latest_job)
        if excel_bytes:
            st.session_state.excel_file_data = excel_bytes
        requirements_payload = load_job_requirements(latest_job)
        if requirements_payload:
            st.session_state.generated_requirements = requirements_payload
    if status == "failed":
        st.session_state.latest_requirement_job_error = latest_job.get("error_message")
    else:
        st.session_state.pop("latest_requirement_job_error", None)

# ---------------------------------
# App Styling with CSS
# ---------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F6F8;
        color: #384671;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #FFFFFF !important;
        color: #384671 !important;
        border: 1px solid #D1D9E0 !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D9E0 !important;
        color: #384671 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #F3F6F8 !important;
        border: 1px solid #D1D9E0 !important;
        color: #384671 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #384671 !important;
    }
    .stFileUploader {
        background-color: #FFFFFF;
        border: 1px solid #D1D9E0;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #384671 !important;
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


if not show_login_form():
    st.stop()

if st.session_state.get("authenticated") and not st.session_state.get("login_logged"):
    log_ui_event_to_api("login_success")
    st.session_state["login_logged"] = True

# Auto-enable global context once per session after successful login
if (st.session_state.get("authenticated")
    and st.session_state.get("user_id")
    and "allow_global_context" not in st.session_state):
    st.session_state["allow_global_context"] = True     # ✅ auto ON at login
    st.session_state["did_context_bootstrap"] = False   # allow one-time bootstrap
    st.session_state.pop("global_ctx_indexed_mtime", None)

# Auto-bootstrap ONCE per session, only if allowed
if (st.session_state.get("authenticated")
    and st.session_state.get("user_id")
    and st.session_state.get("allow_global_context", False)
    and not st.session_state.get("did_context_bootstrap", False)):
    ctx_path = ensure_global_context_bootstrap()
    index_global_context_once(ctx_path)
    st.session_state["did_context_bootstrap"] = True
    st.session_state["context_document_loaded"] = True

# Keep requirement job state in sync for authenticated users
if st.session_state.get("authenticated") and st.session_state.get("user_id"):
    refresh_requirement_job_state()
   
# ---------------------------------
# User Interface
# ---------------------------------

with st.sidebar:
    st.image("halcon_logo.jpeg", use_container_width=True)
    st.header("Controls")

    if st.button("Logout", key="logout_button"):
        log_ui_event_to_api("logout_clicked")
        session_to_save = package_session_for_storage()
        save_session(st.session_state.user_id, session_to_save)

        st.session_state.authenticated = False
        # Clear sensitive and user-specific keys from session state
        for key in list(st.session_state.keys()):
            if key not in ['authenticated']: # Keep authentication status
                del st.session_state[key]

        # Re-initialize for a clean slate, except for auth status
        initialize_session_state()

        st.rerun()

    st.header("Context Document")
    context_uploaded_file = st.file_uploader(
        "Upload Context Document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key="context_file_uploader",
    )

    if context_uploaded_file is not None:
        context_file_info = {"name": context_uploaded_file.name, "size": context_uploaded_file.size}
        if context_file_info != st.session_state.get("processed_context_file_info"):
            saved_path = save_uploaded_file(context_uploaded_file, CONTEXT_PDF_STORAGE_PATH)
            if saved_path:
                raw_docs = load_document(saved_path)
                if raw_docs:
                    _, _, all_chunks = chunk_documents(raw_docs, CONTEXT_PDF_STORAGE_PATH, classify=False)
                    if all_chunks:
                        index_documents(all_chunks, vector_db=st.session_state.CONTEXT_VECTOR_DB)
                        st.session_state.processed_context_file_info = context_file_info
                        st.session_state.context_chunks = all_chunks
                        st.session_state.context_document_loaded = True
                        st.success("Context document successfully uploaded!")
                        log_ui_event_to_api("context_upload",{"name": context_uploaded_file.name, "size": context_uploaded_file.size},)
                    else:
                        st.error("Failed to generate chunks from the context document.")
                else:
                    st.error("Failed to load the context document.")
            else:
                st.error("Failed to save the context document.")

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        log_ui_event_to_api("chat_cleared")

    if st.button("Delete Context Document", key="purge_memory"):
        purge_persistent_memory()
        logger.info("Context document deleted by user.")
        st.success("Context document has been deleted.")
        log_ui_event_to_api("context_deleted")

    if st.button("Reset All Documents & Chat", key="reset_doc_chat_button"):
        logger.info("Resetting all documents and chat.")
        reset_document_states(clear_chat=True)
        reset_file_uploader()
        st.success("All documents and chat reset. You can now upload new documents.")
        logger.info("All documents and chat successfully reset.")
        log_ui_event_to_api("reset_all_done")
        st.rerun()

    job_info = st.session_state.get("latest_requirement_job")
    job_status = job_info.get("status") if job_info else None
    generate_requirements_slot = st.empty()

    if st.session_state.pop("requirement_job_submission_success", False):
        st.sidebar.success("Requirement generation started in the background.")

    if job_info:
        status_label = (job_status or "unknown").capitalize()
        st.sidebar.markdown("---")
        st.sidebar.subheader("Requirement Job Status")
        st.sidebar.write(f"Latest job: **{status_label}**")
        if job_status == "failed":
            error_msg = st.session_state.get("latest_requirement_job_error") or job_info.get("error_message")
            if error_msg:
                st.sidebar.error(f"Job failed: {error_msg}")
        elif job_status in {"queued", "running"}:
            st.sidebar.info("A background job is running. You may continue working or log out safely.")
        elif job_status == "completed":
            st.sidebar.success("Latest requirements are ready to download.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.caption("No requirement generation jobs yet.")

    if st.session_state.get("user_id"):
        recent_jobs = list_requirement_jobs(st.session_state.user_id, limit=3)
        if recent_jobs:
            st.sidebar.caption("Recent jobs:")
            for job in recent_jobs:
                st.sidebar.caption(
                    f"{job['id'][:8]} · {job['status'].title()} · {job['updated_at']}"
                )

    if st.session_state.get("excel_file_data"):
        download_clicked = st.download_button(
            label="Download Requirements",
            data=st.session_state.excel_file_data,
            file_name="generated_requirements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_requirements"
        )

        if download_clicked:
            log_ui_event_to_api("download_requirements_clicked")

    generated_requirements = st.session_state.get("generated_requirements")
    if generated_requirements:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Generated Requirements")
        for i, req in enumerate(generated_requirements):
            if isinstance(req, dict):
                display_value = json.dumps(req, indent=2)
            else:
                display_value = str(req)
            st.sidebar.text_area(
                f"Requirement {i+1}", value=display_value, height=200, disabled=True
            )


st.title("📘 MBSE: JAMA Requirement Generator")
st.markdown("### Your AI Document Assistant")
st.markdown("---")

if st.session_state.get("allow_global_context") and st.session_state.get("context_document_loaded"):
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

    log_ui_event_to_api(
        "files_selected",
        {"files": current_uploaded_files_info},
    )

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
                # saved_path = save_uploaded_file(uploaded_file_obj, PDF_STORAGE_PATH, allow_context=False)
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

                    # Index general context chunks into the general vector DB
                    if general_context_chunks:
                        st.session_state.general_context_chunks = general_context_chunks
                        logger.debug("Starting indexing of general context chunks.")
                        index_documents(general_context_chunks, vector_db=st.session_state.GENERAL_VECTOR_DB)
                        logger.info(f"{len(general_context_chunks)} general context chunks indexed.")
                        st.info(f"ℹ️ {len(general_context_chunks)} chunks have been added to the session's persistent memory.")

                    # Index requirements chunks into the main document vector DB
                    if requirements_chunks:
                        st.session_state.requirements_chunks = requirements_chunks
                        logger.debug("Starting indexing of requirements chunks.")
                        index_documents(requirements_chunks, vector_db=st.session_state.DOCUMENT_VECTOR_DB)
                        logger.info(f"{len(requirements_chunks)} requirements chunks indexed.")

                    if st.session_state.get("document_processed", False):
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
                                log_ui_event_to_api(
                                "documents_processed",
                                {
                                    "filenames": st.session_state.uploaded_filenames,
                                    "num_general_chunks": len(general_context_chunks),
                                    "num_requirements_chunks": len(requirements_chunks),
                                },)
                            else:
                                logger.info("No requirements chunks to index for BM25.")
                                st.success("✅ Documents processed. No specific requirements chunks found for keyword search indexing.")

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

if st.session_state.document_processed:
    generate_disabled = job_status in {"queued", "running"}
    if generate_requirements_slot.button(
        "Generate Requirements",
        key="generate_requirements_button",
        disabled=generate_disabled,
    ):
        with st.spinner("Submitting requirement generation job..."):
            requirements_chunks = get_requirements_chunks(
                document_vector_db=st.session_state.DOCUMENT_VECTOR_DB,
            )
            if not requirements_chunks:
                st.sidebar.warning("No requirements chunks found to process.")
            else:
                verif_docs = get_persistent_context(
                    context_vector_db=st.session_state.PERSISTENT_VECTOR_DB
                )
                general_docs = get_general_context(
                    general_vector_db=st.session_state.GENERAL_VECTOR_DB
                )

                try:
                    job_id = submit_requirement_generation_job(
                        user_id=st.session_state.user_id,
                        language_model=LANGUAGE_MODEL,
                        requirements_chunks=requirements_chunks,
                        verification_docs=verif_docs,
                        general_docs=general_docs,
                    )
                    st.session_state.latest_requirement_job = get_requirement_job(job_id)
                    refresh_requirement_job_state()
                    log_ui_event_to_api(
                        "requirements_job_submitted",
                        {"job_id": job_id, "chunk_count": len(requirements_chunks)},
                    )
                    st.session_state.requirement_job_submission_success = True
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Failed to queue requirement job: {exc}")

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
        log_ui_event_to_api(
        "chat_question",
        {"query": user_input[:200]},  # truncate for safety
        )
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
                    context_vector_db=st.session_state.CONTEXT_VECTOR_DB
                )

                general_context = get_general_context(
                    general_vector_db=st.session_state.GENERAL_VECTOR_DB
                )

                requirements_chunks = get_requirements_chunks(
                    document_vector_db=st.session_state.DOCUMENT_VECTOR_DB,
                )
                all_context_docs = persistent_context + general_context + requirements_chunks

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
                    log_ui_event_to_api(
                        "chat_answer",
                        {"answer_preview": ai_response[:200]},
                    )

            logger.info(
                f"AI Response: {ai_response[:100]}..."
            )  # Log snippet of AI response
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response, "avatar": "🤖"}
            )
            st.session_state.memory.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
else:
    st.info(
        "Please upload one or more PDF, DOCX, or TXT documents to begin your session and ask questions."
    )
