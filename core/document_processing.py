import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document as LangchainDocument
import docx
from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError
import pdfplumber
from .config import PDF_STORAGE_PATH, CONTEXT_PDF_STORAGE_PATH
from .logger_config import get_logger
from .model_loader import get_language_model
from .generation import classify_chunk

# Docling imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from urllib.parse import unquote

import json
from langchain_core.documents import Document
from typing import Any, List, Tuple
from langchain_core.documents import Document as LCDocument

logger = get_logger(__name__)

def save_uploaded_file(uploaded_file, storage_path=PDF_STORAGE_PATH):
    # Create the storage directory if it doesn't exist
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, uploaded_file.name)
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        logger.info(f"File '{uploaded_file.name}' saved to '{file_path}'.")
        return file_path
    except IOError as e:
        user_message = f"Failed to save uploaded file '{uploaded_file.name}'. An I/O error occurred: {e.strerror}."
        logger.error(f"{user_message} Please check permissions and disk space.")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = f"An unexpected error occurred while saving '{uploaded_file.name}'."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return None

def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    try:
        if file_extension == ".pdf":
            try:
                logger.debug(f"Loading PDF: {file_name}")
                document_loader = PDFPlumberLoader(file_path)
                docs = document_loader.load()
                logger.info(f"Successfully loaded PDF: {file_name}, {len(docs)} pages/documents.")
                return docs
            except pdfplumber.exceptions.PDFSyntaxError as pdf_err:
                user_message = f"Failed to load PDF '{file_name}': The file may be corrupted or not a valid PDF."
                logger.error(f"{user_message} Details: {pdf_err}")
                st.error(user_message)
                return []
            except Exception as e:
                user_message = f"Failed to load PDF '{file_name}': An unexpected error occurred during PDF processing."
                logger.exception(f"{user_message} Details: {e}")
                st.error(f"{user_message} Check logs for details.")
                return []
        elif file_extension == ".docx":
            try:
                logger.debug(f"Loading DOCX: {file_name}")
                doc = docx.Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                if not full_text.strip():
                    logger.warning(f"DOCX file '{file_name}' is empty or contains no text.")
                    st.warning(f"DOCX file '{file_name}' appears to be empty or contains no text.")
                    return []
                logger.info(f"Successfully loaded DOCX: {file_name}")
                return [LangchainDocument(page_content=full_text, metadata={"source": file_name})]
            except DocxPackageNotFoundError as e:
                user_message = f"Failed to load DOCX '{file_name}': The file appears to be corrupted or not a valid DOCX file."
                logger.error(f"{user_message} Error: {e}")
                st.error(user_message)
                return []
            except Exception as e:
                user_message = f"Failed to load DOCX '{file_name}': An unexpected error occurred."
                logger.exception(f"{user_message} Details: {e}")
                st.error(f"{user_message} Check logs for details.")
                return []
        elif file_extension == ".txt":
            try:
                logger.debug(f"Loading TXT: {file_name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                if not full_text.strip():
                    logger.warning(f"Text file '{file_name}' is empty.")
                    st.warning(f"Text file '{file_name}' appears to be empty.")
                    return []
                logger.info(f"Successfully loaded TXT: {file_name}")
                return [LangchainDocument(page_content=full_text, metadata={"source": file_name})]
            except UnicodeDecodeError as unicode_err:
                user_message = f"Failed to load TXT file '{file_name}': The file is not UTF-8 encoded."
                logger.error(f"{user_message} Details: {unicode_err}")
                st.error(user_message)
                return []
            except IOError as io_err:
                user_message = f"Failed to load TXT file '{file_name}': An I/O error occurred. {io_err.strerror}."
                logger.error(f"{user_message} Details: {io_err}")
                st.error(user_message)
                return []
            except Exception as e:
                user_message = f"Failed to load TXT file '{file_name}': An unexpected error occurred."
                logger.exception(f"{user_message} Details: {e}")
                st.error(f"{user_message} Check logs for details.")
                return []
        else:
            user_message = f"Unsupported file type: '{file_extension}' for file '{file_name}'."
            logger.warning(user_message)
            st.error(user_message)
            return []
    except MemoryError as mem_err:
        user_message = f"Failed to load document '{file_name}': The file is too large to process with available memory."
        logger.error(f"{user_message} Details: {mem_err}")
        st.error(user_message)
        return []
    except Exception as e:
        user_message = f"An unexpected error occurred while attempting to load '{file_name}'."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return []

def chunk_documents(raw_documents, storage_path=PDF_STORAGE_PATH, classify=False):
    if not raw_documents:
        logger.warning("chunk_documents called with no raw documents.")
        st.warning("No content found in the document to chunk.")
        return [], [], []

    logger.info(f"Starting Docling hybrid chunking on {len(raw_documents)} document(s).")

    try:
        converter = DocumentConverter()
        hf_tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
        chunker = HybridChunker(tokenizer=hf_tokenizer, merge_peers=True)

        all_chunks = []
        general_context_chunks = []
        requirements_chunks = []
        processed_chunk_texts = set()

        for doc in raw_documents:
            source_path = doc.metadata.get("source")
            if not source_path:
                raise ValueError("Document is missing 'source' metadata.")

            full_path = source_path
            if not os.path.exists(full_path):
                logger.warning(f"Source path '{full_path}' not found. Attempting to resolve with storage path '{storage_path}'.")
                full_path = os.path.join(storage_path, os.path.basename(source_path))
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Resolved file path does not exist: {full_path}")

            dl_doc = converter.convert(source=full_path).document
            chunks = list(chunker.chunk(dl_doc))
            logger.info(f"Number of chunks before deduplication: {len(chunks)}")

            for i, c in enumerate(chunks):
                chunk_text = c.text.strip()
                if not chunk_text or chunk_text in processed_chunk_texts:
                    continue

                processed_chunk_texts.add(chunk_text)

                if classify:
                    language_model = get_language_model()
                    classification = classify_chunk(language_model, chunk_text)
                    print(f"--- Chunk {i+1} ---")
                    print(f"Classification: {classification}")
                    print(f"Text: {chunk_text}")
                    print("--------------------")
                    if classification == "General Context":
                        general_context_chunks.append(
                            LangchainDocument(
                                page_content=chunk_text,
                                metadata={**doc.metadata, "headings": c.meta.headings, "in_memory": True},
                            )
                        )
                    else:
                        requirements_chunks.append(
                            LangchainDocument(
                                page_content=chunk_text,
                                metadata={**doc.metadata, "headings": c.meta.headings, "in_memory": False},
                            )
                        )
                else:
                    all_chunks.append(
                        LangchainDocument(
                            page_content=chunk_text,
                            metadata={**doc.metadata, "headings": c.meta.headings, "in_memory": False},
                        )
                    )

        if classify:
            logger.info(f"Docling hybrid chunking and classification complete.")
            logger.info(f"Number of chunks after deduplication: {len(processed_chunk_texts)}")
            logger.info(f"  - General Context chunks: {len(general_context_chunks)}")
            logger.info(f"  - Requirements chunks: {len(requirements_chunks)}")
        else:
            logger.info(f"Docling hybrid chunking complete: {len(all_chunks)} chunks created.")

        return general_context_chunks, requirements_chunks, all_chunks

    except Exception as e:
        logger.exception(f"An error occurred during hybrid chunking using Docling. Details: {e}")
        st.error("An error occurred during hybrid chunking using Docling. Check logs for details.")
        return [], [], []


# If you already have these helpers elsewhere, you can delete these local versions
ALLOWED_META = (str, int, float, bool, type(None))

def _to_texts_and_metas(chunks: List[Any]) -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    for ch in chunks or []:
        if isinstance(ch, LCDocument):
            t = ch.page_content
            m = dict(ch.metadata or {})
        elif isinstance(ch, dict):
            t = ch.get("page_content") or ch.get("text") or ch.get("content") or ""
            m = dict(ch.get("metadata") or {})
        else:
            t, m = str(ch), {}
        t = str(t or "").strip()
        if not t:
            continue
        texts.append(t)
        metas.append(m)
    return texts, metas

def _sanitize_metadatas_json(metadatas: List[dict], n: int) -> List[dict]:
    out = []
    for md in metadatas or []:
        safe = {}
        for k, v in (md or {}).items():
            if isinstance(v, (list, tuple, set)):
                try: v = json.dumps(list(v), ensure_ascii=False)
                except Exception: v = str(list(v))
            elif isinstance(v, dict):
                try: v = json.dumps(v, ensure_ascii=False)
                except Exception: v = str(v)
            elif not isinstance(v, ALLOWED_META):
                v = str(v)
            safe[k] = v
        out.append(safe)
    if len(out) < n: out.extend({} for _ in range(n - len(out)))
    elif len(out) > n: out = out[:n]
    return out

def index_documents(document_chunks: List[Any], vector_db=None) -> int:
    """
    Mirror the old behavior: warn on empty, default DB from session, try/except,
    set document_processed, but embed safely for LM Studio using add_texts.
    """
    if not document_chunks:
        logger.warning("index_documents called with no chunks to index.")
        st.warning("No document chunks available to index.")
        st.session_state["document_processed"] = False
        return 0

    logger.info(f"Indexing {len(document_chunks)} document chunks.")
    try:
        # Keep the same default as before (DOCUMENT_VECTOR_DB). Change to GENERAL_VECTOR_DB if you prefer.
        if vector_db is None:
            vector_db = st.session_state.get("DOCUMENT_VECTOR_DB")
            if vector_db is None:
                raise RuntimeError("No vector_db provided and none found in session_state (DOCUMENT_VECTOR_DB missing).")

        # --- LM Studio–safe path ---
        texts, metas = _to_texts_and_metas(document_chunks)
        if not texts:
            st.session_state["document_processed"] = False
            logger.info("All chunks were empty after coercion; nothing indexed.")
            return 0

        # guard + sanitize
        assert all(isinstance(t, str) and t.strip() for t in texts), "Empty/non-string chunk detected"
        metas = _sanitize_metadatas_json(metas, n=len(texts))

        # CRITICAL: use add_texts so /v1/embeddings receives list[str]
        vector_db.add_texts(texts=texts, metadatas=metas)

        st.session_state["document_processed"] = True
        logger.info("Document chunks indexed successfully into vector store.")
        return len(texts)

    except Exception as e:
        user_message = "An error occurred while indexing document chunks."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        st.session_state["document_processed"] = False
        return 0

def re_index_documents_from_session():
    """
    Re-indexes documents from chunks stored in the session state.
    This is used to repopulate in-memory vector databases after a session is loaded.
    """
    logger.info("Attempting to re-index documents from session state.")

    # Re-index general context chunks
    if "general_context_chunks" in st.session_state and st.session_state.general_context_chunks:
        logger.info(f"Re-indexing {len(st.session_state.general_context_chunks)} general context chunks.")
        index_documents(st.session_state.general_context_chunks, vector_db=st.session_state.GENERAL_VECTOR_DB)
    else:
        logger.info("No general context chunks found in session state to re-index.")

    # Re-index requirements chunks
    if "requirements_chunks" in st.session_state and st.session_state.requirements_chunks:
        logger.info(f"Re-indexing {len(st.session_state.requirements_chunks)} requirements chunks.")
        index_documents(st.session_state.requirements_chunks, vector_db=st.session_state.DOCUMENT_VECTOR_DB)
    else:
        logger.info("No requirements chunks found in session state to re-index.")

    # Re-index standalone context chunks
    if "context_chunks" in st.session_state and st.session_state.context_chunks:
        logger.info(f"Re-indexing {len(st.session_state.context_chunks)} standalone context chunks.")
        index_documents(st.session_state.context_chunks, vector_db=st.session_state.CONTEXT_VECTOR_DB)
        # Also ensure the loaded flag is set if we are re-indexing its chunks
        st.session_state.context_document_loaded = True
    else:
        logger.info("No standalone context chunks found in session state to re-index.")
