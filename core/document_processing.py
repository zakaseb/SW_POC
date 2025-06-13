import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
import docx
import pdfplumber
from .config import PDF_STORAGE_PATH
from .logger_config import get_logger

logger = get_logger(__name__)

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to disk and return the file path.
    """
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
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
    """
    Load documents from PDF, DOCX, or TXT files.
    """
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
            except docx.opc.exceptions.PackageNotFoundError as docx_err:
                user_message = f"Failed to load DOCX '{file_name}': The file appears to be corrupted or not a valid DOCX file."
                logger.error(f"{user_message} Details: {docx_err}")
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
                user_message = f"Failed to load TXT file '{file_name}': The file is not UTF-8 encoded. Please ensure it's a plain text file with UTF-8 encoding."
                logger.error(f"{user_message} Details: {unicode_err}")
                st.error(user_message)
                return []
            except IOError as io_err:
                user_message = f"Failed to load TXT file '{file_name}': An I/O error occurred. {io_err.strerror}."
                logger.error(f"{user_message}")
                st.error(user_message)
                return []
            except Exception as e:
                user_message = f"Failed to load TXT file '{file_name}': An unexpected error occurred."
                logger.exception(f"{user_message} Details: {e}")
                st.error(f"{user_message} Check logs for details.")
                return []
        else:
            user_message = f"Unsupported file type: '{file_extension}' for file '{file_name}'. Please upload a PDF, DOCX, or TXT file."
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

def chunk_documents(raw_documents):
    """
    Split raw documents into manageable chunks using RecursiveCharacterTextSplitter.
    """
    if not raw_documents:
        logger.warning("chunk_documents called with no raw documents.")
        st.warning("No content found in the document to chunk.") # User feedback is fine
        return []
    logger.info(f"Starting to chunk {len(raw_documents)} raw document(s).")
    try:
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        processed_docs = text_processor.split_documents(raw_documents)
        if not processed_docs:
            logger.warning("Document chunking resulted in no processable text chunks.")
            st.warning("Document chunking resulted in no processable text chunks. The document might be valid but contain no extractable text after initial processing, or the text is too short.") # User feedback fine
        else:
            logger.info(f"Chunking complete, {len(processed_docs)} chunks created.")
        return processed_docs
    except Exception as e:
        user_message = "An error occurred while chunking documents. This may be due to unexpected document structure."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return []

def index_documents(document_chunks):
    """
    Add document chunks to the in-memory vector store.
    Note: This function modifies st.session_state directly.
    """
    if not document_chunks:
        logger.warning("index_documents called with no chunks to index.")
        st.warning("No document chunks available to index. This may happen if the document was empty or text extraction failed.") # User feedback fine
        return
    logger.info(f"Indexing {len(document_chunks)} document chunks.")
    try:
        st.session_state.DOCUMENT_VECTOR_DB.add_documents(document_chunks)
        st.session_state.document_processed = True
        logger.info("Document chunks indexed successfully into vector store.")
    except Exception as e:
        user_message = "An error occurred while indexing document chunks."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        st.session_state.document_processed = False
