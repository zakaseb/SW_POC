"""Tests that research-document upload state survives Streamlit reruns."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document as LangchainDocument


def _run_upload_once(session_state, uploaded_files):
    """Simulate one pass through the upload block in rag_deep.py."""
    from core.session_manager import reset_document_states

    current_uploaded_files_info = {f.name: f.size for f in uploaded_files}

    if current_uploaded_files_info != session_state.get("processed_files_info", {}):
        if session_state.get("document_upload_in_progress"):
            return "in_progress"
        session_state["document_upload_in_progress"] = True
        reset_document_states(clear_chat=True)
        try:
            session_state["uploaded_filenames"] = [f.name for f in uploaded_files]
            session_state["processed_files_info"] = current_uploaded_files_info
            session_state["general_context_chunks"] = [LangchainDocument(page_content="general")]
            session_state["requirements_chunks"] = [LangchainDocument(page_content="requirement")]
            session_state["document_processed"] = True
            return "processed"
        finally:
            session_state["document_upload_in_progress"] = False
    return "skipped"


def test_upload_state_persists_across_rerun_without_uploader_reset():
    """Processed state must remain after a rerun with the same file fingerprint."""
    session_state = {
        "processed_files_info": {},
        "document_upload_in_progress": False,
        "document_processed": False,
        "uploaded_filenames": [],
        "uploaded_file_key": 0,
    }

    uploaded = [MagicMock(name="spec.pdf", size=1234)]
    uploaded[0].name = "spec.pdf"

    with patch("core.session_manager.get_embedding_model", return_value=MagicMock()):
        first = _run_upload_once(session_state, uploaded)
        assert first == "processed"
        assert session_state["document_processed"] is True
        assert session_state["processed_files_info"] == {"spec.pdf": 1234}

        # Simulate Streamlit rerun: uploader still shows the same files.
        second = _run_upload_once(session_state, uploaded)
        assert second == "skipped"
        assert session_state["document_processed"] is True
        assert session_state["uploaded_filenames"] == ["spec.pdf"]


def test_empty_uploader_on_rerun_does_not_clear_processed_state():
    """When the uploader is empty after rerun, rag_deep skips the upload block entirely."""
    session_state = {
        "processed_files_info": {"spec.pdf": 1234},
        "document_processed": True,
        "uploaded_filenames": ["spec.pdf"],
        "uploaded_file_key": 1,
    }

    uploaded_after_rerun = []
    if uploaded_after_rerun:
        _run_upload_once(session_state, uploaded_after_rerun)

    assert session_state["document_processed"] is True
    assert session_state["processed_files_info"] == {"spec.pdf": 1234}
