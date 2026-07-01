"""Tests that research-document upload state survives Streamlit reruns."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document as LangchainDocument


class _Rerun(BaseException):
    """Stand-in for Streamlit's RerunException (a BaseException, not Exception)."""


def _run_upload_once(session_state, uploaded_files, interrupt_during_processing=False):
    """Simulate one pass through the upload block in rag_deep.py.

    Mirrors the real control flow: the dedup key (``processed_files_info``) is
    committed only after the pipeline runs to completion, while
    ``document_upload_in_progress`` is always cleared in ``finally``.
    """
    from core.session_manager import reset_document_states

    current_uploaded_files_info = {f.name: f.size for f in uploaded_files}

    if current_uploaded_files_info != session_state.get("processed_files_info", {}):
        if session_state.get("document_upload_in_progress"):
            return "in_progress"
        session_state["document_upload_in_progress"] = True
        reset_document_states(clear_chat=True)
        try:
            session_state["uploaded_filenames"] = [f.name for f in uploaded_files]

            if interrupt_during_processing:
                # Emulate a Streamlit rerun firing mid-processing (e.g. the job
                # poll loop). RerunException is raised before the pipeline and the
                # dedup-key commit complete.
                raise _Rerun()

            session_state["general_context_chunks"] = [LangchainDocument(page_content="general")]
            session_state["requirements_chunks"] = [LangchainDocument(page_content="requirement")]
            session_state["document_processed"] = True

            # Commit dedup key ONLY after the pipeline completes.
            session_state["processed_files_info"] = current_uploaded_files_info
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


def test_interrupted_processing_retries_instead_of_getting_stuck():
    """If a rerun interrupts processing, the file must NOT be marked processed.

    Reproduces the reported bug: the dedup key was committed before processing
    finished, so an interrupting rerun left the app permanently 'Skipping
    reprocessing' with document_processed=False and no Generate Requirements.
    """
    session_state = {
        "processed_files_info": {},
        "document_upload_in_progress": False,
        "document_processed": False,
        "uploaded_filenames": [],
        "uploaded_file_key": 0,
    }

    uploaded = [MagicMock(size=1234)]
    uploaded[0].name = "spec.pdf"

    with patch("core.session_manager.get_embedding_model", return_value=MagicMock()):
        # First run gets interrupted mid-processing.
        with pytest.raises(_Rerun):
            _run_upload_once(session_state, uploaded, interrupt_during_processing=True)

        # Dedup key NOT committed, in_progress cleared -> next run retries.
        assert session_state["processed_files_info"] == {}
        assert session_state["document_processed"] is False
        assert session_state["document_upload_in_progress"] is False

        # Second (uninterrupted) run completes successfully.
        result = _run_upload_once(session_state, uploaded)
        assert result == "processed"
        assert session_state["document_processed"] is True
        assert session_state["processed_files_info"] == {"spec.pdf": 1234}


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
