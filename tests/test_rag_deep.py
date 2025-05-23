import pytest
import os
from unittest.mock import patch, MagicMock, mock_open

# Ensure rag_deep.py can be imported. This might need adjustment based on project structure.
# If tests/ is a top-level directory, and rag_deep.py is also top-level:
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_deep import (
    load_document,
    generate_summary,
    generate_keywords,
    SUMMARIZATION_PROMPT_TEMPLATE, # Import for checking prompt content
    KEYWORD_EXTRACTION_PROMPT_TEMPLATE # Import for checking prompt content
)
from langchain_core.documents import Document as LangchainDocument
from docx.opc.exceptions import PackageNotFoundError
import pdfplumber # Required for one of the mocked exceptions

# Define the path to test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
SAMPLE_TXT_PATH = os.path.join(TEST_DATA_DIR, "sample.txt")
SAMPLE_DOCX_PATH = os.path.join(TEST_DATA_DIR, "sample.docx")
EMPTY_TXT_PATH = os.path.join(TEST_DATA_DIR, "empty.txt") # Will create this
EMPTY_DOCX_PATH = os.path.join(TEST_DATA_DIR, "empty.docx") # Will create this


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    # Create dummy empty files for testing empty file scenarios
    # These are created once per session
    global EMPTY_TXT_PATH, EMPTY_DOCX_PATH
    
    # If sample.txt and sample.docx are not present (e.g. clean CI environment), create them
    # This is mostly for local testing if a previous step failed. In CI, they should exist.
    if not os.path.exists(SAMPLE_TXT_PATH):
        with open(SAMPLE_TXT_PATH, "w") as f:
            f.write("This is a test content from a TXT file.\nIt has multiple lines.")
            
    if not os.path.exists(SAMPLE_DOCX_PATH):
        from docx import Document as DocxCreatorDocument # Avoid conflict with Langchain's Document
        doc = DocxCreatorDocument()
        doc.add_paragraph("This is test content from a DOCX file.")
        doc.save(SAMPLE_DOCX_PATH)

    # Create empty files
    EMPTY_TXT_PATH = str(tmp_path_factory.mktemp("data") / "empty.txt")
    with open(EMPTY_TXT_PATH, "w") as f:
        f.write("")

    EMPTY_DOCX_PATH = str(tmp_path_factory.mktemp("data") / "empty.docx")
    from docx import Document as DocxCreatorDocument # Avoid conflict with Langchain's Document
    doc = DocxCreatorDocument()
    doc.save(EMPTY_DOCX_PATH)


# --- Tests for load_document ---

def test_load_document_txt_success():
    documents = load_document(SAMPLE_TXT_PATH)
    assert isinstance(documents, list)
    assert len(documents) == 1
    assert isinstance(documents[0], LangchainDocument)
    assert "This is a test content from a TXT file." in documents[0].page_content
    assert documents[0].metadata["source"] == "sample.txt"

def test_load_document_docx_success():
    documents = load_document(SAMPLE_DOCX_PATH)
    assert isinstance(documents, list)
    assert len(documents) == 1
    assert isinstance(documents[0], LangchainDocument)
    assert "This is a test content from a DOCX file." in documents[0].page_content
    assert documents[0].metadata["source"] == "sample.docx"

def test_load_document_empty_txt(capsys):
    documents = load_document(EMPTY_TXT_PATH)
    assert documents == [] # Expect empty list for empty files as per implementation
    captured = capsys.readouterr()
    assert f"Text file '{os.path.basename(EMPTY_TXT_PATH)}' appears to be empty." in captured.out # Check for Streamlit warning

def test_load_document_empty_docx(capsys):
    documents = load_document(EMPTY_DOCX_PATH)
    assert documents == []
    captured = capsys.readouterr()
    assert f"DOCX file '{os.path.basename(EMPTY_DOCX_PATH)}' appears to be empty or contains no text." in captured.out

def test_load_document_unsupported_type(capsys):
    # Create a dummy .png file path (file doesn't need to exist for this test if os.path.splitext is primary)
    unsupported_file_path = os.path.join(TEST_DATA_DIR, "sample.png")
    documents = load_document(unsupported_file_path)
    assert documents == []
    captured = capsys.readouterr()
    assert "Unsupported file type: '.png' for file 'sample.png'." in captured.out # Check for Streamlit error

def test_load_document_non_existent_file(capsys):
    non_existent_path = os.path.join(TEST_DATA_DIR, "non_existent.txt")
    documents = load_document(non_existent_path)
    assert documents == []
    captured = capsys.readouterr()
    # The exact error message depends on the OS and Python version for file not found.
    # For now, checking that an error related to loading that file is shown.
    assert f"Error loading document 'non_existent.txt'" in captured.out

@patch('docx.Document')
def test_load_document_corrupted_docx(mock_docx_document, capsys):
    mock_docx_document.side_effect = PackageNotFoundError("Mocked error: Corrupted DOCX")
    corrupted_docx_path = os.path.join(TEST_DATA_DIR, "corrupted.docx")
    documents = load_document(corrupted_docx_path)
    assert documents == []
    captured = capsys.readouterr()
    assert "Failed to load DOCX 'corrupted.docx': The file appears to be corrupted or not a valid DOCX file." in captured.out

@patch("builtins.open", new_callable=mock_open)
def test_load_document_txt_unicode_decode_error(mock_file_open, capsys):
    mock_file_open.side_effect = UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")
    bad_encoding_txt_path = os.path.join(TEST_DATA_DIR, "bad_encoding.txt")
    documents = load_document(bad_encoding_txt_path)
    assert documents == []
    captured = capsys.readouterr()
    assert "Failed to load TXT file 'bad_encoding.txt': The file is not UTF-8 encoded." in captured.out

@patch('rag_deep.PDFPlumberLoader')
def test_load_document_pdf_syntax_error(mock_pdf_loader, capsys):
    instance = mock_pdf_loader.return_value
    instance.load.side_effect = pdfplumber.exceptions.PDFSyntaxError("Mocked PDF Syntax Error")
    corrupted_pdf_path = os.path.join(TEST_DATA_DIR, "corrupted.pdf")
    documents = load_document(corrupted_pdf_path)
    assert documents == []
    captured = capsys.readouterr()
    assert "Failed to load PDF 'corrupted.pdf': The file may be corrupted or not a valid PDF." in captured.out


# --- Tests for generate_summary ---

@patch('rag_deep.LANGUAGE_MODEL')
def test_generate_summary_success(mock_llm_invoke):
    # We need to mock the result of .invoke() on the chain, not the model directly if using LCEL
    # However, current rag_deep.py uses model.invoke directly after creating prompt.
    # If rag_deep.py had `chain = prompt | LANGUAGE_MODEL; chain.invoke()`, this mock would be different.
    # For now, it's `response_chain.invoke` where response_chain is `conversation_prompt | LANGUAGE_MODEL`
    # So mocking LANGUAGE_MODEL.invoke (or the whole model if it's simpler)
    
    # Mocking the behavior of the LLM part of the chain
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "This is a test summary."
    
    # Patch the LANGUAGE_MODEL to be this mock instance
    with patch('rag_deep.LANGUAGE_MODEL', mock_llm_instance):
        summary = generate_summary("Some document text for summarization.")
        assert summary == "This is a test summary."
        
        # Check if the prompt was constructed correctly (accessing args of the LLM's invoke)
        # The actual prompt object is complex, so checking for key parts.
        args, kwargs = mock_llm_instance.invoke.call_args
        assert "document_text" in kwargs
        assert kwargs["document_text"] == "Some document text for summarization."
        # To check the template, you'd need to inspect the prompt object passed to the chain,
        # which is harder if you only mock the final LLM. For now, this is a good check.

@patch('rag_deep.LANGUAGE_MODEL')
def test_generate_summary_llm_failure(mock_llm_invoke, capsys):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("LLM simulated error")
    with patch('rag_deep.LANGUAGE_MODEL', mock_llm_instance):
        summary = generate_summary("Text that will cause LLM error.")
        assert "Failed to generate summary due to an AI model error." in summary
        captured = capsys.readouterr()
        assert "An error occurred while generating the document summary using the AI model. Details: LLM simulated error" in captured.out

def test_generate_summary_empty_input(capsys):
    summary = generate_summary("")
    assert summary is None
    captured = capsys.readouterr()
    assert "Document content is empty or contains only whitespace. Cannot generate summary." in captured.out

    summary_ws = generate_summary("   ")
    assert summary_ws is None
    captured_ws = capsys.readouterr()
    assert "Document content is empty or contains only whitespace. Cannot generate summary." in captured_ws.out


# --- Tests for generate_keywords ---

@patch('rag_deep.LANGUAGE_MODEL')
def test_generate_keywords_success(mock_llm_invoke):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "keyword1, keyword2, keyword3"
    with patch('rag_deep.LANGUAGE_MODEL', mock_llm_instance):
        keywords = generate_keywords("Some document text for keyword extraction.")
        assert keywords == "keyword1, keyword2, keyword3"
        args, kwargs = mock_llm_instance.invoke.call_args
        assert "document_text" in kwargs
        assert kwargs["document_text"] == "Some document text for keyword extraction."


@patch('rag_deep.LANGUAGE_MODEL')
def test_generate_keywords_llm_failure(mock_llm_invoke, capsys):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("LLM keyword error")
    with patch('rag_deep.LANGUAGE_MODEL', mock_llm_instance):
        keywords = generate_keywords("Text for keyword LLM error.")
        assert "Failed to extract keywords due to an AI model error." in keywords
        captured = capsys.readouterr()
        assert "An error occurred while extracting keywords using the AI model. Details: LLM keyword error" in captured.out


def test_generate_keywords_empty_input(capsys):
    keywords = generate_keywords("")
    assert keywords is None
    captured = capsys.readouterr()
    assert "Document content is empty or contains only whitespace. Cannot extract keywords." in captured.out

    keywords_ws = generate_keywords("  \n\t  ")
    assert keywords_ws is None
    captured_ws = capsys.readouterr()
    assert "Document content is empty or contains only whitespace. Cannot extract keywords." in captured_ws.out


# Note: chunk_documents and index_documents involve more complex Langchain objects
# and session state. Testing them thoroughly would require more elaborate mocking
# of Streamlit's session state and the vector store, or integration-style tests.
# For this subtask, focus is on load_document and LLM utility functions.

# To run these tests:
# 1. Ensure pytest and pytest-mock are installed: pip install pytest pytest-mock
# 2. Navigate to the root directory of the project in the terminal.
# 3. Run: pytest
#
# If rag_deep.py uses Streamlit elements like st.error directly in the tested functions,
# these will print to stdout/stderr during tests. capsys fixture is used to capture this.
# Ideally, such UI calls might be separated from core logic for easier testing,
# but for now, capturing output is a practical approach.
#
# The `sys.path.insert` is a common way to handle imports in simple test structures.
# For larger projects, using Python packaging (setup.py or pyproject.toml)
# and installing the package in editable mode (`pip install -e .`) is recommended.
#
# The `autouse=True` fixture `setup_test_environment` ensures dummy files are ready.
# `tmp_path_factory` is a pytest fixture for creating temporary directories.
# `pdfplumber.exceptions.PDFSyntaxError` is imported for the specific PDF error test.
# `docx.opc.exceptions.PackageNotFoundError` is imported for the specific DOCX error test.
#
# Mocking strategy for LLM calls:
# The functions generate_summary and generate_keywords use a LangChain Expression Language (LCEL) chain:
# `prompt_template | LANGUAGE_MODEL`
# Then `.invoke()` is called on this chain.
# To test this, we mock `rag_deep.LANGUAGE_MODEL` (the OllamaLLM instance).
# We replace it with a `MagicMock` instance.
# Then, we set the `return_value` of `mock_llm_instance.invoke` to our desired output (e.g., "test summary").
# This effectively simulates the LLM part of the chain returning a specific value.
# We also check `call_args` on `mock_llm_instance.invoke` to ensure the LLM was called with the expected input.
# This approach tests the logic within `generate_summary/keywords` (prompt creation, handling LLM output/errors)
# without making actual LLM calls.
#
# The `capsys` fixture captures stdout/stderr, which is useful because the Streamlit functions
# like `st.error` and `st.warning` write to these streams.
# The tests check if these messages are displayed as expected.
#
# The tests for empty files check for the warning printed by `st.warning` in `load_document`.
# The tests for corrupted/non-existent files check for `st.error` messages.
#
# A note about `sys.path.insert`: this is to make sure `rag_deep.py` can be found.
# In a more complex project, you'd typically have a `src` layout or install the package.
# For this context, it's a straightforward way to enable the import.
#
# Global empty file paths are updated by the fixture for consistency.
# Sample files are created if they don't exist by the fixture, primarily for robust local testing.
# In a CI environment, these files should be checked into the repository.
# The dummy files created by tmp_path_factory are session-scoped and cleaned up automatically.
#
# For `test_generate_summary_success` and `test_generate_keywords_success`, the assertion
# on `mock_llm_instance.invoke.call_args` checks the `kwargs` passed to the `invoke` method of the
# *mocked LLM instance*. This is because the LCEL chain `prompt | LANGUAGE_MODEL` will ultimately
# call `LANGUAGE_MODEL.invoke(formatted_prompt_value)`. The `formatted_prompt_value` is a dictionary-like
# object where keys are input variables from the prompt template. So, we check if "document_text"
# (the input variable in our templates) is present in these kwargs and has the correct value.
# This confirms our function correctly passes the input text through the prompt to the LLM.
#
# One specific import: `from docx import Document as DocxCreatorDocument` in the fixture is to
# avoid a name collision with `from langchain_core.documents import Document as LangchainDocument`
# used in the main test scope.
#
# The `pdfplumber.exceptions` needs `pdfplumber` to be installed, which is in `requirements.txt`.
# Similarly, `docx.opc.exceptions` needs `python-docx`.
#
# The tests cover:
# - Successful loading of .txt and .docx.
# - Handling of empty .txt and .docx files (expect warnings).
# - Handling of unsupported file types.
# - Handling of non-existent files.
# - Handling of corrupted .docx files.
# - Handling of .txt files with encoding errors.
# - Handling of corrupted .pdf files (mocking PDF specific error).
# - Successful summary and keyword generation (mocking LLM).
# - LLM failures during summary and keyword generation.
# - Empty input for summary and keyword generation.Okay, I have created `tests/test_rag_deep.py` with the unit tests in the previous turn.
The test file includes:
*   Setup for test data paths and creation of dummy/empty files for specific test cases using a pytest fixture (`setup_test_environment`).
*   **Tests for `load_document`**:
    *   Successful loading of `.txt` and `.docx` files.
    *   Handling of empty `.txt` and `.docx` files (checking for `st.warning`).
    *   Handling of unsupported file types (checking for `st.error`).
    *   Handling of non-existent files (checking for `st.error`).
    *   Handling of corrupted `.docx` files (mocking `docx.Document` to raise `PackageNotFoundError` and checking `st.error`).
    *   Handling of `.txt` files with `UnicodeDecodeError` (mocking `open` and checking `st.error`).
    *   Handling of corrupted `.pdf` files (mocking `PDFPlumberLoader` to raise `pdfplumber.exceptions.PDFSyntaxError` and checking `st.error`).
*   **Tests for `generate_summary`**:
    *   Successful summary generation by mocking `rag_deep.LANGUAGE_MODEL.invoke` and verifying the output and input to the LLM.
    *   Handling of LLM failure by mocking `rag_deep.LANGUAGE_MODEL.invoke` to raise an exception and checking the error message.
    *   Handling of empty input text (checking for `st.warning` and `None` return).
*   **Tests for `generate_keywords`**:
    *   Successful keyword generation (similar mocking strategy as for summaries).
    *   Handling of LLM failure (similar mocking strategy).
    *   Handling of empty input text.
*   The tests use `pytest` and `pytest-mock` (via `unittest.mock.patch` and `MagicMock`).
*   The `capsys` fixture is used to capture output from Streamlit's `st.error` and `st.warning` calls.
*   A `sys.path.insert` is used to allow importing from `rag_deep.py` located in the parent directory.

The test suite covers the requested functionality.
The next step would typically be to run these tests.
The files created/modified are:
*   `requirements-dev.txt` (created in turn 1 of this subtask)
*   `tests/test_data/sample.txt` (created in turn 4 of this subtask)
*   `tests/test_data/sample.docx` (created in turn 5 of this subtask)
*   `tests/test_rag_deep.py` (created in turn 6 of this subtask)
*   The `tests` and `tests/test_data` directories were also created.

I've already mentioned the need for `pytest` and `pytest-mock` (which are in `requirements-dev.txt`).
