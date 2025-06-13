import pytest
from unittest.mock import patch, MagicMock, ANY

# Import LangchainDocument for creating test data
from langchain_core.documents import Document as LangchainDocument

# Modules to test
from core.generation import (
    generate_answer,
    generate_summary,
    generate_keywords
)

# Import config to use its prompt templates, and logger for mocking
from core import config
# core.logger_config is not directly imported here, but we'll patch the logger instance
# used within generation.py, which is 'core.generation.logger'

# Paths for mocking
STREAMLIT_ERROR_PATH = 'core.generation.st.error'
STREAMLIT_WARNING_PATH = 'core.generation.st.warning'
GENERATION_LOGGER_PATH = 'core.generation.logger'
CHAT_PROMPT_TEMPLATE_PATH = 'core.generation.ChatPromptTemplate.from_template'

# --- Fixtures ---

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "Mocked LLM response"
    return llm

@pytest.fixture
def mock_doc():
    return lambda content, source="test_source": LangchainDocument(page_content=content, metadata={"source": source})

@pytest.fixture(autouse=True)
def mock_logger_fixture():
    """Automatically mock the logger for all tests in this file."""
    with patch(GENERATION_LOGGER_PATH) as mock_log:
        yield mock_log

# --- Tests for generate_answer ---

@patch(CHAT_PROMPT_TEMPLATE_PATH)
def test_generate_answer_success_with_history(mock_from_template, mock_llm, mock_doc, mock_logger_fixture):
    mock_template_instance = MagicMock()
    mock_chain_instance = MagicMock()
    # Simulate the LCEL chain: prompt | llm -> chain
    # When `mock_template_instance | mock_llm` is executed, it should return our mock_chain_instance
    mock_template_instance.__or__ = MagicMock(return_value=mock_chain_instance)

    mock_chain_instance.invoke.return_value = "Successful answer"
    mock_from_template.return_value = mock_template_instance

    user_query = "What is X?"
    context_docs = [mock_doc("X is Y."), mock_doc("More about X.")]
    history = "User: Tell me about Z\nAssistant: Z is A"

    response = generate_answer(mock_llm, user_query, context_docs, history)

    mock_from_template.assert_called_once_with(config.PROMPT_TEMPLATE)
    # Check that the template was "piped" with the llm
    mock_template_instance.__or__.assert_called_once_with(mock_llm)

    expected_doc_context = "X is Y.\n\nMore about X."
    mock_chain_instance.invoke.assert_called_once_with({
        "user_query": user_query,
        "document_context": expected_doc_context,
        "conversation_history": history
    })
    assert response == "Successful answer"
    mock_logger_fixture.info.assert_any_call(f"Generating answer for query: '{user_query[:50]}...'")
    mock_logger_fixture.info.assert_any_call("Answer generated successfully.")


def test_generate_answer_llm_failure(mock_llm, mock_doc, mock_logger_fixture):
    mock_llm.invoke.side_effect = Exception("LLM exploded")

    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        response = generate_answer(mock_llm, "Query", [mock_doc("Content")], "History")

    assert "I'm sorry, but I encountered an error" in response
    assert "(Details: LLM exploded)" in response # Check if detail is part of the returned user message
    mock_st_error.assert_called_once() # This st.error is called from within generate_answer
    mock_logger_fixture.exception.assert_called_once()


def test_generate_answer_empty_query(mock_llm, mock_doc, mock_logger_fixture):
    response = generate_answer(mock_llm, "", [mock_doc("Content")])
    assert response == "Your question is empty. Please type a question to get an answer."
    mock_logger_fixture.warning.assert_called_once_with("generate_answer called with empty user_query.")


def test_generate_answer_empty_context_docs(mock_llm, mock_logger_fixture):
    response = generate_answer(mock_llm, "Query", [])
    assert response == "I couldn't find relevant information in the document to answer your query. Please try rephrasing your question or ensure the document contains the relevant topics."
    mock_logger_fixture.warning.assert_called_once_with("generate_answer called with no context documents.")


def test_generate_answer_context_docs_no_content(mock_llm, mock_doc, mock_logger_fixture):
    response = generate_answer(mock_llm, "Query", [mock_doc("   ")]) # Content is whitespace
    assert response == "The relevant sections found in the document appear to be empty. Cannot generate an answer."
    mock_logger_fixture.warning.assert_called_once_with("Context text for answer generation is empty after joining docs.")


# --- Tests for generate_summary ---

@patch(CHAT_PROMPT_TEMPLATE_PATH)
def test_generate_summary_success(mock_from_template, mock_llm, mock_logger_fixture):
    mock_template_instance = MagicMock()
    mock_chain_instance = MagicMock()
    mock_template_instance.__or__ = MagicMock(return_value=mock_chain_instance)
    mock_chain_instance.invoke.return_value = "Document summary."
    mock_from_template.return_value = mock_template_instance

    text = "This is a long document text."
    summary = generate_summary(mock_llm, text)

    mock_from_template.assert_called_once_with(config.SUMMARIZATION_PROMPT_TEMPLATE)
    mock_template_instance.__or__.assert_called_once_with(mock_llm)
    mock_chain_instance.invoke.assert_called_once_with({"document_text": text})
    assert summary == "Document summary."
    mock_logger_fixture.info.assert_any_call("Generating summary...")
    mock_logger_fixture.info.assert_any_call("Summary generated successfully.")


def test_generate_summary_llm_failure(mock_llm, mock_logger_fixture):
    mock_llm.invoke.side_effect = Exception("Summarizer exploded")
    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        summary = generate_summary(mock_llm, "Some text")

    assert "Failed to generate summary due to an AI model error." in summary
    assert "(Details: Summarizer exploded)" in summary
    mock_st_error.assert_called_once()
    mock_logger_fixture.exception.assert_called_once()


@patch(STREAMLIT_WARNING_PATH)
def test_generate_summary_empty_input(mock_st_warning, mock_llm, mock_logger_fixture):
    summary = generate_summary(mock_llm, "   ")
    assert summary is None
    mock_st_warning.assert_called_once_with("Document content is empty or contains only whitespace. Cannot generate summary.")
    mock_logger_fixture.warning.assert_called_once_with("generate_summary called with empty document text.")


@patch(STREAMLIT_WARNING_PATH)
def test_generate_summary_llm_returns_empty(mock_st_warning, mock_llm, mock_logger_fixture):
    mock_llm.invoke.return_value = "   "
    summary = generate_summary(mock_llm, "Some text")
    assert summary is None
    mock_st_warning.assert_called_once_with("The AI model returned an empty summary. The document might be too short or lack clear content for summarization.")
    mock_logger_fixture.warning.assert_called_once_with("AI model returned an empty summary.")


# --- Tests for generate_keywords ---

@patch(CHAT_PROMPT_TEMPLATE_PATH)
def test_generate_keywords_success(mock_from_template, mock_llm, mock_logger_fixture):
    mock_template_instance = MagicMock()
    mock_chain_instance = MagicMock()
    mock_template_instance.__or__ = MagicMock(return_value=mock_chain_instance)
    mock_chain_instance.invoke.return_value = "keyword1, keyword2"
    mock_from_template.return_value = mock_template_instance

    text = "This is a document about keywords."
    keywords = generate_keywords(mock_llm, text)

    mock_from_template.assert_called_once_with(config.KEYWORD_EXTRACTION_PROMPT_TEMPLATE)
    mock_template_instance.__or__.assert_called_once_with(mock_llm)
    mock_chain_instance.invoke.assert_called_once_with({"document_text": text})
    assert keywords == "keyword1, keyword2"
    mock_logger_fixture.info.assert_any_call("Generating keywords...")
    mock_logger_fixture.info.assert_any_call("Keywords generated successfully.")


def test_generate_keywords_llm_failure(mock_llm, mock_logger_fixture):
    mock_llm.invoke.side_effect = Exception("Keyword extractor exploded")
    with patch(STREAMLIT_ERROR_PATH) as mock_st_error:
        keywords = generate_keywords(mock_llm, "Some text")

    assert "Failed to extract keywords due to an AI model error." in keywords
    assert "(Details: Keyword extractor exploded)" in keywords
    mock_st_error.assert_called_once()
    mock_logger_fixture.exception.assert_called_once()


@patch(STREAMLIT_WARNING_PATH)
def test_generate_keywords_empty_input(mock_st_warning, mock_llm, mock_logger_fixture):
    keywords = generate_keywords(mock_llm, "")
    assert keywords is None
    mock_st_warning.assert_called_once_with("Document content is empty or contains only whitespace. Cannot extract keywords.")
    mock_logger_fixture.warning.assert_called_once_with("generate_keywords called with empty document text.")


@patch(STREAMLIT_WARNING_PATH)
def test_generate_keywords_llm_returns_empty(mock_st_warning, mock_llm, mock_logger_fixture):
    mock_llm.invoke.return_value = ""
    keywords = generate_keywords(mock_llm, "Some text")
    assert keywords is None
    mock_st_warning.assert_called_once_with("The AI model returned no keywords. The document might be too short or lack distinct terms.")
    mock_logger_fixture.warning.assert_called_once_with("AI model returned no keywords.")
