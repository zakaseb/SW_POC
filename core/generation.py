import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from .config import (
    GENERAL_QA_PROMPT_TEMPLATE,
    REQUIREMENT_JSON_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    KEYWORD_EXTRACTION_PROMPT_TEMPLATE,
    CHUNK_CLASSIFICATION_PROMPT_TEMPLATE,
)
from .logger_config import get_logger

logger = get_logger(__name__)


def generate_answer(
    language_model,
    user_query,
    context_documents,
    conversation_history="",
    persistent_memory="",
):
    """
    Generate an answer based on the user query, context documents, and conversation history.
    """
    if not user_query or not user_query.strip():
        logger.warning("generate_answer called with empty user_query.")
        return "Your question is empty. Please type a question to get an answer."

    if not context_documents or not isinstance(context_documents, list) or len(context_documents) == 0:
        logger.warning("generate_answer called with no context documents.")
        return "I couldn't find relevant information in the document to answer your query. Please try rephrasing your question or ensure the document contains the relevant topics."

    logger.info(f"Generating answer for query: '{user_query[:50]}...'")
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        if not context_text.strip():
            logger.warning("Context text for answer generation is empty after joining docs.")
            return "The relevant sections found in the document appear to be empty. Cannot generate an answer."

        logger.debug(f"Context for prompt: {context_text[:100]}...")
        conversation_prompt = ChatPromptTemplate.from_template(GENERAL_QA_PROMPT_TEMPLATE)
        response_chain = conversation_prompt | language_model

        response = response_chain.invoke(
            {
                "user_query": user_query,
                "document_context": context_text,
                "conversation_history": conversation_history,
                "persistent_memory": persistent_memory,
            }
        )

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for answer generation.")
            return "The AI model returned an empty response. Please try rephrasing your question or try again later."
        logger.info("Answer generated successfully.")
        return response
    except Exception as e:
        user_message = "I'm sorry, but I encountered an error while trying to generate a response."
        logger.exception(f"Error during answer generation: {e}")
        return f"{user_message} Please try again later or rephrase your question. (Details: {e})"


def generate_requirements_json(language_model, requirement_chunk):
    """
    Generates a JSON object for a single requirement chunk.
    """
    logger.info("Generating requirements JSON...")
    try:
        context_text = requirement_chunk.page_content
        if not context_text.strip():
            logger.warning("generate_requirements_json called with empty chunk text.")
            return "{}"

        prompt = ChatPromptTemplate.from_template(REQUIREMENT_JSON_PROMPT_TEMPLATE)
        response_chain = prompt | language_model

        response = response_chain.invoke({"document_context": context_text})

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for requirements JSON generation.")
            return "{}"
        logger.info("Requirements JSON generated successfully.")
        return response
    except Exception as e:
        user_message = "I'm sorry, but I encountered an error while trying to generate the requirements JSON."
        logger.exception(f"Error during requirements JSON generation: {e}")
        return f"{{ 'error': '{user_message}', 'details': '{e}' }}"


def generate_summary(language_model, full_document_text):
    """
    Generates a summary for the given document text.
    """
    if not full_document_text or not full_document_text.strip():
        logger.warning("generate_summary called with empty document text.")
        st.warning(
            "Document content is empty or contains only whitespace. Cannot generate summary."
        )
        return None
    logger.info("Generating summary...")
    try:
        summary_prompt = ChatPromptTemplate.from_template(SUMMARIZATION_PROMPT_TEMPLATE)
        summary_chain = summary_prompt | language_model
        summary = summary_chain.invoke({"document_text": full_document_text})
        if not summary or not summary.strip():
            logger.warning("AI model returned an empty summary.")
            st.warning(
                "The AI model returned an empty summary. The document might be too short or lack clear content for summarization."
            )
            return None
        logger.info("Summary generated successfully.")
        return summary
    except Exception as e:
        user_message = "Failed to generate summary due to an AI model error."
        logger.exception(f"Error during summary generation: {e}")
        st.error(
            f"An error occurred while generating the document summary using the AI model. Details: {e}"
        )
        return f"{user_message} Please try again later. (Details: {e})"


def classify_chunk(language_model, chunk_text):
    """
    Classifies a text chunk as 'General Context' or 'Requirements'.
    """
    if not chunk_text or not chunk_text.strip():
        logger.warning("classify_chunk called with empty chunk_text.")
        return "Requirements"  # Default classification

    logger.debug(f"Classifying chunk: '{chunk_text[:50]}...'")
    try:
        classification_prompt = ChatPromptTemplate.from_template(
            CHUNK_CLASSIFICATION_PROMPT_TEMPLATE
        )
        classification_chain = classification_prompt | language_model
        response = classification_chain.invoke({"chunk_text": chunk_text})

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for chunk classification.")
            return "Requirements"  # Default classification

        #The model's output might have newlines or extra spaces.
        cleaned_response = response.strip()

        if "General Context" in cleaned_response:
            logger.info("Chunk classified as General Context.")
            return "General Context"
        elif "Requirements" in cleaned_response:
            logger.info("Chunk classified as Requirements.")
            return "Requirements"
        else:
            logger.warning(
                f"Unexpected response from AI model during chunk classification: '{cleaned_response}'"
            )
            return "Requirements"  # Default to 'Requirements'

    except Exception as e:
        logger.exception(f"Error during chunk classification: {e}")
        # In case of an error, default to 'Requirements' to be safe.
        return "Requirements"


def generate_keywords(language_model, full_document_text):
    """
    Generates keywords for the given document text.
    """
    if not full_document_text or not full_document_text.strip():
        logger.warning("generate_keywords called with empty document text.")
        st.warning(
            "Document content is empty or contains only whitespace. Cannot extract keywords."
        )
        return None
    logger.info("Generating keywords...")
    try:
        keywords_prompt = ChatPromptTemplate.from_template(
            KEYWORD_EXTRACTION_PROMPT_TEMPLATE
        )
        keywords_chain = keywords_prompt | language_model
        keywords = keywords_chain.invoke({"document_text": full_document_text})
        if not keywords or not keywords.strip():
            logger.warning("AI model returned no keywords.")
            st.warning(
                "The AI model returned no keywords. The document might be too short or lack distinct terms."
            )
            return None
        logger.info("Keywords generated successfully.")
        return keywords
    except Exception as e:
        user_message = "Failed to extract keywords due to an AI model error."
        logger.exception(f"Error during keyword generation: {e}")
        st.error(
            f"An error occurred while extracting keywords using the AI model. Details: {e}"
        )
        return f"{user_message} Please try again later. (Details: {e})"
