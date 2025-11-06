import streamlit as st
import pandas as pd
import json
import io
import re
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

conversation_prompt = ChatPromptTemplate.from_template(GENERAL_QA_PROMPT_TEMPLATE)

def generate_answer(
    language_model,
    user_query,
    context_documents,
    conversation_history="",
    persistent_memory="",
):
    if not user_query or not user_query.strip():
        logger.warning("generate_answer called with empty user_query.")
        return "Your question is empty. Please type a question to get an answer."

    if not context_documents or not isinstance(context_documents, list):
        logger.warning("generate_answer called with no context documents.")
        return ("I couldn't find relevant information in the document to answer your query. "
                "Please try rephrasing your question or ensure the document contains the relevant topics.")

    logger.info(f"Generating answer for query: '{user_query[:50]}...'")
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        if not context_text.strip():
            logger.warning("Context text for answer generation is empty after joining docs.")
            return ("The relevant sections found in the document appear to be empty. "
                    "Cannot generate an answer.")

        # Always returns a str (no AIMessage)
        response_chain = conversation_prompt | language_model | StrOutputParser()

        response: str = response_chain.invoke(
            {
                "user_query": user_query,
                "document_context": context_text,
                "conversation_history": conversation_history,
                "persistent_memory": persistent_memory,
            }
        )

        response = (response or "").strip()
        if not response:
            logger.warning("AI model returned an empty response for answer generation.")
            return ("The AI model returned an empty response. Please rephrase your question or try again later.")

        logger.info("Answer generated successfully.")
        return response

    except Exception as e:
        user_message = "I'm sorry, but I encountered an error while trying to generate a response."
        logger.exception(f"Error during answer generation: {e}")
        return f"{user_message} Please try again later or rephrase your question. (Details: {e})"
    
_req_prompt = ChatPromptTemplate.from_template(REQUIREMENT_JSON_PROMPT_TEMPLATE)
# build the chain once at module load (reuse it)
# you may want to inject language_model later if it’s created lazily
def _req_chain(language_model):
    return _req_prompt | language_model | StrOutputParser()

def generate_requirements_json(language_model, requirement_chunk,
                               verification_methods_context: str = "",
                               general_context: str = ""):
    logger.info("Generating requirements JSON...")
    try:
        context_text = getattr(requirement_chunk, "page_content", "") or ""
        if not context_text.strip():
            logger.warning("generate_requirements_json called with empty chunk text.")
            return "{}"

        response = _req_chain(language_model).invoke({
            "document_context": context_text,
            "verification_methods_context": verification_methods_context or "",
            "general_context": general_context or "",
        })  # response is str

        cleaned = (response or "").strip()
        if not cleaned:
            logger.warning("AI model returned an empty response for requirements JSON generation.")
            return "{}"

        m = re.search(r"\{[\s\S]*\}", cleaned)
        json_candidate = m.group(0) if m else cleaned

        try:
            json.loads(json_candidate)
            logger.info("Requirements JSON generated successfully.")
            return json_candidate
        except Exception:
            logger.warning("Model output was not valid JSON; returning raw cleaned text.")
            return cleaned

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


from langchain_core.output_parsers import StrOutputParser

classification_prompt = ChatPromptTemplate.from_template(CHUNK_CLASSIFICATION_PROMPT_TEMPLATE)
# build once and reuse:
# classification_chain = classification_prompt | language_model | StrOutputParser()

def classify_chunk(language_model, chunk_text):
    if not chunk_text or not chunk_text.strip():
        logger.warning("classify_chunk called with empty chunk_text.")
        return "Requirements"

    logger.debug(f"Classifying chunk: '{chunk_text[:50]}...'")
    try:
        chain = classification_prompt | language_model | StrOutputParser()
        response = chain.invoke({"chunk_text": chunk_text})   # response is str now

        cleaned_response = (response or "").strip()
        if "General Context" in cleaned_response:
            return "General Context"
        if "Requirements" in cleaned_response:
            return "Requirements"

        logger.warning(f"Unexpected response from AI model during chunk classification: '{cleaned_response}'")
        return "Requirements"
    except Exception as e:
        logger.exception(f"Error during chunk classification: {e}")
        return "Requirements"


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
