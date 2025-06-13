[![CodeQL Advanced](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml)
[![Dependency Review](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml)

# DocuMind-AI: Intelligent Document Assistant

**DocuMind-AI** is an AI-powered document assistant that allows users to upload documents (PDF, DOCX, TXT), process their contents, and interactively ask questions about them. Built with Streamlit and LangChain, it leverages embeddings and language models to provide concise, factual answers, summaries, and keyword extractions based on the document's context. This tool is ideal for researchers, students, and professionals who need to quickly extract insights from various document formats.

## Features

- **Multi-Format Document Upload:**  
  Upload one or more research papers, reference materials, or any PDF, DOCX, or TXT documents directly through the interface. The application processes all uploaded documents simultaneously and automatically extracts text for processing from these formats.

- **Document Processing:**  
  The assistant extracts text from all uploaded documents, splits it into manageable chunks, and indexes the combined content using embeddings. This enables efficient retrieval and querying of information across all provided documents.

- **Intelligent Querying with Hybrid Search, Re-ranking & Conversation History:**  
  Ask questions about the document's content and receive concise, contextually relevant answers. The AI uses a language model to generate responses based on the document's context. It employs a sophisticated retrieval pipeline:
    1.  A **hybrid search** strategy combines semantic understanding (vector search) with BM25 keyword matching.
    2.  Results are fused using **Reciprocal Rank Fusion (RRF)**.
    3.  The top RRF results are then **re-ranked** using a CrossEncoder model (`ms-marco-MiniLM-L-6-v2`) to further refine relevance before being passed to the LLM.
  Recent conversation history (last 3 turns) is also considered to better understand follow-up questions.

- **Content Summarization:**
  Generate a concise summary of the combined textual content from all uploaded documents with a single click. The summary is displayed in the sidebar, providing a quick overview.

- **Keyword Extraction:**  
  Extract key phrases and terms from the combined content of all uploaded documents. These keywords are displayed in the sidebar, helping to identify core topics across the documents.

- **Chat-Based UI:**  
  Interact with the assistant via a user-friendly chat interface, making it easy to ask follow-up questions or explore different aspects of the document.

- **Enhanced User Controls (Sidebar):**
  - **Clear Chat History:** Easily clear the current conversation.
  - **Reset All Documents & Chat:** Remove all currently loaded documents and chat history, allowing you to start fresh with new files.
  - **Summarize Uploaded Content:** Generate a summary from the combined text of all processed documents.
  - **Extract Keywords from Content:** Extract keywords from the combined text of all processed documents.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **Ollama** (for running the Deepseek model locally)
- **Git** (for cloning the repository)

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/chintanboghara/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Set Up a Virtual Environment (Recommended)

Using a virtual environment helps isolate project dependencies and avoid conflicts with other Python projects.

#### On Windows

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - **Command Prompt:**

     ```bash
     venv\Scripts\activate
     ```

   - **PowerShell:**

     ```bash
     .\venv\Scripts\Activate.ps1
     ```

   _If you encounter an execution policy error in PowerShell, run:_

   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

#### On Ubuntu

1. **Install Python3 and Virtual Environment Tools (if not installed):**

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment:**

   ```bash
   source venv/bin/activate
   ```

### 3. Install Dependencies

With the virtual environment activated, install the required Python packages using the `requirements.txt` file. This file includes all necessary libraries, such as Streamlit, LangChain, and `python-docx` for handling various document formats.

```bash
pip install -r requirements.txt
```

### 4. Create Required Directories (Default Path)

By default, the application uses `document_store/pdfs/` for storing uploaded documents temporarily during processing. If this path is not configured differently via environment variables (see "Environment Variables" section below), ensure the directory exists. If not, create it:

- **Windows:**

  ```bash
  mkdir document_store\pdfs
  ```

- **Ubuntu:**

  ```bash
  mkdir -p document_store/pdfs
  ```

### 5. Setup Ollama and Download the Deepseek Model

DocuMind-AI uses the Deepseek model through Ollama for embeddings and language model capabilities. Follow these steps:

1. **Install Ollama:**
   - Download and install Ollama from the [Ollama website](https://ollama.com).
   - Ensure that Ollama is running locally before starting the application. You can start Ollama by running `ollama serve` in a separate terminal window.

2. **Download the Deepseek Model:**
   - With Ollama running, download the Deepseek model by executing:

     ```bash
     ollama pull deepseek-r1:1.5b
     ```

   This command downloads the Deepseek model (version 1.5b by default, see Environment Variables to customize) locally, which is required for generating embeddings and language model responses. Ensure you pull any specific models you configure via environment variables.

### 6. Run the Application

Start the Streamlit application with the following command:

```bash
streamlit run rag_deep.py
```

This will launch a local web server, typically at [http://localhost:8501](http://localhost:8501), where you can interact with the assistant.

## Usage

1. **Upload Document(s):**
   - Use the file uploader on the main page to select and upload one or more PDF, DOCX, or TXT documents.
   - The assistant will process all uploaded documents, extract their text, and prepare the combined content for querying. A list of successfully processed filenames will be displayed.
   - _Note:_ Large documents or a large number of documents may take some time to process.

2. **Ask Questions:**  
   - Once the documents are processed, a chat interface will appear.
   - Type your questions into the chat input to ask about the content of the uploaded documents.
   - The assistant will analyze the documents and provide concise answers based on their combined context.

3. **Use Sidebar Features:**
   - **Summarize Uploaded Content:** Click this button in the sidebar to get a concise summary of all uploaded documents.
   - **Extract Keywords from Content:** Click this button to see a list of key terms from all documents.
   - **Clear Chat History / Reset All Documents & Chat:** Use these buttons to manage your session.

### Example

1. Upload a DOCX research paper on climate change and a PDF containing supplementary data.
2. The UI will confirm both documents are processed.
3. Click "Summarize Uploaded Content" to get a quick overview of the combined information.
4. Click "Extract Keywords from Content" to see the main topics from both documents.
5. Ask: "What are the primary mitigation strategies discussed in the research paper?"
6. The assistant will extract and present the relevant information.
7. Follow up with: "What datasets support these findings, considering the supplementary PDF?"
8. The assistant will provide details about the datasets mentioned across the relevant documents.

## Environment Variables

The application can be configured using the following environment variables:

- **`OLLAMA_BASE_URL`**: The base URL for the Ollama API.
  - Default: `http://localhost:11434`
  - Example: `http://my-ollama-server:11434`
- **`OLLAMA_EMBEDDING_MODEL_NAME`**: The name of the Ollama model to use for embeddings.
  - Default: `deepseek-r1:1.5b`
  - Ensure this model is available in your Ollama instance (e.g., via `ollama pull deepseek-r1:1.5b`).
- **`OLLAMA_LLM_NAME`**: The name of the Ollama model to use for language generation (answers, summaries, keywords).
  - Default: `deepseek-r1:1.5b`
  - Ensure this model is available in your Ollama instance.
- **`RERANKER_MODEL_NAME`**: The name of the Sentence Transformers CrossEncoder model to use for re-ranking search results.
  - Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - This model will be downloaded automatically on first use if not cached by Sentence Transformers.
- **`PDF_STORAGE_PATH`**: The directory path for storing uploaded documents temporarily during processing.
  - Default: `document_store/pdfs/`
  - Ensure this directory is writable by the application.
- **`LOG_LEVEL`**: The logging level for the application.
  - Default: `INFO`
  - Supported values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

For local development, you can use a `.env` file (e.g., by using the `python-dotenv` library, not included by default) to manage these variables. Add `.env` to your `.gitignore` file.

## Troubleshooting

- **Ollama Not Running:**  
  If you encounter errors related to the language model, ensure that Ollama is running locally. Start it with `ollama serve` in a separate terminal.

- **Document Processing Issues:**  
  If the assistant fails to process a document, check that the file is not corrupted and that it contains extractable text (especially for PDFs). For DOCX and TXT files, ensure they are standard, readable formats. The application has improved error handling to provide more specific feedback.

- **Performance Issues:**  
  For very large documents, processing may take longer. Model loading is cached for better performance after the initial startup.

- **Dependency Conflicts:**  
  If you encounter issues with package installations, ensure that your virtual environment is activated and that you are using the correct Python version.

## Developer Notes

- The `requirements-dev.txt` file (if present, or to be created) would include packages useful for development, such as `pytest` for testing. (Currently, dev dependencies are not formally separated).
- Logging is configured via `core/logger_config.py`. You can adjust log levels and formats there or via the `LOG_LEVEL` environment variable.

## Running with Docker

This section provides instructions on how to run the DocuMind-AI application using Docker and Docker Compose.

### Prerequisites

- **Docker**: Install Docker Desktop (for Windows/Mac) or Docker Engine (for Linux). Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
- **Docker Compose**: Included with Docker Desktop. For Linux, you might need to install it separately. See the [official Docker Compose installation guide](https://docs.docker.com/compose/install/).
- **Ollama**: Must be installed and running on your system or accessible over the network. The DocuMind-AI application container needs to connect to your Ollama instance to use the language models. Ensure you have pulled the required model (e.g., `ollama pull deepseek-r1:1.5b`).

### Setup & Running

1.  **Clone the Repository**:
    If you haven't already, clone the repository:
    ```bash
    git clone https://github.com/chintanboghara/DocuMind-AI.git
    cd DocuMind-AI
    ```

2.  **Configure Ollama Access**:
    *   The `docker-compose.yml` file included in this repository sets the `OLLAMA_BASE_URL` environment variable for the application container to `http://host.docker.internal:11434` by default. This address is typically used to allow a Docker container to access services running on the host machine when using Docker Desktop on Windows or Mac.
    *   **If Ollama is running elsewhere** (e.g., on a different port, a different host machine, or in its own Docker container on a custom Docker network), you **must** update the `OLLAMA_BASE_URL` in the `docker-compose.yml` file.
        ```yaml
        services:
          app:
            # ... other configurations
            environment:
              - OLLAMA_BASE_URL=http://your-ollama-host-or-ip:your-ollama-port
        ```
    *   **Example for Linux users**: If Ollama is running on your host machine (the same machine running Docker), `host.docker.internal` may not resolve. You can often use your machine's IP address on the Docker bridge network (e.g., `172.17.0.1` by default on some systems, but this can vary). Alternatively, you can run Ollama in a Docker container and connect both containers to a shared Docker network.

3.  **Build and Run the Application**:
    Navigate to the root directory of the cloned repository (where `docker-compose.yml` is located) and run:
    ```bash
    docker-compose up --build -d
    ```
    - `--build`: Forces Docker Compose to build the image from the `Dockerfile` (useful for the first run or after changes).
    - `-d`: Runs the containers in detached mode (in the background).

4.  **Accessing the Application**:
    *   Once the container is running (it might take a minute for the first build and startup), open your web browser and navigate to `http://localhost:8501`.

5.  **Stopping the Application**:
    To stop the application and remove the containers defined in `docker-compose.yml`, run:
    ```bash
    docker-compose down
    ```

### Persistent Storage

- The `document_store/pdfs` directory (which stores uploaded documents) is mapped from your local machine into the container using a volume defined in `docker-compose.yml`:
  ```yaml
  volumes:
    - ./document_store:/app/document_store/
  ```
- This means that any documents you upload will persist on your host machine even if the Docker container is stopped, removed, and rebuilt.
