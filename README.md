[![CodeQL Advanced](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml)
[![Dependency Review](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml)

# DocuMind-AI: Intelligent Document Assistant

**DocuMind-AI** is an AI-powered document assistant that allows users to upload documents (PDF, DOCX, TXT), process their contents, and interactively ask questions about them. Built with Streamlit and LangChain, it leverages embeddings and language models to provide concise, factual answers, summaries, and keyword extractions based on the document's context. This tool is ideal for researchers, students, and professionals who need to quickly extract insights from various document formats.

## Features

- **Multi-Format Document Upload:**  
  Upload research papers, reference materials, or any PDF, DOCX, or TXT document directly through the interface. The application automatically extracts text for processing from these formats.

- **Document Processing:**  
  The assistant extracts text from the uploaded document, splits it into manageable chunks, and indexes the content using embeddings. This enables efficient retrieval and querying of the document's information.

- **Intelligent Querying:**  
  Ask questions about the document's content and receive concise, contextually relevant answers. The AI uses a language model to generate responses based on the document's context.

- **Document Summarization:**  
  Generate a concise summary of the entire document with a single click. The summary is displayed in the sidebar, providing a quick overview of the document's main points.

- **Keyword Extraction:**  
  Extract key phrases and terms from the document. These keywords are displayed in the sidebar, helping to identify the document's core topics.

- **Chat-Based UI:**  
  Interact with the assistant via a user-friendly chat interface, making it easy to ask follow-up questions or explore different aspects of the document.

- **Enhanced User Controls (Sidebar):**
  - **Clear Chat History:** Easily clear the current conversation.
  - **Reset Document:** Remove the currently loaded document and chat, allowing you to start fresh with a new file.

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

### 4. Create Required Directories

Ensure that the `document_store/pdfs/` directory exists for storing uploaded documents (though the name "pdfs" is a bit of a misnomer now, it's the current storage path). If not, create it:

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

   This command downloads the Deepseek model (version 1.5b) locally, which is required for generating embeddings and language model responses.

### 6. Run the Application

Start the Streamlit application with the following command:

```bash
streamlit run rag_deep.py
```

This will launch a local web server, typically at [http://localhost:8501](http://localhost:8501), where you can interact with the assistant.

## Usage

1. **Upload a Document:**  
   - Use the file uploader on the main page to select and upload your PDF, DOCX, or TXT document.  
   - The assistant will process the document, extract its text, and prepare it for querying.  
   - _Note:_ Large documents may take a few moments to process.

2. **Ask Questions:**  
   - Once the document is processed, a chat interface will appear.  
   - Type your questions into the chat input to ask about the document's content.  
   - The assistant will analyze the document and provide concise answers based on its context.

3. **Use Sidebar Features:**
   - **Summarize Document:** Click this button in the sidebar to get a concise summary of the document.
   - **Extract Keywords:** Click this button to see a list of key terms from the document.
   - **Clear Chat History / Reset Document:** Use these buttons to manage your session.

### Example

1. Upload a DOCX research paper on climate change.
2. Click "Summarize Document" to get a quick overview.
3. Click "Extract Keywords" to see the main topics.
4. Ask: "What are the primary mitigation strategies discussed?"  
5. The assistant will extract and present the relevant information.  
6. Follow up with: "What datasets support these findings?"  
7. The assistant will provide details about the datasets mentioned.

## Troubleshooting

- **Ollama Not Running:**  
  If you encounter errors related to the language model, ensure that Ollama is running locally. Start it with `ollama serve` in a separate terminal.

- **Document Processing Issues:**  
  If the assistant fails to process a document, check that the file is not corrupted and that it contains extractable text (especially for PDFs). For DOCX and TXT files, ensure they are standard, readable formats. The application has improved error handling to provide more specific feedback.

- **Performance Issues:**  
  For very large documents, processing may take longer. Model loading is cached for better performance after the initial startup.

- **Dependency Conflicts:**  
  If you encounter issues with package installations, ensure that your virtual environment is activated and that you are using the correct Python version.

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
