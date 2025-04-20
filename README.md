[![CodeQL Advanced](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/codeql.yml)
[![Dependency Review](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/chintanboghara/DocuMind-AI/actions/workflows/dependency-review.yml)

# DocuMind-AI: Intelligent Document Assistant

**DocuMind-AI** is an AI-powered document assistant that allows users to upload PDF documents, process their contents, and interactively ask questions about them. Built with Streamlit and LangChain, it leverages embeddings and language models to provide concise, factual answers based on the document's context. This tool is ideal for researchers, students, and professionals who need to quickly extract insights from PDFs.

## Features

- **PDF Upload:**  
  Upload research papers, reference materials, or any PDF document directly through the interface. Supported file types include standard PDFs, with automatic text extraction for processing.

- **Document Processing:**  
  The assistant extracts text from the uploaded PDF, splits it into manageable chunks, and indexes the content using embeddings. This enables efficient retrieval and querying of the document's information.

- **Intelligent Querying:**  
  Ask questions about the document's content and receive concise, contextually relevant answers. The AI uses a language model to generate responses based on the document's context, ensuring factual accuracy.

- **Chat-Based UI:**  
  Interact with the assistant via a user-friendly chat interface, making it easy to ask follow-up questions or explore different aspects of the document.

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

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Create Required Directories

Ensure that the `document_store/pdfs/` directory exists for storing uploaded PDFs. If not, create it:

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

1. **Upload a PDF:**  
   - Use the file uploader on the main page to select and upload your PDF document.  
   - The assistant will process the PDF, extract its text, and prepare it for querying.  
   - _Note:_ Large PDFs may take a few moments to process.

2. **Ask Questions:**  
   - Once the document is processed, a chat interface will appear.  
   - Type your questions into the chat input to ask about the document's content.  
   - The assistant will analyze the document and provide concise answers based on its context.  
   - You can ask follow-up questions or explore different topics within the document.

### Example

1. Upload a research paper on machine learning.  
2. Ask: "What is the main hypothesis of this paper?"  
3. The assistant will extract and summarize the hypothesis from the document.  
4. Follow up with: "What datasets were used in the experiments?"  
5. The assistant will provide details about the datasets mentioned in the paper.

## Troubleshooting

- **Ollama Not Running:**  
  If you encounter errors related to the language model, ensure that Ollama is running locally. Start it with `ollama serve` in a separate terminal.

- **PDF Processing Issues:**  
  If the assistant fails to process a PDF, check that the file is not corrupted and that it contains extractable text (i.e., not scanned images).

- **Performance Issues:**  
  For large documents, processing may take longer. Consider splitting the document into smaller sections or increasing system resources.

- **Dependency Conflicts:**  
  If you encounter issues with package installations, ensure that your virtual environment is activated and that you are using the correct Python version.
