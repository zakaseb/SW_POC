# DocuMind-AI

**DocuMind-AI** is an intelligent document assistant built with Streamlit and LangChain. It enables users to upload a PDF document, process its contents, and interactively ask questions about it using AI. The assistant leverages embeddings and language models to provide concise, factual answers based on the document's context.

## Features

- **PDF Upload:** Easily upload research or reference PDFs.
- **Document Processing:** Automatically extracts text from PDFs, splits it into manageable chunks, and indexes it.
- **Intelligent Querying:** Ask questions about the document and receive concise answers.
- **Chat-Based UI:** Interact with the assistant via a user-friendly chat interface.

## Installation

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/chintanboghara/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Set Up a Virtual Environment (Recommended)

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

Ensure that the `document_store/pdfs/` directory exists. If not, create it:

- **Windows:**

  ```bash
  mkdir document_store\pdfs
  ```

- **Ubuntu:**

  ```bash
  mkdir -p document_store/pdfs
  ```

### 5. Run the Application

Start the Streamlit application with the following command:

```bash
streamlit run rag_deep.py
```

This will launch a local web server (typically at [http://localhost:8501](http://localhost:8501)) where you can interact with the assistant.

## Usage

1. **Upload a PDF:**  
   Use the file uploader to select and upload your PDF document.

2. **Ask Questions:**  
   Once the document is processed, type your questions into the chat interface. The assistant will analyze the document context and provide answers.
