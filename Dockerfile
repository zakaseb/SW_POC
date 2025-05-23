# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script and the document_store directory into the container at /app
COPY rag_deep.py .
COPY document_store/ ./document_store/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run the app
# Running on 0.0.0.0 allows it to be accessible from outside the container
# Ensure Streamlit uses the exposed port
CMD ["streamlit", "run", "rag_deep.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
