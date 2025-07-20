
# 4. Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for document processing and OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Make port 8000 available to the world outside this container (for Flask API)
EXPOSE 8000

# Make port 8501 available (for Streamlit)
EXPOSE 8501

# Define environment variables from .env file (placeholders)
# These will be overridden by docker-compose or Kubernetes secrets in production
ENV GROQ_API_KEY=""
ENV ASTRA_DB_API_ENDPOINT=""
ENV ASTRA_DB_APPLICATION_TOKEN=""
ENV ASTRA_DB_KEYSPACE=""
ENV ASTRA_DB_COLLECTION_NAME=""
ENV EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"

# Create a startup script
RUN echo '#!/bin/sh' > start.sh && \
    echo 'gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 api:app &' >> start.sh && \
    echo 'streamlit run app.py --server.port 8501 --server.address 0.0.0.0' >> start.sh && \
    chmod +x start.sh

# Run the startup script when the container launches
CMD ["./start.sh"]
