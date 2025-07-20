# Smart Insurance Claim Advisor - HackRx 6.0

This project is a production-grade AI chatbot built for the Bajaj Finserv Health HackRx 6.0 competition. It uses a high-speed RAG (Retrieval-Augmented Generation) pipeline to analyze insurance policy documents and provide instant, explainable decisions on user claims.

The system is architected for **speed**, **accuracy**, and a **premium user experience** to meet the demanding standards of the hackathon.

## Features

-   **Modular Architecture**: Code is logically separated into `config`, `src`, and `utils` for maintainability.
-   **Multi-Format Ingestion**: Upload and process PDFs, DOCX, TXT, and even images (PNG, JPG) containing text via on-the-fly OCR.
-   **Blazing-Fast LLM**: Powered by **Groq** and Google's `gemma2-9b-it` for near-instant inference and query parsing.
-   **Scalable Vector DB**: Uses **Astra DB Serverless** for a persistent, fast, and scalable vector store and conversation memory.
-   **Advanced RAG Pipeline**:
    -   Parses natural language queries into structured data.
    -   Performs semantic search to find the most relevant policy clauses.
    -   Maintains conversation history for contextual follow-up questions.
-   **Explainable AI (XAI)**: Every decision is returned in a structured JSON format, citing the exact policy clauses used for justification, fulfilling a core judging criterion.
-   **Modern UI**: A responsive, streaming chat interface built with Flask, styled to look and feel like **Google's Gemini**, with real-time response animations.
-   **Containerized**: Includes a `Dockerfile` for easy, reproducible deployment.

## Project Structure

```
insurance_claim_advisor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── README.md              # This file
├── .env.example           # Environment variables template
├── config/
│   └── settings.py        # Configuration management
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Document ingestion pipeline entrypoint
│   ├── parse_query.py     # Query parsing and structuring
│   ├── vector_store.py    # Astra DB vector store operations
│   ├── llm_handler.py     # Core RAG chain and Groq LLM interaction
│   └── conversation.py    # Conversation memory management
├── utils/
│   ├── __init__.py
│   ├── document_loader.py # Multi-format document loading
│   ├── chunking.py        # Document chunking utilities
│   └── logging_config.py  # Logging configuration
├── templates/
│   └── index.html         # Single-file frontend with inline CSS & JS
└── data/
    └── documents/         # Place initial documents here
```



## Setup & Running the Application

1.  **Prerequisites**:
    * Python 3.10+
    * Tesseract OCR Engine (install via your system's package manager, e.g., `sudo apt-get install tesseract-ocr` on Debian/Ubuntu).
    * Git

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd insurance_claim_advisor
    ```

3.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
     # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    -   Create a file named `.env` in the project root by copying `.env.example`.
    -   Fill in your actual credentials from Groq and DataStax Astra.

6.  **Initial Document Ingestion (Optional but Recommended):**
    -   Place your initial set of insurance policy documents inside the `data/documents/` directory.
    -   Run the ingestion script. This will process all documents and populate your Astra DB vector store.
    ```bash
    python -m src.ingest
    ```

7.  **Run the Flask application:**
    ```bash
    flask run --host=0.0.0.0 --port=5000
    ```

8.  **Access the Chatbot:**
    -   Open your web browser and navigate to `http://127.0.0.1:5000`.

---


####

# Testing example - 

Mr. swapnil - Fractured Wrist from Bike Accident
🔹 Basic User Details
Name: Rajesh Sharma

Policy Number: ICICI-1234-5678-9012

Relationship to Policyholder: Self

Health Card No.: HLTH-ICICI-8765

Hospital Name: Fortis Hospital, Noida (ICICI Network Provider)

Doctor: Dr. Anjali Mehta

Diagnosis: Left wrist fracture due to bike fall

Treatment Plan: Surgery + 2-day hospitalization

Hospital Contact: +91 98123 45678

Admission Date: 19 July 2025

Accident Date: 18 July 2025

Email: swapnil.sharma@gmail.com

Phone: +91 9876543210

💳 Case A: Cashless Settlement Test (Planned Surgery)

I have a planned wrist surgery at Fortis Hospital Noida on 19 July. I want to apply for cashless hospitali