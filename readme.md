# Smart Insurance Claim Advisor 🤖🏥

Smart Insurance Claim Advisor is an intelligent assistant that:
- **Ingests** various document formats (PDF, DOCX, PPTX, CSV, images with OCR, emails)
- **Chunks and vectorizes** them for semantic search
- **Parses** natural language claim queries (age, location, procedure, policy details)
- **Retrieves** relevant policy clauses via hybrid semantic + metadata search
- **Runs** an LLM to evaluate and justify claim decisions
- **Interacts** with users through a polished Streamlit chat interface

---

## 📁 Repository Structure

├── config/
│ └── settings.py
├── src/
│ ├── ingest.py # Document ingestion pipeline
│ ├── parse_query.py # Query parsing & normalization
│ ├── hybrid_search.py # Hybrid vector + metadata search
│ ├── llm_handler.py # Groq LLM claim decision logic
│ ├── vectorstore.py # AstraDB vector store integration
│ └── conversation.py # Session & memory management
└── utils/
├── chunking.py # Document chunker utilities
├── document_loader.py # Multi-format loader + OCR support
└── logging_config.py # Structured audit & app logging



---

## 🚀 Quick Start

1. **Install dependencies** (see `requirements.txt`).
2. **Set environment variables**:
   - `ASTRA_DB_TOKEN`, `ASTRA_DB_ENDPOINT`, `ASTRA_DB_KEYSPACE`
   - `GROQ_API_KEY`
3. **Run the Streamlit app**:
   ```bash
   streamlit run smart_insurance_claim_app.py


Features
Multilingual document ingestion with OCR support

Semantic chunking (~1 KB chunks with overlap) for robust embedding

Query parsing: age, gender, procedure, location, policy duration, claim amount

Hybrid search combining vector similarity with metadata filters

Groq-powered LLM evaluation with justification from policy clauses

Conversation tracking, summaries, and analytics dashboard

