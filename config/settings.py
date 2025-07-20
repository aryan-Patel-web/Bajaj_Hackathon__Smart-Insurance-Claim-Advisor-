# config/settings.py

"""
Centralized configuration management for the application.
Loads environment variables from a .env file and provides them as typed settings.
This approach decouples configuration from the code, making it easier to manage
across different environments (dev, staging, prod).
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    """
    A singleton-like class to hold all application settings.
    """
    def __init__(self):
        # --- Groq API Configuration ---
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

        # --- Astra DB Configuration ---
        self.ASTRA_DB_API_ENDPOINT: str = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.ASTRA_DB_APPLICATION_TOKEN: str = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.ASTRA_DB_KEYSPACE: str = os.getenv("ASTRA_DB_KEYSPACE")
        self.ASTRA_DB_COLLECTION_NAME: str = os.getenv("ASTRA_DB_COLLECTION_NAME")

        # --- Validation Logic (Moved inside __init__) ---
        required_vars = [
            "GROQ_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE", "ASTRA_DB_COLLECTION_NAME"
        ]
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required Astra DB environment variables: {', '.join(missing_vars)}")

        # --- Embedding Model Configuration ---
        self.EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.EMBEDDING_DIMENSION: int = 384

        # --- API Configuration ---
        self.API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))

        # --- Project Directories ---
        self.BASE_DIR: Path = Path(__file__).resolve().parent.parent
        self.DATA_DIR: Path = self.BASE_DIR / "data"
        self.DOCUMENTS_DIR: Path = self.DATA_DIR / "documents"

        # --- Chunking Configuration ---
        self.CHUNK_SIZE: int = 1000
        self.CHUNK_OVERLAP: int = 200

# Create a single instance of the settings to be imported across the application
settings = Settings()

# Create necessary directories on startup
settings.DATA_DIR.mkdir(exist_ok=True)
settings.DOCUMENTS_DIR.mkdir(exist_ok=True)

if __name__ == '__main__':
    print("--- Application Settings Verification ---")
    print(f"Groq API Key Loaded: {bool(settings.GROQ_API_KEY)}")
    print(f"Astra DB Endpoint: {settings.ASTRA_DB_API_ENDPOINT}")
    print(f"Astra DB Keyspace: {settings.ASTRA_DB_KEYSPACE}")
    print(f"Astra DB Collection: {settings.ASTRA_DB_COLLECTION_NAME}")
    print(f"Astra Token Loaded: {bool(settings.ASTRA_DB_APPLICATION_TOKEN)}")
    print("---------------------------------------")
