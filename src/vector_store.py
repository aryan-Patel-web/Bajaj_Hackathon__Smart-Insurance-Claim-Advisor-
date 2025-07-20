# src/vector_store.py

"""
Manages the connection to and interaction with the Astra DB vector store.
This module initializes the embedding model and the LangChain vector store
component, providing a centralized point for all database operations like
adding documents and retrieving them.
"""


# Use the new, dedicated packages as recommended by LangChain deprecation warnings.
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    """
    A manager class for handling all interactions with Astra DB.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing VectorStoreManager with new packages...")
        try:
            # 1. Initialize the embedding model using the new package
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
            
            # 2. Initialize the Astra DB vector store using the new package
            self.vector_store = AstraDBVectorStore(
                embedding=self.embedding_model,
                collection_name=settings.ASTRA_DB_COLLECTION_NAME,
                api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
                token=settings.ASTRA_DB_APPLICATION_TOKEN,
                namespace=settings.ASTRA_DB_KEYSPACE,
            )
            self._initialized = True
            logger.info("Astra DB vector store initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreManager: {e}", exc_info=True)
            self._initialized = False
            raise

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of chunked documents to the Astra DB vector store.

        Args:
            documents (List[Document]): The documents to be added.
        """
        if not self._initialized:
            logger.error("Vector store is not initialized. Cannot add documents.")
            return

        if not documents:
            logger.warning("Attempted to add an empty list of documents.")
            return

        logger.info(f"Adding {len(documents)} document chunks to Astra DB...")
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} chunks to the collection '{settings.ASTRA_DB_COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to Astra DB: {e}", exc_info=True)
            raise

    def get_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Gets a retriever object from the vector store.

        Args:
            search_type (str): The type of search to perform ('similarity', 'mmr', etc.).
            search_kwargs (dict): Keyword arguments for the search (e.g., {'k': 5}).

        Returns:
            A LangChain retriever object.
        """
        if not self._initialized:
            logger.error("Vector store is not initialized. Cannot get retriever.")
            return None
        
        if search_kwargs is None:
            search_kwargs = {'k': 5} # Default to retrieving top 5 results
            
        logger.info(f"Creating retriever with search_type='{search_type}' and search_kwargs={search_kwargs}")
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

# Singleton instance to be used across the application
vector_store_manager = VectorStoreManager()
