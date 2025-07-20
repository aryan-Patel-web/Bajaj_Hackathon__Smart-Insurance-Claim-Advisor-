# src/ingest.py

"""
Defines the document ingestion pipeline.
This module coordinates the process of loading files from a directory,
chunking them into manageable pieces, and then adding them to the
Astra DB vector store for future retrieval.
"""

import os
from typing import List

from config.settings import settings
from utils.logging_config import get_logger
from utils.document_loader import load_documents_from_directory, load_single_document
from utils.chunking import chunk_documents
from src.vector_store import vector_store_manager

logger = get_logger(__name__)

def process_and_embed_documents(directory_path: str = None, file_paths: List[str] = None):
    """
    The main pipeline function to process and embed documents.
    It can either process all documents in a directory or a specific list of files.

    Args:
        directory_path (str, optional): The path to the directory of documents.
        file_paths (List[str], optional): A list of absolute paths to specific files.
    
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    if not directory_path and not file_paths:
        logger.error("Must provide either a directory_path or a list of file_paths.")
        return False

    try:
        # Step 1: Load documents from source
        if directory_path:
            logger.info(f"Loading documents from directory: {directory_path}")
            documents = load_documents_from_directory(directory_path)
        else:
            logger.info(f"Loading {len(file_paths)} specific documents.")
            all_docs = []
            for path in file_paths:
                all_docs.extend(load_single_document(path))
            documents = all_docs

        if not documents:
            logger.warning("No documents were loaded. Ingestion process halted.")
            return False
        
        logger.info(f"Successfully loaded {len(documents)} document parts.")

        # Step 2: Chunk the loaded documents
        chunked_docs = chunk_documents(documents)

        if not chunked_docs:
            logger.warning("No chunks were created from the documents. Ingestion process halted.")
            return False
        
        logger.info(f"Successfully chunked documents into {len(chunked_docs)} pieces.")

        # Step 3: Add the chunks to the vector store
        # This will also handle embedding generation
        vector_store_manager.add_documents(chunked_docs)

        logger.info("Document ingestion and embedding pipeline completed successfully.")
        return True

    except Exception as e:
        logger.error(f"An error occurred during the ingestion pipeline: {e}", exc_info=True)
        return False

def process_uploaded_files(uploaded_files: List) -> (bool, int):
    """
    Handles files uploaded via the Streamlit interface.

    Args:
        uploaded_files (List): A list of Streamlit UploadedFile objects.

    Returns:
        Tuple[bool, int]: A tuple containing a success flag and the number of processed files.
    """
    if not uploaded_files:
        return False, 0
    
    saved_file_paths = []
    # Save uploaded files to the designated documents directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(settings.DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)
        logger.info(f"Saved uploaded file to {file_path}")

    # Process the newly saved files
    success = process_and_embed_documents(file_paths=saved_file_paths)
    
    return success, len(saved_file_paths)


if __name__ == '__main__':
    # Example of running the full ingestion pipeline on the documents directory
    print("--- Starting Full Document Ingestion Pipeline ---")
    
    # This assumes you have placed some sample documents in the `data/documents` folder
    # as specified in the README.
    if not any(settings.DOCUMENTS_DIR.iterdir()):
        print("\nWARNING: The `data/documents` directory is empty.")
        print("Please add some PDF, DOCX, or other supported files to test the ingestion pipeline.")
    else:
        print(f"Processing all documents in: {settings.DOCUMENTS_DIR}")
        success = process_and_embed_documents(directory_path=str(settings.DOCUMENTS_DIR))
        
        if success:
            print("\nIngestion pipeline completed successfully!")
            print("Documents have been chunked, embedded, and stored in Astra DB.")
        else:
            print("\nIngestion pipeline failed. Please check the 'app.log' file for errors.")
    
    print("\n--- Pipeline Test Finished ---")

