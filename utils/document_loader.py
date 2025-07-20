# utils/document_loader.py

"""
Handles loading and parsing of various document formats.
This module abstracts the complexity of file handling and content extraction,
providing a unified interface to the ingestion pipeline. It supports PDF, DOCX,
PPTX, and common image formats (using OCR).
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader,
    UnstructuredFileLoader,
)
from langchain.docstore.document import Document
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Mapping file extensions to their respective loaders
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    # Add other file types and their loaders here
    # For images, we'll use a more general unstructured loader
    ".png": (UnstructuredFileLoader, {"mode": "single", "strategy": "hi_res"}),
    ".jpg": (UnstructuredFileLoader, {"mode": "single", "strategy": "hi_res"}),
    ".jpeg": (UnstructuredFileLoader, {"mode": "single", "strategy": "hi_res"}),
}


def load_single_document(file_path: str) -> List[Document]:
    """
    Loads a single document from the given file path and returns its content
    as a list of LangChain Document objects.

    Args:
        file_path (str): The full path to the document file.

    Returns:
        List[Document]: A list of Document objects, where each object
                        might represent a page or the entire document.
    
    Raises:
        ValueError: If the file extension is not supported.
        Exception: For any other loading errors.
    """
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext not in LOADER_MAPPING:
        logger.error(f"Unsupported file extension: '{ext}' for file: {file_path}")
        raise ValueError(f"Unsupported file extension: '{ext}'")

    loader_class, loader_args = LOADER_MAPPING[ext]
    
    try:
        logger.info(f"Loading document: {file_path} using {loader_class.__name__}")
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} parts from {file_path}.")
        return documents
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {e}", exc_info=True)
        # Return an empty list or re-raise the exception depending on desired behavior
        return []


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Loads all supported documents from a specified directory.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        List[Document]: A list of all loaded Document objects from the directory.
    """
    all_documents = []
    path = Path(directory_path)
    if not path.is_dir():
        logger.error(f"Provided path is not a directory: {directory_path}")
        return []

    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in LOADER_MAPPING:
            docs = load_single_document(str(file_path))
            if docs:
                all_documents.extend(docs)
    
    logger.info(f"Loaded a total of {len(all_documents)} document parts from directory: {directory_path}")
    return all_documents


if __name__ == '__main__':
    # Example usage:
    # Create dummy files for testing
    from config.settings import settings
    
    dummy_pdf_path = settings.DOCUMENTS_DIR / "sample.pdf"
    dummy_docx_path = settings.DOCUMENTS_DIR / "sample.docx"
    
    # You would need to place actual files in the `data/documents` directory
    # For this example, we'll just check if they exist and try to load them.
    
    if dummy_pdf_path.exists():
        print("\n--- Loading PDF ---")
        pdf_docs = load_single_document(str(dummy_pdf_path))
        if pdf_docs:
            print(f"Loaded {len(pdf_docs)} pages from PDF.")
            print("Metadata of first page:", pdf_docs[0].metadata)
            print("Content snippet:", pdf_docs[0].page_content[:200])
    else:
        print(f"Skipping PDF test: {dummy_pdf_path} not found.")

    if dummy_docx_path.exists():
        print("\n--- Loading DOCX ---")
        docx_docs = load_single_document(str(dummy_docx_path))
        if docx_docs:
            print(f"Loaded {len(docx_docs)} parts from DOCX.")
            print("Metadata of first part:", docx_docs[0].metadata)
            print("Content snippet:", docx_docs[0].page_content[:200])
    else:
        print(f"Skipping DOCX test: {dummy_docx_path} not found.")

    print("\n--- Loading all documents from directory ---")
    all_docs = load_documents_from_directory(str(settings.DOCUMENTS_DIR))
    print(f"Total documents loaded: {len(all_docs)}")

