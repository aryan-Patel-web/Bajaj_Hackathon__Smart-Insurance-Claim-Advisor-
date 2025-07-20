# utils/chunking.py

"""
Provides functionality for splitting large documents into smaller, manageable chunks.
This is a crucial step for embedding, as LLMs and vector stores have context limits.
Using a RecursiveCharacterTextSplitter with overlap helps maintain semantic context
between chunks.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.

    Args:
        documents (List[Document]): The list of documents to be chunked.

    Returns:
        List[Document]: A new list of chunked Document objects. Each chunk
                        retains the metadata of its source document.
    """
    if not documents:
        logger.warning("chunk_documents called with an empty list of documents.")
        return []

    logger.info(f"Starting to chunk {len(documents)} document parts...")
    
    # Initialize the text splitter with parameters from the settings file.
    # This splitter tries to split on common separators like newlines first.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helps in identifying chunk position
    )

    # The `split_documents` method handles iterating through docs and splitting them.
    chunked_documents = text_splitter.split_documents(documents)

    logger.info(f"Chunking complete. Original parts: {len(documents)}, Total chunks: {len(chunked_documents)}")
    
    # Add a custom chunk_id to the metadata for precise referencing
    for i, chunk in enumerate(chunked_documents):
        source_file = Path(chunk.metadata.get("source", "unknown_source")).name
        chunk.metadata["chunk_id"] = f"{source_file}|chunk_{i}"

    return chunked_documents

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Document Chunking ---")
    
    # Create a dummy document for testing purposes
    sample_content = "This is the first sentence. " * 100
    sample_content += "This is the middle part of the document that should be split. " * 200
    sample_content += "This is the final sentence. " * 100
    
    dummy_doc = Document(
        page_content=sample_content,
        metadata={"source": "/path/to/fake_document.pdf", "page": 1}
    )
    
    print(f"Original document length: {len(dummy_doc.page_content)} characters.")
    
    # Chunk the dummy document
    chunks = chunk_documents([dummy_doc])
    
    print(f"Document split into {len(chunks)} chunks.")
    
    if chunks:
        print("\n--- Details of the first chunk ---")
        print(f"Content length: {len(chunks[0].page_content)}")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content snippet: '{chunks[0].page_content[:250]}...'")

        if len(chunks) > 1:
            print("\n--- Details of the second chunk ---")
            print(f"Content length: {len(chunks[1].page_content)}")
            print(f"Metadata: {chunks[1].metadata}")
            
            # Check for overlap
            overlap_start = chunks[0].page_content[-settings.CHUNK_OVERLAP:]
            second_chunk_start = chunks[1].page_content[:settings.CHUNK_OVERLAP]
            print(f"Overlap check: The end of chunk 1 should match the start of chunk 2.")
            # This is an approximation, as the splitter might not be exact
            print(f"End of chunk 1 (last 50 chars): '...{overlap_start[-50:]}'")
            print(f"Start of chunk 2 (first 50 chars): '{second_chunk_start[:50]}...'")

