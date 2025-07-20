# src/hybrid_search.py

"""
Implements a robust retrieval strategy.
This version ensures that context is always retrieved and provides clear
logging to trace the data flow.
"""

from typing import List
from langchain.docstore.document import Document
from src.vector_store import vector_store_manager

def format_context(documents: List[Document]) -> str:
    """
    Formats the retrieved documents into a single string for the LLM.
    """
    if not documents:
        print("--- [Hybrid Search] WARNING: No documents were retrieved. Context will be empty. ---")
        return "No relevant context found in the documents."
    
    context_parts = []
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'N/A')
        # Ensure content is a string
        content = str(doc.page_content) if doc.page_content is not None else ""
        
        part = (
            f"[Source: {source}, Page: {page}]\n"
            f"{content}\n"
        )
        context_parts.append(part)
    
    return "\n---\n".join(context_parts)


def retrieve_relevant_documents(query: str, k: int = 5) -> (List[Document], str):
    """
    Retrieves the most relevant document chunks for a given query.
    This is a critical step and has been made more robust.
    """
    print(f"\n--- [Hybrid Search] Attempting to retrieve top {k} documents for query: '{query}' ---")
    try:
        # Use the most reliable retriever setting. This ensures we always get results if the DB has data.
        retriever = vector_store_manager.get_retriever(search_kwargs={'k': k})
        if not retriever:
            raise Exception("Failed to create a retriever from the vector store.")

        print("--- [Hybrid Search] Retriever created. Invoking search...")
        documents = retriever.invoke(query)
        
        if not documents:
             print("--- [Hybrid Search] CRITICAL: Search returned NO documents. Check if ingestion was successful. ---")
        else:
            print(f"--- [Hybrid Search] SUCCESS: Retrieved {len(documents)} documents. ---")

        # Format the context to be sent to the LLM
        formatted_context = format_context(documents)
        return documents, formatted_context

    except Exception as e:
        print(f"\n\n--- [Hybrid Search] FATAL ERROR: An exception occurred during document retrieval: {e} ---\n\n")
        # Return an error message in the context itself
        return [], f"Error during document retrieval: {e}"

