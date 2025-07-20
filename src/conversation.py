# src/conversation.py

# CORRECTED: Import the singleton instances, not non-existent functions.
from src.vector_store import vector_store_manager
from src.llm_handler import llm_handler, ClaimDecision

def format_chat_history(history: list[dict]):
    """Formats the chat history for the LLM."""
    if not history:
        return "No previous conversation."
    # Take the last 4 messages to keep the context window focused
    recent_history = history[-4:]
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

# CORRECTED: Renamed function to match the import in app.py
def get_full_response(query: str, chat_history: list[dict]):
    """
    Orchestrates the full RAG pipeline with robust debugging.
    This is the core logic engine of the chatbot.
    """
    print("\n\n================ NEW QUERY RECEIVED ================")
    print(f"USER QUERY: {query}")
    
    # --- 1. Document Retrieval ---
    print("\n[STEP 1] Retrieving documents from vector store...")
    try:
        # CORRECTED: Use the get_retriever method from the manager instance
        retriever = vector_store_manager.get_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'fetch_k': 25} # Retrieve more chunks for better context
        )
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            print("[CRITICAL FAILURE] The vector store returned ZERO documents.")
            context = "No relevant context could be found in the uploaded documents."
        else:
            print(f"[SUCCESS] Retrieved {len(retrieved_docs)} document chunks.")
            context = "\n\n---\n\n".join(
                [f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in retrieved_docs]
            )
    except Exception as e:
        print(f"[FATAL ERROR in STEP 1] Could not retrieve documents: {e}")
        context = "An error occurred while searching the policy documents."

    # --- 2. Context Preparation ---
    print("\n[STEP 2] Preparing final context for the AI.")
    print("==================== CONTEXT START ====================")
    print(context)
    print("===================== CONTEXT END =====================")

    # --- 3. LLM Invocation ---
    try:
        formatted_history = format_chat_history(chat_history)
        print(f"\n[STEP 3] Invoking the LLM...")
        
        # CORRECTED: Use the chain attribute from the llm_handler instance
        response_stream = llm_handler.chain.stream({
            "context": context,
            "chat_history": formatted_history,
            "query": query,
        })
        
        print("[SUCCESS] LLM stream initiated. Sending to frontend...")
        yield from response_stream
        print("--- Stream finished ---")

    except Exception as e:
        print(f"[FATAL ERROR in STEP 3] Could not get response from LLM: {e}")
        # Yield a final error message to the user
        error_response = {
            "decision": "Needs More Information",
            "amount": "N/A",
            "summary": f"I'm sorry, a critical error occurred in the AI model: {e}",
            "justification": [],
            "reasoning_steps": [],
            "follow_up_questions": [],
            "confidence_score": 0.0
        }
        error_decision = ClaimDecision(**error_response)
        yield error_decision