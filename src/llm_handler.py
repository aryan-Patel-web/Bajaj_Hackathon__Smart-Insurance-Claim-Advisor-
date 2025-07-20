# src/llm_handler.py

"""
Handles all interactions with the Groq LLM with enhanced debugging.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal

from config.settings import settings

# --- Pydantic Model for Structured JSON Output ---
class Justification(BaseModel):
    clause_id: str = Field(..., description="The unique identifier for the document chunk, e.g., 'policy_doc_1|chunk_23'.")
    text: str = Field(..., description="The exact text of the relevant clause from the source document.")
    source_file: str = Field(..., description="The name of the source file, e.g., 'health_policy_v2.pdf'.")
    page_number: int = Field(..., description="The page number in the source file where the clause was found.")
    relevance_score: float = Field(..., description="The relevance score (0.0 to 1.0) of this clause to the query.")

class ClaimDecision(BaseModel):
    decision: Literal["Approved", "Rejected", "Needs More Information"] = Field(..., description="The final claim decision.")
    amount: str = Field(..., description="The approved or relevant claim amount, e.g., '₹50,000' or 'Not Applicable'.")
    confidence_score: float = Field(..., description="The model's confidence in its decision, from 0.0 to 1.0.")
    justification: List[Justification] = Field(..., description="A list of clauses and evidence justifying the decision.")
    reasoning_steps: List[str] = Field(..., description="A step-by-step explanation of how the decision was reached.")
    follow_up_questions: List[str] = Field(..., description="Relevant follow-up questions for the user.")
    summary: str = Field(..., description="A concise, natural language summary of the decision for the chat response.")


class LLMHandler:
    def __init__(self):
        self.chat_model = ChatGroq(
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
            model="gemma2-9b-it",
        ).with_structured_output(ClaimDecision)
        
        self.system_prompt = "You are an expert AI Insurance Claim Advisor..." # Same prompt as before
        self.human_prompt = """Context: {context}\n\nHistory: {chat_history}\n\nQuery: {query}"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.human_prompt),
        ])
        
        self.chain = self.prompt_template | self.chat_model

    def get_streaming_response(self, query: str, context: str, chat_history: str):
        print("--- [LLM Handler] Preparing to invoke LLM chain. ---")
        try:
            # This is a likely point of failure if the API call hangs
            response_stream = self.chain.stream({
                "query": query,
                "context": context,
                "chat_history": chat_history
            })
            print("--- [LLM Handler] Chain stream initiated successfully. ---")
            yield from response_stream
        except Exception as e:
            print(f"\n\n--- [LLM Handler] FATAL ERROR invoking LLM chain: {e} ---\n\n")
            # Create an error response to send back to the user
            error_decision = ClaimDecision(
                decision="Needs More Information",
                amount="N/A",
                confidence_score=0.0,
                justification=[],
                reasoning_steps=[f"An error occurred while communicating with the AI model: {e}"],
                follow_up_questions=["Please try your query again or check the server logs."],
                summary=f"I'm sorry, I encountered a critical error trying to process your request. Details: {e}"
            )
            yield error_decision


llm_handler = LLMHandler()
