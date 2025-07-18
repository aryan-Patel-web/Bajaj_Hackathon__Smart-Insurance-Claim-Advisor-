import os
import json
import logging
from typing import Dict, List, Any, Optional
from groq import Groq
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

class ClauseJustification(BaseModel):
    """Model for clause justification"""
    clause_id: str = Field(description="Source file and chunk ID")
    text: str = Field(description="Relevant clause text")

class ClaimDecision(BaseModel):
    """Model for claim decision response"""
    decision: str = Field(description="Approved or Rejected")
    amount: str = Field(description="Amount in ₹ format")
    justification: List[ClauseJustification] = Field(description="List of supporting clauses")

class LLMHandler:
    """Handles interactions with Groq LLM for insurance claim processing"""
    
    def __init__(self):
        """Initialize the LLM handler"""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "gemma2-9b-it"
        
    def generate_response(
        self, 
        query: str,
        parsed_query: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response for insurance claim query
        
        Args:
            query: Original user query
            parsed_query: Structured query fields
            search_results: Retrieved policy clauses
            context: Conversation context
            
        Returns:
            Dict containing natural response and structured JSON
        """
        logger.info(f"Generating response for query: {query}")
        
        try:
            # Prepare the prompt
            prompt = self._create_prompt(query, parsed_query, search_results, context)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
                top_p=0.9
            )
            
            # Extract response content
            # response_content = response.choices[0].message.content
            response_content = response.choices[0].message.content
            logger.info(f"LLM response: {response_content}")

                        # Ensure response_content is a string
            if response_content is None:
                response_content = ""
            
            # Parse the structured response
            structured_response = self._parse_structured_response(response_content)
            
            # Generate natural language response
            natural_response = self._generate_natural_response(structured_response)
            
            return {
                "natural_response": natural_response,
                "structured_response": structured_response,
                "raw_response": response_content
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "natural_response": f"I apologize, but I encountered an error processing your claim: {str(e)}",
                "structured_response": {
                    "decision": "Error",
                    "amount": "N/A",
                    "justification": []
                },
                "raw_response": ""
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are an expert insurance claim advisor AI assistant. Your task is to analyze insurance claims based on policy documents and provide structured decisions.

CRITICAL INSTRUCTIONS:
1. Analyze the provided policy clauses to determine claim eligibility
2. Consider the claimant's details (age, procedure, location, policy duration)
3. Provide a decision (Approved/Rejected) with amount and justification
4. ALWAYS return your response in the exact JSON format specified
5. Be thorough in your analysis and cite specific clauses
6. Use Indian Rupee (₹) format for amounts
7. Ensure decisions are based on the provided policy clauses

You must be accurate, fair, and transparent in your decision-making process."""

    def _create_prompt(
        self, 
        query: str, 
        parsed_query: Dict[str, Any], 
        search_results: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a detailed prompt for the LLM"""
        
        # Format search results
        clauses_text = ""
        for i, result in enumerate(search_results, 1):
            clauses_text += f"\nClause {i}:\n"
            clauses_text += f"Source: {result.get('source', 'Unknown')}\n"
            clauses_text += f"Chunk ID: {result.get('chunk_id', 'Unknown')}\n"
            clauses_text += f"Content: {result.get('content', '')}\n"
            clauses_text += f"Relevance Score: {result.get('score', 0)}\n"
            clauses_text += "-" * 50
        
        # Format parsed query
        query_details = f"""
        Age: {parsed_query.get('age', 'Not specified')}
        Procedure: {parsed_query.get('procedure', 'Not specified')}
        Location: {parsed_query.get('location', 'Not specified')}
        Policy Duration: {parsed_query.get('policy_duration', 'Not specified')}
        """
        
        # Add context if available
        context_text = ""
        if context and context.get('previous_queries'):
            context_text = f"\nPrevious conversation context:\n{context['previous_queries']}\n"
        
        prompt = f"""
INSURANCE CLAIM ANALYSIS REQUEST

Original Query: "{query}"

Parsed Query Details:{query_details}

{context_text}

Retrieved Policy Clauses:
{clauses_text}

TASK:
Analyze the above information and provide a structured decision on the insurance claim.

RESPONSE FORMAT:
You must respond with a JSON object in the following exact format:

{{
    "decision": "Approved" or "Rejected",
    "amount": "₹[amount]" or "₹0" if rejected,
    "justification": [
        {{
            "clause_id": "source_file|chunk_id",
            "text": "relevant clause excerpt that supports the decision"
        }}
    ]
}}

ANALYSIS GUIDELINES:
1. Carefully examine each policy clause for relevance to the claim
2. Consider the claimant's age, procedure type, location, and policy duration
3. Check for any exclusions, waiting periods, or coverage limitations
4. Provide clear justification with specific clause references
5. If approved, calculate the appropriate amount based on policy terms
6. If rejected, explain the specific reasons with clause citations

Begin your analysis:
"""
        
        return prompt
    
    def _parse_structured_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the structured JSON response from LLM"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                
                # Validate using Pydantic
                decision = ClaimDecision(**parsed_response)
                return decision.dict()
            else:
                logger.warning("No JSON found in response, creating default structure")
                return self._create_default_response()
                
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            return self._create_default_response()
    
    def _create_default_response(self) -> Dict[str, Any]:
        """Create a default response structure"""
        return {
            "decision": "Error",
            "amount": "₹0",
            "justification": [{
                "clause_id": "system|error",
                "text": "Unable to process the claim due to system error"
            }]
        }
    
    def _generate_natural_response(self, structured_response: Dict[str, Any]) -> str:
        """Generate a natural language response from structured data"""
        decision = structured_response.get('decision', 'Unknown')
        amount = structured_response.get('amount', '₹0')
        justifications = structured_response.get('justification', [])
        
        if decision == 'Approved':
            response = f"✅ **Good news!** Your insurance claim has been **approved** for {amount}.\n\n"
            response += "**Justification:**\n"
            for i, just in enumerate(justifications, 1):
                response += f"{i}. {just.get('text', '')}\n"
        elif decision == 'Rejected':
            response = f"❌ **Unfortunately**, your insurance claim has been **rejected**.\n\n"
            response += "**Reasons for rejection:**\n"
            for i, just in enumerate(justifications, 1):
                response += f"{i}. {just.get('text', '')}\n"
        else:
            response = f"⚠️ **Status**: {decision}\n\n"
            response += "Please review the details below for more information.\n"
        
        return response
    
    def generate_follow_up_response(
        self, 
        query: str, 
        previous_decision: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response for follow-up queries"""
        logger.info(f"Generating follow-up response for: {query}")
        
        try:
            prompt = f"""
FOLLOW-UP QUERY ANALYSIS

Previous Decision: {json.dumps(previous_decision, indent=2)}

Current Query: "{query}"

Context: {json.dumps(context, indent=2)}

TASK:
Provide a helpful response to the follow-up query based on the previous decision and context.
If the query asks for clarification, provide detailed explanations.
If the query requests additional information, search for relevant clauses.

Respond in a natural, helpful manner while maintaining consistency with the previous decision.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful insurance advisor providing follow-up information based on previous claim decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            natural_response = response.choices[0].message.content
            
            return {
                "natural_response": natural_response,
                "structured_response": previous_decision,  # Keep previous decision
                "raw_response": natural_response
            }
            
        except Exception as e:
            logger.error(f"Error generating follow-up response: {str(e)}")
            return {
                "natural_response": f"I apologize, but I couldn't process your follow-up query: {str(e)}",
                "structured_response": {},
                "raw_response": ""
            }