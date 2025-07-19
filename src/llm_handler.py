"""
Enhanced LLM Handler with Multi-Model Support
Supports quick responses and detailed analysis
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Types of responses the LLM can generate"""
    QUICK_EXPLANATION = "quick_explanation"
    DETAILED_ANALYSIS = "detailed_analysis"
    CLAIM_DECISION = "claim_decision"
    POLICY_QUERY = "policy_query"

@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    response_type: ResponseType
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    structured_data: Optional[Dict[str, Any]] = None

class EnhancedLLMHandler:
    """Enhanced LLM Handler with multi-model support"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced LLM handler"""
        self.config = config or {}
        self.primary_model = self.config.get('primary_model', 'groq')
        self.quick_model = self.config.get('quick_model', 'groq-quick')
        self.setup_models()
        
    def setup_models(self):
        """Setup different models for different purposes"""
        try:
            # Primary model for detailed analysis
            self.primary_llm = self._initialize_groq_model()
            
            # Quick model for immediate responses
            self.quick_llm = self._initialize_groq_model(model_name="llama3-8b-8192")
            
            logger.info("Enhanced LLM models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM models: {e}")
            self.primary_llm = None
            self.quick_llm = None
    
    def _initialize_groq_model(self, model_name: str = "llama3-70b-8192"):
        """Initialize Groq model"""
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            client = Groq(api_key=api_key)
            return client
            
        except ImportError:
            logger.error("Groq library not installed. Install with: pip install groq")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Groq model: {e}")
            return None
    
    def get_quick_explanation(self, query: str) -> LLMResponse:
        """Get quick explanation for immediate user feedback"""
        start_time = time.time()
        
        try:
            # Quick response patterns
            quick_patterns = {
                'surgery': "🏥 I'm analyzing your surgery claim. Checking policy coverage, age eligibility, and waiting periods...",
                'claim': "📋 Processing your claim request. Reviewing policy terms and calculating eligible amounts...",
                'dental': "🦷 Checking your dental coverage. Reviewing annual limits and treatment eligibility...",
                'maternity': "👶 Analyzing maternity benefits. Checking waiting periods and coverage terms...",
                'accident': "🚑 Processing accident claim. Reviewing emergency coverage and policy terms...",
                'medication': "💊 Checking medication coverage. Reviewing prescription benefits and limits...",
                'therapy': "🧘 Analyzing therapy coverage. Checking session limits and eligibility criteria...",
                'diagnostic': "🔬 Reviewing diagnostic test coverage. Checking policy limits and pre-authorization requirements...",
                'checkup': "👨‍⚕️ Analyzing health checkup coverage. Reviewing annual limits and preventive care benefits...",
                'emergency': "🚨 Processing emergency claim. Reviewing immediate care coverage and policy terms...",
                'hospitalization': "🏥 Checking hospitalization benefits. Reviewing room rent limits and coverage duration...",
                'outpatient': "🏥 Analyzing outpatient coverage. Checking consultation limits and treatment eligibility..."
            }
            
            # Generate contextual quick response
            query_lower = query.lower()
            explanation = "🔍 I'm analyzing your insurance query. Let me process the information and provide you with a detailed response..."
            
            for keyword, response in quick_patterns.items():
                if keyword in query_lower:
                    explanation = response
                    break
            
            # Add query-specific context
            if any(word in query_lower for word in ['age', 'old', 'years']):
                explanation += " I'll also verify age-related eligibility criteria."
            
            if any(word in query_lower for word in ['policy', 'months', 'years']):
                explanation += " I'll check your policy tenure and waiting periods."
            
            if any(word in query_lower for word in ['amount', 'cost', 'price', 'rupees']):
                explanation += " I'll calculate the eligible claim amount."
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=explanation,
                response_type=ResponseType.QUICK_EXPLANATION,
                confidence=0.9,
                processing_time=processing_time,
                metadata={
                    'query_keywords': [word for word in quick_patterns.keys() if word in query_lower],
                    'model_used': 'pattern_matching',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Quick explanation error: {e}")
            return LLMResponse(
                content="🤖 Processing your query...",
                response_type=ResponseType.QUICK_EXPLANATION,
                confidence=0.5,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def get_detailed_analysis(self, query: str, context: Optional[str] = None) -> LLMResponse:
        """Get detailed analysis with structured claim decision"""
        start_time = time.time()
        
        try:
            # Construct prompt for detailed analysis
            prompt = self._construct_detailed_prompt(query, context)
            
            # Call primary LLM
            if self.primary_llm:
                response = self._call_groq_model(prompt, self.primary_llm)
                structured_data = self._extract_structured_data(response)
            else:
                # Fallback to mock response
                response, structured_data = self._generate_mock_response(query)
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=response,
                response_type=ResponseType.DETAILED_ANALYSIS,
                confidence=0.85,
                processing_time=processing_time,
                metadata={
                    'model_used': 'groq_primary',
                    'has_context': context is not None,
                    'timestamp': datetime.now().isoformat()
                },
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Detailed analysis error: {e}")
            return self._generate_error_response(query, str(e), start_time)
    
    def _construct_detailed_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Construct detailed prompt for LLM"""
        
        base_prompt = f"""
You are an expert insurance claim advisor for Bajaj Allianz. Analyze the following query and provide a structured response.

Query: {query}

Context: {context or "No additional context provided"}

Please provide a JSON response with the following structure:
{{
    "decision": "Approved/Rejected/Under Review",
    "amount": "Claim amount in INR (e.g., ₹50,000)",
    "explanation": "Brief explanation of the decision in 2-3 sentences",
    "justification": [
        {{
            "clause_id": "Policy clause reference",
            "text": "Detailed justification text"
        }}
    ],
    "confidence": 0.85,
    "recommendations": [
        "Any additional recommendations"
    ],
    "next_steps": [
        "Required actions or documents"
    ],
    "coverage_details": {{
        "policy_type": "Health/Motor/Life",
        "coverage_limit": "Annual limit",
        "waiting_period": "Applicable waiting period",
        "exclusions": ["List of relevant exclusions"]
    }}
}}

Consider the following factors:
1. Age and eligibility criteria
2. Policy tenure and waiting periods
3. Coverage limits and exclusions
4. Pre-existing conditions
5. Documentation requirements
6. Claim history and patterns
7. Treatment necessity and medical guidelines
8. Geographic coverage (if applicable)

Provide accurate, helpful, and professional advice. If information is insufficient, mention what additional details are needed.
"""
        
        return base_prompt
    
    def _call_groq_model(self, prompt: str, client, model_name: str = "llama3-70b-8192") -> str:
        """Call Groq model with error handling"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert insurance claim advisor specializing in Indian insurance policies. Provide accurate, helpful responses based on standard insurance practices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
    
    def _extract_structured_data(self, response: str) -> Dict[str, Any]:
        """Extract structured data from LLM response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # If no JSON found, parse manually
            return self._parse_response_manually(response)
            
        except Exception as e:
            logger.error(f"Failed to extract structured data: {e}")
            return self._generate_fallback_structure(response)
    
    def _parse_response_manually(self, response: str) -> Dict[str, Any]:
        """Parse response manually if JSON extraction fails"""
        
        # Extract decision
        decision = "Under Review"
        if "approved" in response.lower():
            decision = "Approved"
        elif "rejected" in response.lower() or "denied" in response.lower():
            decision = "Rejected"
        
        # Extract amount
        amount_match = re.search(r'₹[\d,]+|INR\s*[\d,]+|rupees?\s*[\d,]+', response, re.IGNORECASE)
        amount = amount_match.group(0) if amount_match else "Amount to be determined"
        
        # Extract explanation (first paragraph)
        explanation = response.split('\n')[0][:200] + "..." if len(response.split('\n')[0]) > 200 else response.split('\n')[0]
        
        return {
            "decision": decision,
            "amount": amount,
            "explanation": explanation,
            "justification": [
                {
                    "clause_id": "PARSED_001",
                    "text": "Based on policy analysis and claim evaluation"
                }
            ],
            "confidence": 0.75,
            "recommendations": ["Please provide additional documentation if required"],
            "next_steps": ["Review the decision", "Submit any missing documents"],
            "coverage_details": {
                "policy_type": "Health",
                "coverage_limit": "As per policy terms",
                "waiting_period": "Standard waiting periods apply",
                "exclusions": []
            }
        }
    
    def _generate_fallback_structure(self, response: str) -> Dict[str, Any]:
        """Generate fallback structure when parsing fails"""
        return {
            "decision": "Under Review",
            "amount": "₹0",
            "explanation": "Unable to process the claim automatically. Manual review required.",
            "justification": [
                {
                    "clause_id": "FALLBACK_001",
                    "text": "System encountered an error during processing. Please contact customer service."
                }
            ],
            "confidence": 0.3,
            "recommendations": ["Contact customer service for manual review"],
            "next_steps": ["Call customer service", "Submit claim manually"],
            "coverage_details": {
                "policy_type": "Unknown",
                "coverage_limit": "Unknown",
                "waiting_period": "Unknown",
                "exclusions": []
            }
        }
    
    def _generate_mock_response(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate mock response when LLM is unavailable"""
        
        # Analyze query for mock response
        query_lower = query.lower()
        
        # Mock decision logic
        if any(word in query_lower for word in ['emergency', 'accident', 'urgent']):
            decision = "Approved"
            amount = "₹75,000"
            explanation = "Emergency claim approved based on policy terms."
        elif any(word in query_lower for word in ['cosmetic', 'aesthetic', 'beauty']):
            decision = "Rejected"
            amount = "₹0"
            explanation = "Cosmetic procedures are not covered under this policy."
        else:
            decision = "Approved"
            amount = "₹50,000"
            explanation = "Claim approved after policy verification."
        
        response_text = f"Mock Analysis: {explanation}"
        
        structured_data = {
            "decision": decision,
            "amount": amount,
            "explanation": explanation,
            "justification": [
                {
                    "clause_id": "MOCK_001",
                    "text": "Mock response generated for demonstration purposes"
                }
            ],
            "confidence": 0.6,
            "recommendations": ["This is a mock response for demo purposes"],
            "next_steps": ["Configure actual LLM for real processing"],
            "coverage_details": {
                "policy_type": "Health",
                "coverage_limit": "₹5,00,000",
                "waiting_period": "30 days",
                "exclusions": ["Pre-existing conditions", "Cosmetic procedures"]
            }
        }
        
        return response_text, structured_data
    
    def _generate_error_response(self, query: str, error: str, start_time: float) -> LLMResponse:
        """Generate error response"""
        return LLMResponse(
            content=f"❌ Error processing query: {error}",
            response_type=ResponseType.DETAILED_ANALYSIS,
            confidence=0.0,
            processing_time=time.time() - start_time,
            metadata={
                'error': error,
                'query': query,
                'timestamp': datetime.now().isoformat()
            },
            structured_data={
                "decision": "Error",
                "amount": "₹0",
                "explanation": f"System error: {error}",
                "justification": [],
                "confidence": 0.0,
                "recommendations": ["Please try again or contact support"],
                "next_steps": ["Retry the query", "Contact technical support"]
            }
        )
    
    def process_claim_query(self, query: str, context: Optional[str] = None) -> Tuple[LLMResponse, LLMResponse]:
        """Process claim query with both quick and detailed responses"""
        
        # Get quick explanation first
        quick_response = self.get_quick_explanation(query)
        
        # Get detailed analysis
        detailed_response = self.get_detailed_analysis(query, context)
        
        return quick_response, detailed_response
    
    def get_contextual_help(self, query: str) -> str:
        """Get contextual help based on query"""
        query_lower = query.lower()
        
        help_responses = {
            'surgery': "For surgery claims, ensure you have: Pre-authorization, discharge summary, bills, and medical reports.",
            'dental': "Dental claims require: Treatment plan, bills, X-rays (if applicable), and dentist's prescription.",
            'maternity': "Maternity claims need: Discharge summary, delivery bills, baby's birth certificate, and pre-natal records.",
            'accident': "Accident claims require: Police report (if applicable), medical reports, bills, and accident details.",
            'medication': "Medicine claims need: Prescription, purchase bills, and medical reports supporting the treatment.",
            'emergency': "Emergency claims require: Admission details, discharge summary, bills, and medical emergency proof."
        }
        
        for keyword, help_text in help_responses.items():
            if keyword in query_lower:
                return help_text
        
        return "Ensure you have all relevant medical documents, bills, and policy details for faster processing."
    
    def validate_claim_data(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data structure"""
        required_fields = ['decision', 'amount', 'explanation', 'justification', 'confidence']
        
        for field in required_fields:
            if field not in claim_data:
                claim_data[field] = self._get_default_value(field)
        
        # Ensure confidence is between 0 and 1
        if isinstance(claim_data.get('confidence'), (int, float)):
            claim_data['confidence'] = max(0, min(1, claim_data['confidence']))
        
        return claim_data
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'decision': 'Under Review',
            'amount': '₹0',
            'explanation': 'Processing...',
            'justification': [],
            'confidence': 0.5,
            'recommendations': [],
            'next_steps': []
        }
        return defaults.get(field, '')

# Legacy LLMHandler class for backward compatibility
class LLMHandler:
    """Legacy LLM Handler for backward compatibility"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.enhanced_handler = EnhancedLLMHandler(config)
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Get response using enhanced handler"""
        _, detailed_response = self.enhanced_handler.process_claim_query(query)
        return detailed_response.structured_data or {}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query using enhanced handler"""
        return self.get_response(query)
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Predict using enhanced handler"""
        return self.get_response(query)