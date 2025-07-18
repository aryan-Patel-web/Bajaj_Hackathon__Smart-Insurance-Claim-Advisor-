"""
Query parsing and structuring for insurance claim queries.
Extracts structured fields: age, procedure, location, policy duration.
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field
import json

from utils.logging_config import logger

@dataclass
class ParsedQuery:
    """Structured representation of a parsed insurance query."""
    original_query: str
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    policy_type: Optional[str] = None
    claim_amount: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class QueryParser:
    """Advanced query parser for insurance claim queries."""
    
    def __init__(self):
        """Initialize parser with patterns and dictionaries."""
        self.age_patterns = [
            r'(\d+)\s*(?:year|yr|y)\s*(?:old|o)',
            r'(\d+)\s*(?:M|F|male|female)',
            r'age\s*(?:is|:)?\s*(\d+)',
            r'(\d+)\s*(?:aged|age)',
            r'(\d+)(?=\s*(?:year|yr|y|M|F|male|female))',
        ]
        
        self.gender_patterns = [
            r'\b(male|female|M|F)\b',
            r'\b(man|woman|boy|girl)\b',
        ]
        
        self.location_patterns = [
            r'\b(Mumbai|Delhi|Bangalore|Hyderabad|Ahmedabad|Chennai|Kolkata|Surat|Pune|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Pimpri|Patna|Vadodara|Ghaziabad|Ludhiana|Agra|Nashik|Faridabad|Meerut|Rajkot|Kalyan|Vasai|Varanasi|Srinagar|Aurangabad|Dhanbad|Amritsar|Navi Mumbai|Allahabad|Ranchi|Howrah|Coimbatore|Jabalpur|Gwalior|Vijayawada|Jodhpur|Madurai|Raipur|Kota|Chandigarh|Guwahati|Solapur|Hubli|Tiruchirappalli|Bareilly|Mysore|Tiruppur|Gurgaon|Aligarh|Jalandhar|Bhubaneswar|Salem|Warangal|Guntur|Bhiwandi|Saharanpur|Gorakhpur|Bikaner|Amravati|Noida|Jamshedpur|Bhilai|Cuttack|Firozabad|Kochi|Nellore|Bhavnagar|Dehradun|Durgapur|Asansol|Rourkela|Nanded|Kolhapur|Ajmer|Akola|Gulbarga|Jamnagar|Ujjain|Loni|Siliguri|Jhansi|Ulhasnagar|Jammu|Sangli|Mangalore|Erode|Belgaum|Ambattur|Tirunelveli|Malegaon|Gaya|Jalgaon|Udaipur|Maheshtala)\b',
            r'\bin\s+([A-Z][a-z]+)',
        ]
        
        self.procedure_patterns = [
            r'\b(surgery|operation|procedure|treatment|therapy|diagnostic|test|examination|consultation|checkup)\b',
            r'\b(knee|hip|heart|cardiac|brain|spine|eye|dental|orthopedic|neurological|gynecological|urological|dermatological)\b',
            r'\b(bypass|replacement|transplant|biopsy|scan|MRI|CT|X-ray|ultrasound|endoscopy|colonoscopy|mammography)\b',
        ]
        
        self.policy_duration_patterns = [
            r'(\d+)\s*(?:month|mon|m)\s*(?:old|policy|duration)?',
            r'(\d+)\s*(?:year|yr|y)\s*(?:old|policy|duration)?',
            r'policy\s*(?:is|for|of)?\s*(\d+)\s*(?:month|year)',
            r'(\d+)\s*(?:month|year)\s*policy',
        ]
        
        self.claim_amount_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(?:Rs|INR|rupees?)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:Rs|INR|rupees?)',
            r'claim\s*(?:amount|of)?\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
        ]
        
        # Common medical procedures and their synonyms
        self.procedure_synonyms = {
            'knee': ['knee replacement', 'knee surgery', 'ACL', 'meniscus', 'arthroscopy'],
            'heart': ['cardiac', 'bypass', 'angioplasty', 'stent', 'pacemaker'],
            'eye': ['cataract', 'lasik', 'retinal', 'glaucoma'],
            'dental': ['tooth', 'root canal', 'extraction', 'implant', 'crown'],
            'spine': ['spinal', 'disc', 'vertebrae', 'back surgery'],
        }
        
        # Indian cities and states
        self.indian_locations = {
            'Mumbai': ['Bombay', 'Mumbai'],
            'Delhi': ['New Delhi', 'Delhi', 'NCR'],
            'Bangalore': ['Bengaluru', 'Bangalore'],
            'Hyderabad': ['Hyderabad', 'Secunderabad'],
            'Chennai': ['Madras', 'Chennai'],
            'Kolkata': ['Calcutta', 'Kolkata'],
            'Pune': ['Pune', 'Poona'],
        }
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language query into structured format.
        
        Args:
            query: Natural language query
            
        Returns:
            ParsedQuery object with extracted information
        """
        try:
            logger.info(f"Parsing query: {query[:100]}...")
            
            # Clean and normalize query
            normalized_query = self._normalize_query(query)
            
            # Extract structured fields
            age = self._extract_age(normalized_query)
            gender = self._extract_gender(normalized_query)
            procedure = self._extract_procedure(normalized_query)
            location = self._extract_location(normalized_query)
            policy_duration = self._extract_policy_duration(normalized_query)
            claim_amount = self._extract_claim_amount(normalized_query)
            policy_type = self._extract_policy_type(normalized_query)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                age, gender, procedure, location, policy_duration, claim_amount
            )
            
            parsed_query = ParsedQuery(
                original_query=query,
                age=age,
                gender=gender,
                procedure=procedure,
                location=location,
                policy_duration=policy_duration,
                policy_type=policy_type,
                claim_amount=claim_amount,
                additional_info=self._extract_additional_info(normalized_query),
                confidence=confidence
            )
            
            logger.info(f"Parsed query with confidence: {confidence:.2f}")
            return parsed_query
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            return ParsedQuery(
                original_query=query,
                confidence=0.0
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better parsing."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Normalize common abbreviations
        query = re.sub(r'\byrs?\b', 'year', query)
        query = re.sub(r'\bmons?\b', 'month', query)
        query = re.sub(r'\bm\b(?!\w)', 'male', query)
        query = re.sub(r'\bf\b(?!\w)', 'female', query)
        
        return query.strip()
    
    def _extract_age(self, query: str) -> Optional[int]:
        """Extract age from query."""
        for pattern in self.age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 120:  # Reasonable age range
                    return age
        return None
    
    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract gender from query."""
        for pattern in self.gender_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                gender = match.group(1).lower()
                if gender in ['male', 'm', 'man', 'boy']:
                    return 'male'
                elif gender in ['female', 'f', 'woman', 'girl']:
                    return 'female'
        return None
    
    def _extract_procedure(self, query: str) -> Optional[str]:
        """Extract medical procedure from query."""
        procedures = []
        
        for pattern in self.procedure_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            procedures.extend(matches)
        
        # Look for specific procedure synonyms
        for main_procedure, synonyms in self.procedure_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query:
                    procedures.append(main_procedure)
                    break
        
        if procedures:
            # Return the most specific procedure found
            return ', '.join(set(procedures))
        
        return None
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query."""
        for pattern in self.location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1) if match.groups() else match.group(0)
                # Normalize location name
                for standard_name, variants in self.indian_locations.items():
                    if location.lower() in [v.lower() for v in variants]:
                        return standard_name
                return location.title()
        return None
    
    def _extract_policy_duration(self, query: str) -> Optional[str]:
        """Extract policy duration from query."""
        for pattern in self.policy_duration_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                duration = match.group(1)
                # Determine if it's months or years
                if 'month' in match.group(0).lower():
                    return f"{duration} months"
                elif 'year' in match.group(0).lower():
                    return f"{duration} years"
                else:
                    # Default to months if under 12, years if over
                    num = int(duration)
                    return f"{duration} months" if num <= 12 else f"{duration} years"
        return None
    
    def _extract_claim_amount(self, query: str) -> Optional[str]:
        """Extract claim amount from query."""
        for pattern in self.claim_amount_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount = match.group(1)
                # Clean amount (remove commas, etc.)
                amount = amount.replace(',', '')
                return f"₹{amount}"
        return None
    
    def _extract_policy_type(self, query: str) -> Optional[str]:
        """Extract policy type from query."""
        policy_types = [
            'health', 'medical', 'life', 'term', 'whole life',
            'critical illness', 'accident', 'disability', 'travel'
        ]
        
        for policy_type in policy_types:
            if policy_type in query.lower():
                return policy_type
        
        return None
    
    def _extract_additional_info(self, query: str) -> Dict[str, Any]:
        """Extract additional information from query."""
        additional_info = {}
        
        # Check for urgency keywords
        urgency_keywords = ['urgent', 'emergency', 'immediate', 'asap', 'quickly']
        if any(keyword in query.lower() for keyword in urgency_keywords):
            additional_info['urgency'] = True
        
        # Check for prior conditions
        prior_keywords = ['previous', 'prior', 'existing', 'history', 'before']
        if any(keyword in query.lower() for keyword in prior_keywords):
            additional_info['prior_conditions'] = True
        
        # Check for family history
        family_keywords = ['family', 'hereditary', 'genetic', 'inherited']
        if any(keyword in query.lower() for keyword in family_keywords):
            additional_info['family_history'] = True
        
        return additional_info
    
    def _calculate_confidence(self, age: Optional[int], gender: Optional[str], 
                            procedure: Optional[str], location: Optional[str],
                            policy_duration: Optional[str], claim_amount: Optional[str]) -> float:
        """Calculate confidence score based on extracted information."""
        confidence = 0.0
        total_fields = 6
        
        if age is not None:
            confidence += 1.0
        if gender is not None:
            confidence += 1.0
        if procedure is not None:
            confidence += 1.5  # Procedure is more important
        if location is not None:
            confidence += 1.0
        if policy_duration is not None:
            confidence += 1.0
        if claim_amount is not None:
            confidence += 0.5
        
        # Normalize to 0-1 scale
        return min(confidence / total_fields, 1.0)
    
    def to_dict(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Convert ParsedQuery to dictionary."""
        return {
            'original_query': parsed_query.original_query,
            'age': parsed_query.age,
            'gender': parsed_query.gender,
            'procedure': parsed_query.procedure,
            'location': parsed_query.location,
            'policy_duration': parsed_query.policy_duration,
            'policy_type': parsed_query.policy_type,
            'claim_amount': parsed_query.claim_amount,
            'additional_info': parsed_query.additional_info,
            'confidence': parsed_query.confidence
        }
    
    def validate_query(self, parsed_query: ParsedQuery) -> List[str]:
        """Validate parsed query and return list of issues."""
        issues = []
        
        if parsed_query.confidence < 0.3:
            issues.append("Low confidence in query parsing")
        
        if not parsed_query.procedure:
            issues.append("No medical procedure identified")
        
        if not parsed_query.age:
            issues.append("Age not specified")
        
        if not parsed_query.policy_duration:
            issues.append("Policy duration not specified")
        
        return issues

# Global query parser instance
query_parser = QueryParser()