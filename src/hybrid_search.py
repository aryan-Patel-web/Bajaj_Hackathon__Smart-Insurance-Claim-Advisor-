"""
Hybrid Search Module for Smart Insurance Claim Advisor
Combines semantic vector search with metadata filtering for optimal retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import AstraDB
from langchain.schema import Document
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result with relevance scoring"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    relevance_score: float
    clause_id: str
    source_file: str
    chunk_id: str

class HybridSearchEngine:
    """
    Advanced hybrid search engine combining:
    1. Semantic vector similarity search
    2. Metadata-based filtering
    3. Relevance scoring and ranking
    4. Query expansion and refinement
    """
    
    def __init__(self, vector_store: AstraDB, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid search engine
        
        Args:
            vector_store: Astra DB vector store instance
            embedding_model: SentenceTransformer model for embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(embedding_model)
        self.search_weights = {
            'semantic': 0.7,
            'metadata': 0.3
        }
        logger.info(f"Initialized HybridSearchEngine with model: {embedding_model}")
    
    def search(self, 
               query: str, 
               structured_query: Dict[str, Any],
               k: int = 10,
               similarity_threshold: float = 0.6) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and metadata filtering
        
        Args:
            query: Natural language query
            structured_query: Parsed query with age, procedure, location, etc.
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of ranked SearchResult objects
        """
        logger.info(f"Performing hybrid search for query: {query}")
        
        try:
            # Step 1: Semantic vector search
            semantic_results = self._semantic_search(query, k * 2)
            
            # Step 2: Metadata filtering
            filtered_results = self._metadata_filter(semantic_results, structured_query)
            
            # Step 3: Relevance scoring
            scored_results = self._calculate_relevance_scores(filtered_results, structured_query)
            
            # Step 4: Re-ranking and final selection
            final_results = self._rerank_results(scored_results, similarity_threshold)
            
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def _semantic_search(self, query: str, k: int) -> List[Document]:
        """
        Perform semantic vector similarity search
        
        Args:
            query: Natural language query
            k: Number of results to retrieve
            
        Returns:
            List of Document objects from vector store
        """
        try:
            # Expand query with insurance-specific terms
            expanded_query = self._expand_query(query)
            
            # Perform vector similarity search
            docs = self.vector_store.similarity_search(
                expanded_query, 
                k=k
            )
            
            logger.info(f"Semantic search retrieved {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with insurance domain-specific terms
        
        Args:
            query: Original query
            
        Returns:
            Expanded query string
        """
        # Insurance domain keywords mapping
        expansions = {
            'surgery': ['operation', 'procedure', 'medical intervention', 'treatment'],
            'claim': ['reimbursement', 'coverage', 'benefit', 'payout'],
            'policy': ['insurance', 'coverage', 'plan', 'benefits'],
            'hospital': ['medical facility', 'healthcare center', 'clinic'],
            'age': ['years old', 'age group', 'demographic'],
            'location': ['city', 'state', 'region', 'area']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for key, synonyms in expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def _metadata_filter(self, 
                        documents: List[Document], 
                        structured_query: Dict[str, Any]) -> List[Document]:
        """
        Filter documents based on metadata criteria
        
        Args:
            documents: List of documents from semantic search
            structured_query: Structured query parameters
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.metadata
            
            # Apply metadata filters
            if self._matches_metadata_criteria(metadata, structured_query):
                filtered_docs.append(doc)
        
        logger.info(f"Metadata filtering retained {len(filtered_docs)} documents")
        return filtered_docs
    
    def _matches_metadata_criteria(self, 
                                 metadata: Dict[str, Any], 
                                 structured_query: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches query criteria
        
        Args:
            metadata: Document metadata
            structured_query: Structured query parameters
            
        Returns:
            Boolean indicating if criteria match
        """
        # Age-based filtering
        if 'age' in structured_query and structured_query['age']:
            age = structured_query['age']
            doc_content = metadata.get('content', '').lower()
            
            # Check for age-specific clauses
            if age < 18 and 'minor' not in doc_content and 'child' not in doc_content:
                if 'adult' in doc_content:
                    return False
            elif age >= 60 and 'senior' not in doc_content and 'elderly' not in doc_content:
                if 'young' in doc_content:
                    return False
        
        # Procedure-based filtering
        if 'procedure' in structured_query and structured_query['procedure']:
            procedure = structured_query['procedure'].lower()
            doc_content = metadata.get('content', '').lower()
            
            # Check for procedure mentions
            if procedure in doc_content or any(proc in doc_content for proc in self._get_procedure_synonyms(procedure)):
                return True
        
        # Location-based filtering
        if 'location' in structured_query and structured_query['location']:
            location = structured_query['location'].lower()
            doc_content = metadata.get('content', '').lower()
            
            # Check for location-specific clauses
            if location in doc_content:
                return True
        
        # Policy duration filtering
        if 'policy_duration' in structured_query and structured_query['policy_duration']:
            duration = structured_query['policy_duration']
            doc_content = metadata.get('content', '').lower()
            
            # Check for waiting period clauses
            if duration < 30 and 'waiting period' in doc_content:
                return True
        
        # Default: include document if no specific filters exclude it
        return True
    
    def _get_procedure_synonyms(self, procedure: str) -> List[str]:
        """
        Get synonyms for medical procedures
        
        Args:
            procedure: Medical procedure name
            
        Returns:
            List of synonyms
        """
        synonyms_map = {
            'surgery': ['operation', 'surgical procedure', 'intervention'],
            'knee': ['joint', 'orthopedic', 'leg'],
            'heart': ['cardiac', 'cardiovascular', 'chest'],
            'eye': ['ophthalmic', 'vision', 'ocular'],
            'brain': ['neurological', 'neuro', 'head'],
            'cancer': ['oncology', 'tumor', 'malignancy'],
            'diabetes': ['diabetic', 'blood sugar', 'glucose']
        }
        
        return synonyms_map.get(procedure.lower(), [])
    
    def _calculate_relevance_scores(self, 
                                  documents: List[Document], 
                                  structured_query: Dict[str, Any]) -> List[SearchResult]:
        """
        Calculate relevance scores for filtered documents
        
        Args:
            documents: Filtered documents
            structured_query: Structured query parameters
            
        Returns:
            List of SearchResult objects with relevance scores
        """
        search_results = []
        
        for doc in documents:
            # Extract metadata
            metadata = doc.metadata
            content = doc.page_content
            
            # Calculate various scoring factors
            semantic_score = self._calculate_semantic_score(content, structured_query)
            metadata_score = self._calculate_metadata_score(metadata, structured_query)
            content_quality_score = self._calculate_content_quality_score(content)
            
            # Combined relevance score
            relevance_score = (
                self.search_weights['semantic'] * semantic_score +
                self.search_weights['metadata'] * metadata_score +
                0.1 * content_quality_score
            )
            
            # Create SearchResult object
            search_result = SearchResult(
                content=content,
                metadata=metadata,
                similarity_score=metadata.get('similarity_score', 0.0),
                relevance_score=relevance_score,
                clause_id=f"{metadata.get('source_file', 'unknown')}|{metadata.get('chunk_id', 'unknown')}",
                source_file=metadata.get('source_file', 'unknown'),
                chunk_id=metadata.get('chunk_id', 'unknown')
            )
            
            search_results.append(search_result)
        
        return search_results
    
    def _calculate_semantic_score(self, content: str, structured_query: Dict[str, Any]) -> float:
        """
        Calculate semantic relevance score
        
        Args:
            content: Document content
            structured_query: Structured query parameters
            
        Returns:
            Semantic relevance score (0-1)
        """
        score = 0.0
        content_lower = content.lower()
        
        # Procedure relevance
        if 'procedure' in structured_query and structured_query['procedure']:
            procedure = structured_query['procedure'].lower()
            if procedure in content_lower:
                score += 0.4
            elif any(syn in content_lower for syn in self._get_procedure_synonyms(procedure)):
                score += 0.2
        
        # Age relevance
        if 'age' in structured_query and structured_query['age']:
            age = structured_query['age']
            if age < 18 and any(term in content_lower for term in ['minor', 'child', 'pediatric']):
                score += 0.2
            elif age >= 60 and any(term in content_lower for term in ['senior', 'elderly', 'geriatric']):
                score += 0.2
            elif 18 <= age < 60 and 'adult' in content_lower:
                score += 0.1
        
        # Location relevance
        if 'location' in structured_query and structured_query['location']:
            location = structured_query['location'].lower()
            if location in content_lower:
                score += 0.2
        
        # Policy duration relevance
        if 'policy_duration' in structured_query and structured_query['policy_duration']:
            duration = structured_query['policy_duration']
            if duration < 30 and 'waiting period' in content_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any], structured_query: Dict[str, Any]) -> float:
        """
        Calculate metadata-based relevance score
        
        Args:
            metadata: Document metadata
            structured_query: Structured query parameters
            
        Returns:
            Metadata relevance score (0-1)
        """
        score = 0.0
        
        # Document type relevance
        doc_type = metadata.get('document_type', '').lower()
        if doc_type in ['policy', 'terms', 'conditions', 'coverage']:
            score += 0.3
        
        # Section relevance
        section = metadata.get('section', '').lower()
        if any(term in section for term in ['coverage', 'benefits', 'claims', 'exclusions']):
            score += 0.2
        
        # Recency bonus
        if 'date_created' in metadata:
            # Prefer more recent documents
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_content_quality_score(self, content: str) -> float:
        """
        Calculate content quality score
        
        Args:
            content: Document content
            
        Returns:
            Content quality score (0-1)
        """
        score = 0.0
        
        # Length penalty for very short or very long content
        length = len(content)
        if 100 <= length <= 2000:
            score += 0.3
        elif 50 <= length < 100 or 2000 < length <= 5000:
            score += 0.2
        elif length < 50:
            score += 0.1
        
        # Structure bonus
        if any(marker in content for marker in ['•', '-', '1.', '2.', 'Section', 'Clause']):
            score += 0.2
        
        # Insurance terminology bonus
        insurance_terms = ['coverage', 'premium', 'deductible', 'benefit', 'claim', 'policy', 'insured']
        term_count = sum(1 for term in insurance_terms if term in content.lower())
        score += min(term_count * 0.1, 0.5)
        
        return min(score, 1.0)
    
    def _rerank_results(self, 
                       search_results: List[SearchResult], 
                       similarity_threshold: float) -> List[SearchResult]:
        """
        Re-rank search results based on combined scores
        
        Args:
            search_results: List of SearchResult objects
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Re-ranked and filtered list of SearchResult objects
        """
        # Filter by similarity threshold
        filtered_results = [
            result for result in search_results 
            if result.similarity_score >= similarity_threshold
        ]
        
        # Sort by relevance score (descending)
        ranked_results = sorted(
            filtered_results, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Remove duplicates based on content similarity
        unique_results = self._remove_duplicate_results(ranked_results)
        
        logger.info(f"Re-ranking produced {len(unique_results)} unique results")
        return unique_results
    
    def _remove_duplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate or highly similar results
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            List with duplicates removed
        """
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Create a normalized version of content for comparison
            normalized_content = re.sub(r'\s+', ' ', result.content.lower().strip())
            
            # Check if we've seen similar content
            is_duplicate = False
            for seen_content in seen_contents:
                if self._calculate_content_similarity(normalized_content, seen_content) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.add(normalized_content)
        
        return unique_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity for now
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_search_explanation(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Generate explanation for search results
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with search explanation
        """
        if not results:
            return {"message": "No relevant results found"}
        
        explanation = {
            "total_results": len(results),
            "avg_relevance_score": sum(r.relevance_score for r in results) / len(results),
            "avg_similarity_score": sum(r.similarity_score for r in results) / len(results),
            "source_files": list(set(r.source_file for r in results)),
            "top_result_score": results[0].relevance_score if results else 0,
            "search_strategy": "Hybrid semantic + metadata filtering"
        }
        
        return explanation