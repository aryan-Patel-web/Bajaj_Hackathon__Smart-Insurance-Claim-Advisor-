import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

# Temporary VectorStore implementation if src/vector_store.py is missing or incomplete
class VectorStore:
    def store_conversation_turn(self, turn_id, content, metadata):
        pass  # Implement storing logic or leave as placeholder

    def search_conversations(self, query, limit=5):
        return []  # Implement search logic or leave as placeholder

vector_store = VectorStore()  # Create the global instance

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    session_id: str
    timestamp: str
    user_query: str
    parsed_query: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    llm_response: Dict[str, Any]
    feedback: Optional[str] = None

@dataclass
class ConversationSession:
    """Represents a complete conversation session"""
    session_id: str
    start_time: str
    last_activity: str
    turns: List[ConversationTurn]
    user_context: Dict[str, Any]

class ConversationManager:
    """Manages conversation memory and context for insurance claim processing"""

    def __init__(self):
        """Initialize the conversation manager"""
        self.vector_store = vector_store
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_session_age = timedelta(hours=24)  # Sessions expire after 24 hours
        self.max_turns_per_session = 50

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        current_time = datetime.now().isoformat()

        session = ConversationSession(
            session_id=session_id,
            start_time=current_time,
            last_activity=current_time,
            turns=[],
            user_context={}
        )

        self.sessions[session_id] = session
        logger.info(f"Created new conversation session: {session_id}")

        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session"""
        session = self.sessions.get(session_id)

        if session and self._is_session_expired(session):
            logger.info(f"Session {session_id} expired, removing")
            del self.sessions[session_id]
            return None

        return session

    def update_memory(
        self,
        session_id: str,
        user_query: str,
        llm_response: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        parsed_query: Optional[Dict[str, Any]] = None
    ) -> str:
        """Update conversation memory with a new turn"""

        # Get or create session
        session = self.get_session(session_id)
        if session is None:
            session_id = self.create_session(session_id)
            session = self.get_session(session_id)
        if session is None:
            logger.error("Failed to create or retrieve session")
            return ""

        # Create new turn
        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            parsed_query=parsed_query or {},
            search_results=search_results,
            llm_response=llm_response
        )

        # Add turn to session
        session.turns.append(turn)
        session.last_activity = datetime.now().isoformat()

        # Update user context
        self._update_user_context(session, parsed_query or {})

        # Limit session turns
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]

        # Store in vector store for long-term memory
        self._store_turn_in_vector_store(turn)

        logger.info(f"Updated memory for session {session_id}, turn {turn_id}")

        return turn_id

    def get_context(
        self,
        session_id: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Get conversation context for query processing"""

        session = self.get_session(session_id)
        if session is None:
            return {
                'session_id': session_id,
                'previous_queries': [],
                'user_context': {},
                'conversation_summary': ''
            }

        # Get recent turns
        recent_turns = session.turns[-5:]  # Last 5 turns

        # Extract previous queries and responses
        previous_queries = []
        for turn in recent_turns:
            previous_queries.append({
                'query': turn.user_query,
                'response': turn.llm_response.get('natural_response', ''),
                'decision': turn.llm_response.get('structured_response', {}).get('decision', ''),
                'timestamp': turn.timestamp
            })

        # Generate conversation summary
        conversation_summary = self._generate_conversation_summary(session)

        return {
            'session_id': session_id,
            'previous_queries': previous_queries,
            'user_context': session.user_context,
            'conversation_summary': conversation_summary,
            'turn_count': len(session.turns),
            'session_start': session.start_time
        }

    def _update_user_context(self, session: ConversationSession, parsed_query: Dict[str, Any]):
        """Update user context based on parsed query"""
        if not parsed_query:
            return

        # Update user demographics
        if 'age' in parsed_query and parsed_query['age']:
            session.user_context['age'] = parsed_query['age']

        if 'gender' in parsed_query and parsed_query['gender']:
            session.user_context['gender'] = parsed_query['gender']

        if 'location' in parsed_query and parsed_query['location']:
            session.user_context['location'] = parsed_query['location']

        # Update policy information
        if 'policy_duration' in parsed_query and parsed_query['policy_duration']:
            session.user_context['policy_duration'] = parsed_query['policy_duration']

        # Track procedures mentioned
        if 'procedure' in parsed_query and parsed_query['procedure']:
            if 'procedures' not in session.user_context:
                session.user_context['procedures'] = []

            procedure = parsed_query['procedure']
            if procedure not in session.user_context['procedures']:
                session.user_context['procedures'].append(procedure)

        # Update last activity
        session.user_context['last_update'] = datetime.now().isoformat()

    # ...existing code unchanged...
    def _generate_conversation_summary(self, session: ConversationSession) -> str:
        """Generate a summary of the conversation"""
        if not session.turns:
            return "No previous conversation"
        
        summary_parts = []
        
        # Add user context
        if session.user_context:
            context_parts = []
            if 'age' in session.user_context:
                context_parts.append(f"age {session.user_context['age']}")
            if 'gender' in session.user_context:
                context_parts.append(session.user_context['gender'])
            if 'location' in session.user_context:
                context_parts.append(f"in {session.user_context['location']}")
            if 'procedures' in session.user_context:
                context_parts.append(f"procedures: {', '.join(session.user_context['procedures'])}")
            
            if context_parts:
                summary_parts.append(f"User context: {', '.join(context_parts)}")
        
        # Add recent decisions
        recent_decisions = []
        for turn in session.turns[-3:]:  # Last 3 turns
            decision = turn.llm_response.get('structured_response', {}).get('decision', '')
            if decision:
                recent_decisions.append(decision)
        
        if recent_decisions:
            summary_parts.append(f"Recent decisions: {', '.join(recent_decisions)}")
        
        return "; ".join(summary_parts) if summary_parts else "Ongoing conversation"
    
    def _store_turn_in_vector_store(self, turn: ConversationTurn):
        """Store conversation turn in vector store for long-term memory"""
        try:
            # Create a searchable representation of the turn
            turn_text = f"Query: {turn.user_query}\n"
            turn_text += f"Response: {turn.llm_response.get('natural_response', '')}\n"
            turn_text += f"Decision: {turn.llm_response.get('structured_response', {}).get('decision', '')}\n"
            turn_text += f"Timestamp: {turn.timestamp}"
            
            # Create metadata
            metadata = {
                'type': 'conversation_turn',
                'session_id': turn.session_id,
                'turn_id': turn.turn_id,
                'timestamp': turn.timestamp,
                'user_query': turn.user_query,
                'decision': turn.llm_response.get('structured_response', {}).get('decision', ''),
                'source': f"conversation_{turn.session_id}"
            }
            
            # Store in vector store
            self.vector_store.store_conversation_turn(
                turn_id=turn.turn_id,
                content=turn_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error storing turn in vector store: {str(e)}")
    
    def _is_session_expired(self, session: ConversationSession) -> bool:
        """Check if a session has expired"""
        try:
            last_activity = datetime.fromisoformat(session.last_activity)
            return datetime.now() - last_activity > self.max_session_age
        except:
            return True
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get formatted session history"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        history = []
        for turn in session.turns:
            history.append({
                'turn_id': turn.turn_id,
                'timestamp': turn.timestamp,
                'user_query': turn.user_query,
                'response': turn.llm_response.get('natural_response', ''),
                'decision': turn.llm_response.get('structured_response', {}).get('decision', ''),
                'amount': turn.llm_response.get('structured_response', {}).get('amount', ''),
                'feedback': turn.feedback
            })
        
        return history
    
    def add_feedback(self, session_id: str, turn_id: str, feedback: str) -> bool:
        """Add feedback to a specific turn"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        for turn in session.turns:
            if turn.turn_id == turn_id:
                turn.feedback = feedback
                logger.info(f"Added feedback to turn {turn_id} in session {session_id}")
                return True
        
        return False
    
    def get_similar_conversations(
        self, 
        current_query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar conversations from vector store"""
        try:
            # Search for similar conversation turns
            similar_turns = self.vector_store.search_conversations(
                query=current_query,
                limit=limit
            )
            
            return similar_turns
            
        except Exception as e:
            logger.error(f"Error getting similar conversations: {str(e)}")
            return []
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data for analysis or backup"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Convert to dict
        session_dict = asdict(session)
        
        # Add statistics
        session_dict['statistics'] = {
            'total_turns': len(session.turns),
            'approved_claims': sum(1 for turn in session.turns 
                                 if turn.llm_response.get('structured_response', {}).get('decision') == 'Approved'),
            'rejected_claims': sum(1 for turn in session.turns 
                                 if turn.llm_response.get('structured_response', {}).get('decision') == 'Rejected'),
            'duration_minutes': self._calculate_session_duration(session)
        }
        
        return session_dict
    
    def _calculate_session_duration(self, session: ConversationSession) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(session.start_time)
            last_activity = datetime.fromisoformat(session.last_activity)
            duration = last_activity - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0