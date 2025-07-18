"""
Smart Insurance Claim Advisor - Streamlit Frontend
Main application entry point for HackRx 6.0 submission
"""

import streamlit as st
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import time


# Ensure required directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('src', exist_ok=True)
os.makedirs('utils', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules with better error handling
modules_loaded = {}

# ...existing code...

try:
    from src.ingest import DocumentProcessor
    modules_loaded['ingest'] = True
except ImportError as e:
    logger.error(f"Failed to import ingest module: {e}")
    modules_loaded['ingest'] = False

try:
    from src.parse_query import QueryParser
    modules_loaded['parse_query'] = True
except ImportError as e:
    logger.error(f"Failed to import parse_query module: {e}")
    modules_loaded['parse_query'] = False


# Vector store is not imported because no vector_store instance is defined in src/vector_store.py
modules_loaded['vector_store'] = False

try:
    from src.llm_handler import LLMHandler
    modules_loaded['llm_handler'] = True
except ImportError as e:
    logger.error(f"Failed to import llm_handler module: {e}")
    modules_loaded['llm_handler'] = False

try:
    from src.hybrid_search import HybridSearchEngine
    modules_loaded['hybrid_search'] = True
except ImportError as e:
    logger.error(f"Failed to import hybrid_search module: {e}")
    modules_loaded['hybrid_search'] = False

try:
    from src.conversation import ConversationManager
    modules_loaded['conversation'] = True
except ImportError as e:
    logger.error(f"Failed to import conversation module: {e}")
    modules_loaded['conversation'] = False


# Settings import is handled below
modules_loaded['settings'] = False

try:
    from utils.logging_config import setup_logging
    setup_logging()
    modules_loaded['logging_config'] = True
except ImportError as e:
    logger.error(f"Failed to import logging_config module: {e}")
    modules_loaded['logging_config'] = False

# ...rest of your code...

try:
    from src.parse_query import QueryParser
    modules_loaded['parse_query'] = True
except ImportError as e:
    logger.error(f"Failed to import parse_query module: {e}")
    modules_loaded['parse_query'] = False


# VectorStoreManager is not defined in src/vector_store.py
modules_loaded['vector_store'] = False


# GroqLLMHandler is not defined in src/llm_handler.py
modules_loaded['llm_handler'] = False

try:
    from src.hybrid_search import HybridSearchEngine
    modules_loaded['hybrid_search'] = True
except ImportError as e:
    logger.error(f"Failed to import hybrid_search module: {e}")
    modules_loaded['hybrid_search'] = False

try:
    from src.conversation import ConversationManager
    modules_loaded['conversation'] = True
except ImportError as e:
    logger.error(f"Failed to import conversation module: {e}")
    modules_loaded['conversation'] = False


try:
    from config.settings import Settings
    modules_loaded['settings'] = True
except ImportError as e:
    logger.error(f"Failed to import settings module: {e}")
    modules_loaded['settings'] = False

try:
    from utils.logging_config import setup_logging
    setup_logging()
    modules_loaded['logging_config'] = True
except ImportError as e:
    logger.error(f"Failed to import logging_config module: {e}")
    modules_loaded['logging_config'] = False

# Check if critical modules are loaded
critical_modules = ['settings', 'vector_store', 'llm_handler']
missing_critical = [mod for mod in critical_modules if not modules_loaded.get(mod, False)]

if missing_critical:
    st.error(f"Critical modules missing: {', '.join(missing_critical)}")
    st.error("Please ensure all required files are in the correct directories.")
    st.info("Run the setup script first to create required module files.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Insurance Claim Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .claim-result {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .approved {
        border-left-color: #28a745;
        background: #d4edda;
    }
    
    .rejected {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    
    .module-status {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border: 1px solid #dee2e6;
    }
    
    .module-loaded {
        border-left: 4px solid #28a745;
    }
    
    .module-failed {
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class SmartInsuranceClaimAdvisor:
    """Main application class for the Smart Insurance Claim Advisor"""
    
    def __init__(self):
        """Initialize the application"""
        self.modules_loaded = modules_loaded
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.documents_ingested = False
            st.session_state.conversation_history = []
            st.session_state.ingestion_status = "Not started"
            st.session_state.total_documents = 0
            st.session_state.total_chunks = 0
            st.session_state.conversation_id = None
            st.session_state.demo_mode = True  # Enable demo mode if modules are missing
    
    def initialize_components(self):
        """Initialize application components"""
        try:
            # Initialize settings first
            if modules_loaded.get('settings', False):
                self.settings = Settings()
                try:
                    self.settings.validate_config()
                except ValueError as e:
                    st.warning(f"Configuration warning: {e}")
                    st.info("Running in demo mode with mock data.")
                    st.session_state.demo_mode = True
            else:
                st.session_state.demo_mode = True
            
            # Initialize other components if available

            # VectorStoreManager is not defined in your codebase, so skip initialization
            
            if modules_loaded.get('parse_query', False):
                self.query_parser = QueryParser()
            
            if modules_loaded.get('conversation', False):
                self.conversation_manager = ConversationManager()
            
            if modules_loaded.get('llm_handler', False):
                self.llm_handler = LLMHandler()
            

            # DocumentIngestionPipeline is not defined in your codebase, so skip initialization
            
            # Initialize hybrid search (will be set after vector store is ready)
            self.hybrid_search = None
            
            st.session_state.initialized = True
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            st.error(f"Failed to initialize application: {str(e)}")
            st.info("Some features may not work properly. Running in limited mode.")
            st.session_state.demo_mode = True
    
    def render_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏥 Smart Insurance Claim Advisor</h1>
            <p>HackRx 6.0 - Intelligent Document Processing & Claim Evaluation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo mode warning if applicable
        if st.session_state.get('demo_mode', False):
            st.warning("⚠️ Running in Demo Mode - Some features may use mock data")
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        st.sidebar.title("🔧 Control Panel")
        
        # Module Status
        st.sidebar.subheader("📦 Module Status")
        for module_name, loaded in self.modules_loaded.items():
            status_class = "module-loaded" if loaded else "module-failed"
            status_icon = "✅" if loaded else "❌"
            st.sidebar.markdown(f"""
            <div class="module-status {status_class}">
                {status_icon} {module_name}
            </div>
            """, unsafe_allow_html=True)
        
        # System Status
        st.sidebar.subheader("📊 System Status")
        
        status_color = "🟢" if st.session_state.initialized else "🔴"
        st.sidebar.markdown(f"{status_color} **System**: {'Ready' if st.session_state.initialized else 'Initializing'}")
        
        mode_color = "🟡" if st.session_state.get('demo_mode', False) else "🟢"
        mode_text = "Demo Mode" if st.session_state.get('demo_mode', False) else "Full Mode"
        st.sidebar.markdown(f"{mode_color} **Mode**: {mode_text}")
        
        ingestion_color = "🟢" if st.session_state.documents_ingested else "🟡"
        st.sidebar.markdown(f"{ingestion_color} **Documents**: {st.session_state.ingestion_status}")
        
        if st.session_state.total_documents > 0:
            st.sidebar.metric("Documents Processed", st.session_state.total_documents)
            st.sidebar.metric("Text Chunks", st.session_state.total_chunks)
        
        # Configuration
        st.sidebar.subheader("⚙️ Configuration")
        
        # Search settings
        search_threshold = st.sidebar.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.1,
            help="Minimum similarity score for search results"
        )
        
        max_results = st.sidebar.selectbox(
            "Max Results", 
            options=[5, 10, 15, 20], 
            index=1,
            help="Maximum number of search results to return"
        )
        
        # Store settings in session state
        st.session_state.search_threshold = search_threshold
        st.session_state.max_results = max_results
        
        # Debug Mode
        st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
        
        # Clear conversation
        if st.sidebar.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.conversation_id = None
            st.success("Conversation cleared!")
    
    def render_demo_data(self):
        """Render demo data section"""
        st.subheader("🎯 Demo Mode")
        
        st.info("""
        **Demo Mode Active**: The application is running with mock data and limited functionality.
        
        **To enable full functionality:**
        1. Install all required dependencies
        2. Create the required module files
        3. Set up your environment variables (.env file)
        4. Restart the application
        """)
        
        if st.button("🚀 Run Demo Query"):
            # Add demo conversation to history
            demo_response = {
                "decision": "Approved",
                "amount": "₹75,000",
                "justification": [
                    {
                        "clause_id": "DEMO_001",
                        "text": "Policy covers knee surgery for members aged 40-50"
                    },
                    {
                        "clause_id": "DEMO_002", 
                        "text": "3-month waiting period satisfied"
                    }
                ]
            }
            
            st.session_state.conversation_history.append({
                'query': "46-year-old male, knee surgery in Pune, 3-month-old policy",
                'response': demo_response,
                'timestamp': datetime.now().isoformat()
            })
            
            st.success("Demo query processed!")
            st.rerun()
    
    def render_troubleshooting(self):
        """Render troubleshooting information"""
        st.subheader("🔧 Troubleshooting")
        
        st.markdown("""
        ### Common Issues:
        
        **1. Module Import Errors**
        - Ensure all required files exist in src/, utils/, and config/ directories
        - Check that __init__.py files are present in each directory
        
        **2. Missing Dependencies**
        - Install all required packages using pip
        - Check Python version compatibility
        
        **3. Environment Variables**
        - Create .env file with required API keys
        - Ensure environment variables are correctly set
        
        **4. File Permissions**
        - Check write permissions for logs directory
        - Ensure temp directory access for file uploads
        """)
        
        if st.button("🔍 Run System Check"):
            st.markdown("### System Check Results:")
            
            # Check Python version
            st.write(f"**Python Version**: {sys.version}")
            
            # Check critical directories
            dirs_to_check = ['src', 'utils', 'config', 'logs']
            for dir_name in dirs_to_check:
                exists = os.path.exists(dir_name)
                st.write(f"**{dir_name}/ directory**: {'✅ Exists' if exists else '❌ Missing'}")
            
            # Check environment variables
            env_vars = ['GROQ_API_KEY', 'ASTRA_DB_TOKEN', 'ASTRA_DB_ENDPOINT']
            for var in env_vars:
                value = os.getenv(var)
                st.write(f"**{var}**: {'✅ Set' if value else '❌ Missing'}")
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        if not st.session_state.initialized:
            st.error("⚠️ Application not initialized properly.")
            self.render_troubleshooting()
            return
        
        # Create tabs for different sections
        if st.session_state.get('demo_mode', False):
            tab1, tab2, tab3 = st.tabs(["🎯 Demo", "📊 Analytics", "🔧 Troubleshooting"])
            
            with tab1:
                self.render_demo_data()
                
                # Show conversation history if any
                if st.session_state.conversation_history:
                    st.subheader("💬 Conversation History")
                    for i, exchange in enumerate(st.session_state.conversation_history):
                        self.render_chat_message(exchange, i)
            
            with tab2:
                self.render_analytics()
            
            with tab3:
                self.render_troubleshooting()
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Documents", "💬 Chat", "📊 Analytics", "❓ Help"])
            
            with tab1:
                self.render_document_upload()
            
            with tab2:
                self.render_chat_interface()
            
            with tab3:
                self.render_analytics()
            
            with tab4:
                self.render_help_section()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8em;">
            🏆 HackRx 6.0 - Smart Insurance Claim Advisor<br>
            Built with ❤️ using Streamlit, LangChain, Astra DB, and Groq
        </div>
        """, unsafe_allow_html=True)
    
    def render_chat_message(self, exchange: Dict[str, Any], index: int):
        """Render a single chat message exchange"""
        # User message
        st.markdown(f"""
        <div class="chat-message">
            <strong>👤 You:</strong> {exchange['query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot response
        response = exchange['response']
        bot_class = "bot-message approved" if response.get('decision') == 'Approved' else "bot-message rejected"
        
        st.markdown(f"""
        <div class="chat-message {bot_class}">
            <strong>🤖 Insurance Advisor:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Render structured response
        self.render_claim_result(response, index)
    
    def render_claim_result(self, response: Dict[str, Any], index: int):
        """Render the claim evaluation result"""
        decision = response.get('decision', 'Unknown')
        amount = response.get('amount', 'N/A')
        justification = response.get('justification', [])
        
        # Decision and amount
        result_class = "approved" if decision == 'Approved' else "rejected"
        
        st.markdown(f"""
        <div class="claim-result {result_class}">
            <h4>📋 Claim Decision: {decision}</h4>
            <p><strong>💰 Amount:</strong> {amount}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Justification details
        if justification:
            st.markdown("**📚 Justification Details:**")
            
            for i, item in enumerate(justification):
                clause_id = item.get('clause_id', f'clause_{i}')
                text = item.get('text', 'No text available')
                
                st.markdown(f"""
                <div class="justification-item">
                    <strong>📄 Clause ID:</strong> {clause_id}<br>
                    <strong>📝 Text:</strong> {text}
                </div>
                """, unsafe_allow_html=True)
        
        # Raw JSON (for debug mode)
        if st.session_state.get('debug_mode', False):
            with st.expander(f"🔍 Raw JSON Response - Query {index + 1}"):
                st.json(response)
    
    def render_analytics(self):
        """Render analytics and insights"""
        if not st.session_state.conversation_history:
            st.info("No conversation data available yet.")
            return
        
        st.subheader("📊 Analytics & Insights")
        
        # Conversation statistics
        total_queries = len(st.session_state.conversation_history)
        approved_claims = sum(1 for ex in st.session_state.conversation_history 
                             if ex['response'].get('decision') == 'Approved')
        rejected_claims = total_queries - approved_claims
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", total_queries)
        
        with col2:
            st.metric("Approved Claims", approved_claims)
        
        with col3:
            st.metric("Rejected Claims", rejected_claims)
        
        # Approval rate chart
        if total_queries > 0:
            approval_rate = (approved_claims / total_queries) * 100
            
            chart_data = pd.DataFrame({
                'Decision': ['Approved', 'Rejected'],
                'Count': [approved_claims, rejected_claims]
            })
            
            st.bar_chart(chart_data.set_index('Decision'))
            
            st.info(f"📈 Current approval rate: {approval_rate:.1f}%")
    
    # Placeholder methods for full functionality
    def render_document_upload(self):
        """Render document upload section"""
        st.subheader("📁 Document Ingestion")
        st.info("Document upload functionality requires full module setup.")
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Insurance Claim Chat")
        st.info("Chat interface requires full module setup.")
    
    def render_help_section(self):
        """Render help and documentation"""
        st.subheader("❓ Help & Documentation")
        st.info("Help section for full functionality.")

# Main execution
if __name__ == "__main__":
    try:
        # Initialize and run the application
        app = SmartInsuranceClaimAdvisor()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        st.info("Please check your setup and try again.")
        logger.error(f"Application startup error: {str(e)}", exc_info=True)