import logging
import logging.handlers
import os
import sys
from datetime import datetime

# Fix import path if needed
try:
    from config.settings import settings
except ImportError:
    settings = None  # Fallback if settings import fails

class AuditLogger:
    def __init__(self, name: str = "insurance_claim_advisor"):
        self.logger = logging.getLogger(name)
        self.setup_logging()

    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        log_level = getattr(logging, getattr(settings, "log_level", "INFO").upper(), logging.INFO) if settings else logging.INFO
        self.logger.setLevel(log_level)

        if self.logger.handlers:
            return

        detailed_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        console_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_fmt)
        self.logger.addHandler(ch)

        fh = logging.handlers.RotatingFileHandler('logs/app.log', maxBytes=10*1024*1024, backupCount=5)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(detailed_fmt)
        self.logger.addHandler(fh)

        ah = logging.handlers.RotatingFileHandler('logs/audit.log', maxBytes=10*1024*1024, backupCount=10)
        ah.setLevel(logging.INFO)
        ah.setFormatter(detailed_fmt)
        audit_logger = logging.getLogger("audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(ah)

    def log_document_ingestion(self, filename: str, file_size: int, chunks_created: int):
        logging.getLogger("audit").info(
            f"DOCUMENT_INGESTION - File: {filename}, Size: {file_size}, Chunks: {chunks_created}, Timestamp: {datetime.now().isoformat()}"
        )

    def log_query_processing(self, query: str, user_id: str = None):
        user_id_str = user_id if user_id is not None else ""
        logging.getLogger("audit").info(
            f"QUERY_PROCESSING - Query: {query[:100]}..., User: {user_id_str}, Timestamp: {datetime.now().isoformat()}"
        )

    def log_claim_decision(self, query: str, decision: str, amount: str, justification_count: int, user_id: str = None):
        user_id_str = user_id if user_id is not None else ""
        logging.getLogger("audit").info(
            f"CLAIM_DECISION - Query: {query[:100]}..., Decision: {decision}, Amount: {amount}, Justifications: {justification_count}, User: {user_id_str}, Timestamp: {datetime.now().isoformat()}"
        )

    def log_error(self, error_type: str, error_message: str, context: str = None):
        context_str = context if context is not None else ""
        logging.getLogger("audit").error(
            f"ERROR - Type: {error_type}, Message: {error_message}, Context: {context_str}, Timestamp: {datetime.now().isoformat()}"
        )

    def info(self, msg: str): self.logger.info(msg)
    def debug(self, msg: str): self.logger.debug(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def critical(self, msg: str): self.logger.critical(msg)

def setup_logging():
    pass  # Already handled by AuditLogger on import

logger = AuditLogger()