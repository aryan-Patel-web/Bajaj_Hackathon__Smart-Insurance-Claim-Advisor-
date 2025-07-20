# utils/logging_config.py

"""
Configures structured logging for the application.
Ensures that logs are consistent, informative, and easy to parse, which is
critical for debugging and monitoring in a production environment.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging():
    """
    Initializes the logging configuration for the entire application.
    """
    # Define the logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    
    # Create a logger instance
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    # Logs messages to the standard output (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # --- File Handler ---
    # Rotates log files daily, keeping 7 days of history.
    # This prevents log files from growing indefinitely.
    # In a real production setup, logs would be shipped to a centralized
    # logging service like ELK Stack, Datadog, or Splunk.
    file_handler = TimedRotatingFileHandler(
        "app.log", when="midnight", interval=1, backupCount=7
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Set the logging level for noisy libraries to WARNING to reduce clutter
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)

    logging.info("Logging configured successfully.")

# Initialize logging when this module is imported
setup_logging()

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for a specific module.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)

if __name__ == '__main__':
    # Example usage
    logger = get_logger(__name__)
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
