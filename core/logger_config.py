import logging
import os
import sys

def setup_logging():
    """
    Configures basic logging for the application.
    Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
    Logs are sent to stdout.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Use a basic formatter for now, can be more detailed
    # Example: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Get the root logger
    # logger = logging.getLogger()
    # Using a specific logger for the app might be cleaner if other libs use root
    logger = logging.getLogger("rag_app") # Specific logger for our app
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs if this function is called multiple times
    # Though for st.cache_resource or module-level setup, this might not be an issue.
    # For a simple script, ensuring it's called once is key.
    if not logger.handlers: # Add handler only if no handlers are configured.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Logging initialized with level {log_level_str}")

# To make it easy to get this logger in other modules
def get_logger(name):
    return logging.getLogger(f"rag_app.{name}")

# Example of how a module would get its logger:
# from .logger_config import get_logger
# logger = get_logger(__name__)
