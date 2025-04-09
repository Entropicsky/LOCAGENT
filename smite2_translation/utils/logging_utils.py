"""Logging configuration utility."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Default log directory (can be overridden)
LOG_DIR = "logs"

def setup_logging(log_dir: str = LOG_DIR, log_level: int = logging.INFO):
    """Configures logging for the application.

    Sets up logging to both console (INFO level) and a rotating file 
    (DEBUG level if verbose, otherwise INFO).

    Args:
        log_dir: The directory to store log files.
        log_level: The logging level for the console handler (e.g., logging.INFO, logging.DEBUG).
    """
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
    except OSError as e:
        print(f"Error creating log directory '{log_dir}': {e}", file=sys.stderr)
        # Continue without file logging if directory creation fails
        log_dir = None

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger() # Get root logger
    # Set root logger level to the lowest level we want to handle (e.g., DEBUG)
    # Handlers will then filter based on their own levels.
    root_logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers if called multiple times (e.g., in tests)
    if root_logger.hasHandlers():
        # Clear existing handlers - careful if other libraries configure logging
        # A more robust approach might check handler names or types
        # For this project, clearing is likely acceptable.
        print("Clearing existing logging handlers.")
        root_logger.handlers.clear()

    # Console Handler (INFO level by default, controlled by log_level argument)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Rotating, DEBUG level)
    if log_dir:
        log_filepath = os.path.join(log_dir, "translation_app.log")
        # Use RotatingFileHandler to limit log file size
        file_handler = RotatingFileHandler(
            log_filepath, 
            maxBytes=5*1024*1024, # 5 MB per file
            backupCount=3,        # Keep 3 backup files
            encoding='utf-8'
        )
        # File handler always logs at DEBUG level to capture everything
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        print(f"Logging configured. Console level: {logging.getLevelName(log_level)}, File level: DEBUG, File path: {log_filepath}")
    else:
        print(f"Logging configured. Console level: {logging.getLevelName(log_level)}. File logging disabled.")

# Example usage:
# if __name__ == '__main__':
#     setup_logging(log_level=logging.DEBUG)
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     logging.error("This is an error message.")
#     logging.critical("This is a critical message.") 