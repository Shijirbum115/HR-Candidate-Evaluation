import logging
import os

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Ensure DEBUG level

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # File handler with UTF-8
    if not os.path.exists("logs"):
        os.makedirs("logs")
    file_handler = logging.FileHandler("logs/app.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Configure the root logger as well to catch all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Only add handlers if they don't exist
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                     for h in root_logger.handlers)
    if not has_console:
        root_console = logging.StreamHandler()
        root_console.setLevel(logging.DEBUG)
        root_console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(root_console)

    return logger