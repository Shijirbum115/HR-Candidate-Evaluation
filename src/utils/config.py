# In src/utils/config.py

from dotenv import load_dotenv
import os

def load_config():
    load_dotenv()  # Load .env file
    
    # Get all required configuration values with defaults
    config = {
        # Original MySQL config
        "CLOUD_DB_HOST": os.getenv("CLOUD_DB_HOST"),
        "CLOUD_DB_USER": os.getenv("CLOUD_DB_USER"),
        "CLOUD_DB_PASSWORD": os.getenv("CLOUD_DB_PASSWORD"),
        "CLOUD_DB_NAME": os.getenv("CLOUD_DB_NAME"),
        "LOCAL_DB_HOST": os.getenv("LOCAL_DB_HOST"),
        "LOCAL_DB_USER": os.getenv("LOCAL_DB_USER"),
        "LOCAL_DB_PASSWORD": os.getenv("LOCAL_DB_PASSWORD"),
        "LOCAL_DB_NAME": os.getenv("LOCAL_DB_NAME"),
        
        # PostgreSQL config (new)
        "PG_DB_HOST": os.getenv("PG_DB_HOST", "localhost"),
        "PG_DB_USER": os.getenv("PG_DB_USER", "postgres"),
        "PG_DB_PASSWORD": os.getenv("PG_DB_PASSWORD", ""),
        "PG_DB_NAME": os.getenv("PG_DB_NAME", "hr_db"),
        
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    return config