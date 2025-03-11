from dotenv import load_dotenv
import os

def load_config():
    load_dotenv()  # Loads .env from the current working directory
    return {
        "CLOUD_DB_HOST": os.getenv("CLOUD_DB_HOST"),
        "CLOUD_DB_USER": os.getenv("CLOUD_DB_USER"),
        "CLOUD_DB_PASSWORD": os.getenv("CLOUD_DB_PASSWORD"),
        "CLOUD_DB_NAME": os.getenv("CLOUD_DB_NAME"),
        "LOCAL_DB_HOST": os.getenv("LOCAL_DB_HOST"),
        "LOCAL_DB_USER": os.getenv("LOCAL_DB_USER"),
        "LOCAL_DB_PASSWORD": os.getenv("LOCAL_DB_PASSWORD"),
        "LOCAL_DB_NAME": os.getenv("LOCAL_DB_NAME"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }