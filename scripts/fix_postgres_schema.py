import sys
import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/migration_fix.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("migration_fix")

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    return {
        "PG_HOST": os.getenv("PG_HOST"),
        "PG_PORT": os.getenv("PG_PORT"),
        "PG_USER": os.getenv("PG_USER"),
        "PG_PASSWORD": os.getenv("PG_PASSWORD"),
        "PG_DB": os.getenv("PG_DB")
    }

def alter_column_lengths():
    """Alter PostgreSQL columns to handle longer strings"""
    config = load_config()
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=config["PG_HOST"],
            port=config["PG_PORT"],
            user=config["PG_USER"],
            password=config["PG_PASSWORD"],
            database=config["PG_DB"]
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        logger.info("Connected to PostgreSQL database")
        
        # Modify the columns that might contain long strings
        # First, let's check which table and column is causing the issue
        tables_to_check = [
            ("candidate_education", ["school", "degree"]),
            ("candidate_experiences", ["company", "company_industry", "position"]),
            ("candidate_profiles", ["firstname", "lastname"])
        ]
        
        for table, columns in tables_to_check:
            for column in columns:
                logger.info(f"Altering {table}.{column} to TEXT type")
                alter_query = f"ALTER TABLE {table} ALTER COLUMN {column} TYPE TEXT;"
                cursor.execute(alter_query)
        
        logger.info("Successfully modified column types")
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Failed to alter column lengths: {str(e)}")
        return False

def check_data_lengths():
    """Check MySQL data to identify potentially long fields"""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    try:
        from src.data.db_connect import CloudDatabase
        
        cloud_db = CloudDatabase()
        logger.info("Connected to MySQL database")
        
        # Check education data
        query = """
        SELECT MAX(LENGTH(school)), MAX(LENGTH(pro)) 
        FROM career_cv_edus
        """
        cloud_db.cursor.execute(query)
        max_school_len, max_degree_len = cloud_db.cursor.fetchone()
        
        # Check experience data
        query = """
        SELECT MAX(LENGTH(company)), MAX(LENGTH(position)) 
        FROM career_cv_exprs
        """
        cloud_db.cursor.execute(query)
        max_company_len, max_position_len = cloud_db.cursor.fetchone()
        
        # Check company industry data
        query = """
        SELECT MAX(LENGTH(title)) 
        FROM career_site_data 
        WHERE grp_id = 3
        """
        cloud_db.cursor.execute(query)
        max_industry_len = cloud_db.cursor.fetchone()[0]
        
        # Check name data
        query = """
        SELECT MAX(LENGTH(firstname)), MAX(LENGTH(lastname)) 
        FROM career_user_data
        """
        cloud_db.cursor.execute(query)
        max_fname_len, max_lname_len = cloud_db.cursor.fetchone()
        
        cloud_db.close()
        
        logger.info("Maximum string lengths in MySQL data:")
        logger.info(f"School: {max_school_len} characters")
        logger.info(f"Degree: {max_degree_len} characters")
        logger.info(f"Company: {max_company_len} characters")
        logger.info(f"Position: {max_position_len} characters")
        logger.info(f"Industry: {max_industry_len} characters")
        logger.info(f"First name: {max_fname_len} characters")
        logger.info(f"Last name: {max_lname_len} characters")
        
        return {
            "school": max_school_len,
            "degree": max_degree_len,
            "company": max_company_len,
            "position": max_position_len,
            "industry": max_industry_len,
            "firstname": max_fname_len,
            "lastname": max_lname_len
        }
    except Exception as e:
        logger.error(f"Failed to check data lengths: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting migration fix script")
    
    # Check data lengths to identify problematic columns
    lengths = check_data_lengths()
    if lengths:
        long_fields = [field for field, length in lengths.items() if length and length > 255]
        if long_fields:
            logger.warning(f"Found fields exceeding 255 characters: {', '.join(long_fields)}")
        else:
            logger.info("No fields exceed 255 characters in length")
    
    # Alter column types regardless, to prevent future issues
    success = alter_column_lengths()
    
    if success:
        logger.info("Column modifications completed successfully. You can now re-run the migration script.")
    else:
        logger.error("Failed to modify columns. Please check the database configuration and permissions.")