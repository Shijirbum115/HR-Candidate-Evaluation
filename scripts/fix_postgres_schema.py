import sys
import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def ensure_logs_directory():
    """Make sure the logs directory exists."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
        logger.info("Created logs directory")

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    return {
        "PG_HOST": os.getenv("PG_DB_HOST", "localhost"),
        "PG_PORT": os.getenv("PG_DB_PORT", "5432"),
        "PG_USER": os.getenv("PG_DB_USER", "postgres"),
        "PG_PASSWORD": os.getenv("PG_DB_PASSWORD", ""),
        "PG_DB": os.getenv("PG_DB_NAME", "hr_db")
    }

def get_postgres_connection():
    """Create and return a PostgreSQL connection."""
    config = load_config()
    try:
        conn = psycopg2.connect(
            host=config["PG_HOST"],
            port=config["PG_PORT"],
            user=config["PG_USER"],
            password=config["PG_PASSWORD"],
            database=config["PG_DB"]
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        logger.info("Connected to PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise

def create_tables():
    """Create the PostgreSQL tables if they don't exist."""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Create the tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_profiles (
            candidate_id INT PRIMARY KEY,
            birthdate DATE,
            firstname TEXT,
            lastname TEXT
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_experiences (
            exp_id SERIAL PRIMARY KEY,
            candidate_id INT,
            company TEXT,
            company_industry TEXT,
            position TEXT,
            start_date DATE,
            end_date DATE,
            FOREIGN KEY (candidate_id) REFERENCES candidate_profiles(candidate_id)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_education (
            edu_id SERIAL PRIMARY KEY,
            candidate_id INT,
            school TEXT,
            university_rank INT,
            degree TEXT,
            start_year INT,
            end_year INT,
            FOREIGN KEY (candidate_id) REFERENCES candidate_profiles(candidate_id),
            UNIQUE (candidate_id, school, start_year, end_year)
        );
        """)
        
        logger.info("Tables created successfully (if they didn't exist)")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}")
        return False

def add_vector_columns():
    """Add vector columns to the tables for pgvector."""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone() is None:
            logger.info("Installing pgvector extension")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Check if position_vector column already exists in candidate_experiences
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'candidate_experiences' 
            AND column_name = 'position_vector'
        """)
        if cursor.fetchone() is None:
            logger.info("Adding position_vector column to candidate_experiences")
            cursor.execute("ALTER TABLE candidate_experiences ADD COLUMN position_vector vector(1536);")
            
        # Check if degree_vector column already exists in candidate_education
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'candidate_education' 
            AND column_name = 'degree_vector'
        """)
        if cursor.fetchone() is None:
            logger.info("Adding degree_vector column to candidate_education")
            cursor.execute("ALTER TABLE candidate_education ADD COLUMN degree_vector vector(1536);")
        
        # Create indices for vector similarity search
        logger.info("Creating vector indices")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experience_vector ON candidate_experiences USING ivfflat (position_vector vector_cosine_ops);")
        except Exception as e:
            logger.warning(f"Could not create vector index on experiences: {str(e)}")
            
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_education_vector ON candidate_education USING ivfflat (degree_vector vector_cosine_ops);")
        except Exception as e:
            logger.warning(f"Could not create vector index on education: {str(e)}")
        
        logger.info("Vector columns and indices added successfully")
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Failed to add vector columns: {str(e)}")
        return False

def alter_column_lengths():
    """Alter PostgreSQL columns to handle longer strings."""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Get the actual max lengths from MySQL data
        lengths = check_data_lengths()
        
        # Dynamically determine which columns need altering
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
    """Check MySQL data to identify potentially long fields."""
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
    # Ensure logs directory exists
    ensure_logs_directory()
    
    logger.info("Starting schema setup script")
    
    # Create tables if they don't exist
    if create_tables():
        logger.info("Tables created or already exist")
    else:
        logger.error("Failed to create tables")
        sys.exit(1)
    
    # Add vector columns for similarity search
    if add_vector_columns():
        logger.info("Vector columns added or already exist")
    else:
        logger.error("Failed to add vector columns")
        sys.exit(1)
    
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
        logger.info("Schema setup completed successfully. You can now run the migration script.")
    else:
        logger.error("Failed to complete schema setup. Please check the database configuration and permissions.")