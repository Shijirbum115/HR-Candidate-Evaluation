import mysql.connector
from mysql.connector import pooling
import time
import numpy as np
from datetime import date
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.model.vector_scoring import generate_search_embedding

logger = setup_logging()

# Global connection pools
_mysql_pool = None
_pg_pool = None

def get_mysql_pool():
    """Get or create a MySQL connection pool."""
    global _mysql_pool
    if _mysql_pool is None:
        config = load_config()
        _mysql_pool = pooling.MySQLConnectionPool(
            pool_name="mysql_pool",
            pool_size=5,
            host=config["LOCAL_DB_HOST"],
            user=config["LOCAL_DB_USER"],
            password=config["LOCAL_DB_PASSWORD"],
            database=config["LOCAL_DB_NAME"],
            charset='utf8mb4',
            use_pure=True
        )
        logger.info("MySQL database connection pool established")
    return _mysql_pool

def get_pg_pool():
    """Get or create a PostgreSQL connection pool if PostgreSQL is configured."""
    global _pg_pool
    if _pg_pool is None:
        config = load_config()
        # Check if PostgreSQL config is available
        if config.get("PG_DB_HOST"):
            try:
                # Import libraries only if needed
                from psycopg2.pool import ThreadedConnectionPool
                import pgvector.psycopg2
                
                _pg_pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    host=config["PG_DB_HOST"],
                    user=config["PG_DB_USER"],
                    password=config["PG_DB_PASSWORD"],
                    database=config["PG_DB_NAME"]
                )
                logger.info("PostgreSQL database connection pool established")
            except ImportError:
                logger.warning("PostgreSQL or pgvector not available. Using MySQL only.")
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL pool: {str(e)}")
    return _pg_pool

class LocalDatabase:
    """Class for interacting with the database with connection pooling."""
    
    def __init__(self):
        """Initialize connection from pool with database type detection."""
        self.pg_pool = get_pg_pool()
        self.mysql_pool = get_mysql_pool()
        
        # Determine which database to use
        self.use_postgres = self.pg_pool is not None
        
        # Initialize connections
        self.conn = None
        self.cursor = None
        self._get_connection()
        
        if self.use_postgres:
            logger.info("Using PostgreSQL with vector capabilities")
        else:
            logger.info("Using MySQL database")

    def _get_connection(self):
        """Get a connection from the appropriate pool with retry logic."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if self.use_postgres:
                    self.conn = self.pg_pool.getconn()
                    self.cursor = self.conn.cursor()
                else:
                    self.conn = self.mysql_pool.get_connection()
                    self.cursor = self.conn.cursor(buffered=True)
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    logger.error(f"Failed to get database connection after {max_attempts} attempts")
                    raise

    def _check_connection(self):
        """Check if connection is alive, reconnect if needed."""
        try:
            # Ping the connection
            if self.use_postgres:
                self.cursor.execute("SELECT 1")
            else:
                self.conn.ping(reconnect=False, attempts=1, delay=0)
        except:
            logger.warning("Connection lost, reconnecting...")
            self.close()
            self._get_connection()

    def fetch_profiles(self, search_term):
        """Fetch candidate profiles matching a search term with vector similarity if possible."""
        try:
            self._check_connection()
            
            if self.use_postgres:
                # Vector-based search in PostgreSQL
                # Generate embedding for search term
                search_embedding = generate_search_embedding(search_term)
                
                if search_embedding is not None:
                    # Convert numpy array to list for PostgreSQL
                    vector_str = ','.join(str(x) for x in search_embedding)
                    
                    # Use pgvector's cosine distance operator (<=>)
                    query = """
                    WITH ranked_experiences AS (
                        SELECT 
                            cp.candidate_id, 
                            cp.birthdate,
                            ce.company,
                            ce.company_industry,
                            ce.position,
                            ce.start_date,
                            ce.end_date,
                            ced.school,
                            ced.degree,
                            cp.firstname,
                            cp.lastname,
                            ced.university_rank,
                            1 - (ce.position_vector <=> %s::vector) AS position_similarity_score
                        FROM candidate_profiles cp
                        JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                        LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                        WHERE ce.position_vector IS NOT NULL
                        ORDER BY position_similarity_score DESC
                        LIMIT 50
                    )
                    SELECT * FROM ranked_experiences
                    """
                    self.cursor.execute(query, (vector_str,))
                else:
                    # Fallback to text search if embedding generation fails
                    query = """
                    SELECT 
                        cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                        ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                        ced.university_rank, 0 AS position_similarity_score
                    FROM candidate_profiles cp
                    LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                    LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                    WHERE ce.position ILIKE %s OR ce.company_industry ILIKE %s
                    LIMIT 50
                    """
                    self.cursor.execute(query, (f"%{search_term}%", f"%{search_term}%"))
            else:
                # MySQL text-based search
                # First, get candidate IDs matching the search term
                query_ids = """
                SELECT DISTINCT cp.candidate_id
                FROM candidate_profiles cp
                LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                WHERE ce.position LIKE %s OR ce.company_industry LIKE %s
                """
                self.cursor.execute(query_ids, (f"%{search_term}%", f"%{search_term}%"))
                candidate_ids = [row[0] for row in self.cursor.fetchall()]
                
                if not candidate_ids:
                    return []

                # Then, fetch all data for those candidates
                # Include a fixed value for position_similarity_score since we can't calculate it
                placeholders = ','.join(['%s'] * len(candidate_ids))
                query = f"""
                SELECT 
                    cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                    ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                    ced.university_rank, 0 AS position_similarity_score
                FROM candidate_profiles cp
                LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                WHERE cp.candidate_id IN ({placeholders})
                """
                
                self.cursor.execute(query, tuple(candidate_ids))
            
            # Fetch results
            results = self.cursor.fetchall()
            logger.debug(f"Fetched {len(results)} profiles for '{search_term}'")
            return results
        except Exception as e:
            logger.error(f"Error fetching profiles: {str(e)}")
            # Try to reconnect and retry
            self._get_connection()
            # We'd need to repeat the entire operation here
            # For brevity, just raising the error after reconnection attempt
            raise

    def close(self):
        """Close cursor and return connection to pool."""
        if self.cursor:
            self.cursor.close()
        
        if self.conn:
            if self.use_postgres:
                self.pg_pool.putconn(self.conn)
            else:
                self.conn.close()
        
        self.cursor = None
        self.conn = None
        logger.info("Database connection closed")