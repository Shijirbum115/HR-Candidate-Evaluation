# Updated src/data/db_connect.py with recency filter capability

import mysql.connector
from mysql.connector import pooling
import time
import numpy as np
from datetime import date, datetime, timedelta
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.model.vector_scoring import generate_search_embedding, extract_from_search_term

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

    def calculate_recency_date(self, months_ago):
        """Calculate the date threshold for recency filter.
        
        Args:
            months_ago (int): Number of months to look back
            
        Returns:
            date: Date threshold
        """
        if not months_ago or months_ago <= 0:
            # Default to 3 months if invalid
            months_ago = 3
            
        today = date.today()
        # Calculate date 'months_ago' months in the past
        days_in_month = 30  # Approximate
        days_ago = months_ago * days_in_month
        return today - timedelta(days=days_ago)

    def fetch_profiles(self, search_term, recency_months=3):
        """Fetch candidate profiles matching a search term with recency filter.
        
        Args:
            search_term (str): Search term to match
            recency_months (int): Only fetch candidates active in the last N months
            
        Returns:
            list: Matching profiles
        """
        try:
            self._check_connection()
            
            # Calculate recency date threshold
            recency_date = self.calculate_recency_date(recency_months)
            logger.info(f"Using recency filter: candidates active since {recency_date}")
            
            # Extract search terms for improved matching
            job_title, industry, experience_years = extract_from_search_term(search_term)
            
            if self.use_postgres:
                # Vector-based search in PostgreSQL with recency filter
                search_embedding = generate_search_embedding(search_term)
                
                if search_embedding is not None:
                    expected_dim = 1536
                    if len(search_embedding) < expected_dim:
                        # Pad the vector by repeating it
                        repeat_count = expected_dim // len(search_embedding)
                        remainder = expected_dim % len(search_embedding)
                        
                        # Create padded vector
                        padded_embedding = np.tile(search_embedding, repeat_count)
                        if remainder > 0:
                            padded_embedding = np.concatenate([padded_embedding, search_embedding[:remainder]])
                        
                        search_embedding = padded_embedding
                    
                    # Convert numpy array to list for PostgreSQL
                    vector_str = '[' + ','.join(str(x) for x in search_embedding) + ']'
                    
                    # More effective query that combines vector search with text search
                    query = """
                    WITH experience_data AS (
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
                            cp.last_login_date,
                            ced.university_rank,
                            ce.position_vector,
                            -- Calculate both vector similarity and text matching scores
                            1 - (ce.position_vector <=> %s::vector) AS vector_score,
                            CASE 
                                WHEN LOWER(ce.position) = LOWER(%s) THEN 1.0  -- Exact match
                                WHEN LOWER(ce.position) LIKE '%%' || LOWER(%s) || '%%' THEN 0.8  -- Contains
                                WHEN LOWER(%s) LIKE '%%' || LOWER(ce.position) || '%%' THEN 0.7  -- Is contained in
                                ELSE 0.0
                            END AS text_score,
                            -- Calculate years of experience
                            CASE 
                                WHEN ce.start_date IS NOT NULL THEN
                                    EXTRACT(YEAR FROM AGE(COALESCE(ce.end_date, CURRENT_DATE), ce.start_date))
                                ELSE 0
                            END AS years_experience
                        FROM candidate_profiles cp
                        JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                        LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                        WHERE (cp.last_login_date IS NULL OR cp.last_login_date >= %s)
                        -- Include both vector matches and text matches
                        AND (
                            ce.position_vector IS NOT NULL
                            OR LOWER(ce.position) LIKE '%%' || LOWER(%s) || '%%'
                            OR LOWER(ce.company_industry) LIKE '%%' || LOWER(%s) || '%%'
                        )
                    )
                    SELECT 
                        *,
                        -- Combined score (max of vector and text score)
                        GREATEST(vector_score, text_score) AS position_similarity_score
                    FROM experience_data
                    ORDER BY 
                        position_similarity_score DESC,
                        years_experience DESC
                    LIMIT 100
                    """
                    # Execute with both vector and text parameters
                    self.cursor.execute(query, (
                        vector_str,    # For vector search
                        job_title,     # For exact text match
                        job_title,     # For contains text match
                        job_title,     # For is contained in match
                        recency_date,  # For recency filter
                        job_title,     # For fallback text search
                        industry if industry else ""  # For industry text search
                    ))
                    
                    # Log the first few rows to verify query effectiveness
                    sample_results = self.cursor.fetchmany(3)
                    if sample_results:
                        logger.debug(f"Sample query results:")
                        for row in sample_results:
                            logger.debug(f"Candidate ID: {row[0]}, Position: {row[4]}, Vector score: {row[14]:.2f}, Text score: {row[15]:.2f}, Combined: {row[17]:.2f}")
                        
                        # Re-execute to get all results
                        self.cursor.execute(query, (
                            vector_str, job_title, job_title, job_title, recency_date, job_title, industry if industry else ""
                        ))
                    
                    logger.debug(f"Executed vector query with {len(search_embedding)}-dimensional vector")
                else:
                    # Fallback to text search if embedding generation fails
                    query = """
                    SELECT 
                        cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                        ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                        cp.last_login_date, ced.university_rank, NULL as position_vector, 0 AS vector_score,
                        CASE 
                            WHEN LOWER(ce.position) = LOWER(%s) THEN 1.0  -- Exact match
                            WHEN LOWER(ce.position) LIKE '%%' || LOWER(%s) || '%%' THEN 0.8  -- Contains
                            WHEN LOWER(%s) LIKE '%%' || LOWER(ce.position) || '%%' THEN 0.7  -- Is contained in
                            ELSE 0.0
                        END AS text_score,
                        0 as years_experience,
                        CASE 
                            WHEN LOWER(ce.position) = LOWER(%s) THEN 1.0  -- Exact match
                            WHEN LOWER(ce.position) LIKE '%%' || LOWER(%s) || '%%' THEN 0.8  -- Contains
                            WHEN LOWER(%s) LIKE '%%' || LOWER(ce.position) || '%%' THEN 0.7  -- Is contained in
                            ELSE 0.0
                        END AS position_similarity_score
                    FROM candidate_profiles cp
                    LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                    LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                    WHERE (ce.position ILIKE '%%' || %s || '%%' OR ce.company_industry ILIKE '%%' || %s || '%%')
                    AND (cp.last_login_date IS NULL OR cp.last_login_date >= %s)
                    ORDER BY position_similarity_score DESC
                    LIMIT 100
                    """
                    self.cursor.execute(query, (
                        job_title, job_title, job_title, 
                        job_title, job_title, job_title,
                        job_title, industry if industry else "", recency_date
                    ))
            else:
                # MySQL text-based search with recency filter
                # First, get candidate IDs matching the search term and recency
                query_ids = """
                SELECT DISTINCT cp.candidate_id
                FROM candidate_profiles cp
                LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                WHERE (ce.position LIKE %s OR ce.company_industry LIKE %s)
                AND (cp.last_login_date IS NULL OR cp.last_login_date >= %s)
                """
                self.cursor.execute(query_ids, (f"%{job_title}%", f"%{industry if industry else ''}%", recency_date))
                candidate_ids = [row[0] for row in self.cursor.fetchall()]
                
                if not candidate_ids:
                    return []

                # Then, fetch all data for those candidates
                placeholders = ','.join(['%s'] * len(candidate_ids))
                query = f"""
                SELECT 
                    cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                    ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                    cp.last_login_date, ced.university_rank, NULL as position_vector, 0 AS vector_score,
                    CASE 
                        WHEN LOWER(ce.position) = LOWER(%s) THEN 1.0
                        WHEN LOWER(ce.position) LIKE CONCAT('%%', LOWER(%s), '%%') THEN 0.8
                        WHEN LOWER(%s) LIKE CONCAT('%%', LOWER(ce.position), '%%') THEN 0.7
                        ELSE 0.0
                    END AS text_score,
                    CASE 
                        WHEN ce.start_date IS NOT NULL THEN
                            TIMESTAMPDIFF(YEAR, ce.start_date, COALESCE(ce.end_date, CURDATE()))
                        ELSE 0
                    END AS years_experience,
                    CASE 
                        WHEN LOWER(ce.position) = LOWER(%s) THEN 1.0
                        WHEN LOWER(ce.position) LIKE CONCAT('%%', LOWER(%s), '%%') THEN 0.8
                        WHEN LOWER(%s) LIKE CONCAT('%%', LOWER(ce.position), '%%') THEN 0.7
                        ELSE 0.0
                    END AS position_similarity_score
                FROM candidate_profiles cp
                LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                WHERE cp.candidate_id IN ({placeholders})
                ORDER BY position_similarity_score DESC, years_experience DESC
                """
                
                params = (job_title, job_title, job_title, job_title, job_title, job_title) + tuple(candidate_ids)
                self.cursor.execute(query, params)
            
            # Fetch results
            results = self.cursor.fetchall()
            # Log a sample result to debug vector format
            if results and len(results) > 0:
                sample = results[0]
                logger.debug(f"Sample result structure: {sample[:13]} + vector data + similarity scores")
                
            logger.debug(f"Fetched {len(results)} profiles for '{search_term}' with {recency_months} month recency filter")
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