import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
import pgvector.psycopg2
from src.utils.config import load_config
from src.utils.logger import setup_logging
import threading
import time

logger = setup_logging()

# Global connection pool singleton
_pool = None
_pool_lock = threading.Lock()

def get_pg_pool():
    """Get or create a PostgreSQL connection pool.
    
    Returns:
        ThreadedConnectionPool: A connection pool for PostgreSQL
    """
    global _pool
    
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # Double-check under lock
                config = load_config()
                _pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,  # Adjust based on your needs
                    host=config["PG_DB_HOST"],
                    user=config["PG_DB_USER"],
                    password=config["PG_DB_PASSWORD"],
                    database=config["PG_DB_NAME"]
                )
                logger.info("PostgreSQL connection pool established")
    
    return _pool

class LocalPostgresDatabase:
    """PostgreSQL database connector with connection pooling and error handling."""
    
    def __init__(self):
        """Initialize a PostgreSQL database connection from the pool."""
        self.pool = get_pg_pool()
        self.conn = None
        self.cursor = None
        self._get_connection()
    
    def _get_connection(self):
        """Get a connection from the pool with retry logic."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.conn = self.pool.getconn()
                self.conn.set_session(autocommit=False)  # Use explicit transactions
                self.cursor = self.conn.cursor()
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    logger.error(f"Failed to get database connection after {max_attempts} attempts")
                    raise

    def save_candidates(self, data):
        """Save candidate profile data to PostgreSQL.
        
        Args:
            data (list): List of tuples with candidate data
        """
        try:
            query = """
            INSERT INTO candidate_profiles 
            (candidate_id, birthdate, firstname, lastname)
            VALUES %s
            ON CONFLICT (candidate_id) DO UPDATE SET
                birthdate = EXCLUDED.birthdate,
                firstname = EXCLUDED.firstname,
                lastname = EXCLUDED.lastname
            """
            execute_values(self.cursor, query, data)
            self.conn.commit()
            logger.info(f"Saved {len(data)} candidates")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save candidates: {str(e)}")
            raise

    def save_experiences(self, data):
        """Save work experience data to PostgreSQL.
        
        Args:
            data (list): List of tuples with experience data
        """
        try:
            # Check if position_vector column exists in the table
            self.cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_experiences' 
                AND column_name = 'position_vector'
            """)
            has_vector_column = self.cursor.fetchone() is not None
            
            if has_vector_column:
                # Process with vector embeddings
                query = """
                INSERT INTO candidate_experiences 
                (candidate_id, company, company_industry, position, start_date, end_date, position_vector)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                # Preprocess data to add embedding vectors for positions
                processed_data = []
                for row in data:
                    # Create a copy of the row as a list to allow modification
                    row_list = list(row)
                    # Generate embedding for position (assuming position is at index 3)
                    position_embedding = self.generate_embedding(row[3]) if row[3] else None
                    # Append the embedding to the row
                    row_list.append(position_embedding)
                    processed_data.append(tuple(row_list))
                    
                execute_values(self.cursor, query, processed_data)
            else:
                # Process without vector embeddings
                query = """
                INSERT INTO candidate_experiences 
                (candidate_id, company, company_industry, position, start_date, end_date)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                execute_values(self.cursor, query, data)
                
            self.conn.commit()
            logger.info(f"Saved {len(data)} experiences")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save experiences: {str(e)}")
            raise

    def save_education(self, data):
        """Save education data to PostgreSQL.
        
        Args:
            data (list): List of tuples with education data
        """
        try:
            # Check if degree_vector column exists in the table
            self.cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_education' 
                AND column_name = 'degree_vector'
            """)
            has_vector_column = self.cursor.fetchone() is not None
            
            if has_vector_column:
                # Process with vector embeddings
                query = """
                INSERT INTO candidate_education 
                (candidate_id, school, university_rank, degree, start_year, end_year, degree_vector)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                # Preprocess data to add embedding vectors for degrees
                processed_data = []
                for row in data:
                    # Create a copy of the row as a list to allow modification
                    row_list = list(row)
                    # Generate embedding for degree (assuming degree is at index 3)
                    degree_embedding = self.generate_embedding(row[3]) if row[3] else None
                    # Append the embedding to the row
                    row_list.append(degree_embedding)
                    processed_data.append(tuple(row_list))
                    
                execute_values(self.cursor, query, processed_data)
            else:
                # Process without vector embeddings
                query = """
                INSERT INTO candidate_education 
                (candidate_id, school, university_rank, degree, start_year, end_year)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                execute_values(self.cursor, query, data)
                
            self.conn.commit()
            logger.info(f"Saved {len(data)} education records")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save education records: {str(e)}")
            raise

    def generate_embedding(self, text):
        """Generate embedding vector for text using an embedding model.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            list: Vector embedding
        """
        # This is a placeholder - in a real implementation, you would call an 
        # embedding service like OpenAI or a local model
        try:
            from openai import OpenAI
            client = OpenAI(api_key=load_config()["OPENAI_API_KEY"])
            
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None

    def fetch_profiles(self, search_term):
        """Fetch candidate profiles matching a search term using vector similarity.
        
        Args:
            search_term (str): Search term
            
        Returns:
            list: List of matching profiles
        """
        try:
            # Check if PostgreSQL has vector search capability
            try:
                # Try to get embedding for search term
                search_embedding = self.generate_embedding(search_term)
                
                if search_embedding:
                    # Vector similarity search using cosine distance
                    query = """
                    WITH ranked_experiences AS (
                        SELECT 
                            cp.candidate_id, 
                            cp.firstname, 
                            cp.lastname,
                            ce.company,
                            ce.company_industry,
                            ce.position,
                            ce.start_date,
                            ce.end_date,
                            ced.school,
                            ced.degree,
                            ced.university_rank,
                            1 - (ce.position_vector <=> %s) AS position_similarity_score
                        FROM candidate_profiles cp
                        JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                        LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                        WHERE ce.position_vector IS NOT NULL
                        ORDER BY position_similarity_score DESC
                        LIMIT 50
                    )
                    SELECT * FROM ranked_experiences
                    """
                    self.cursor.execute(query, (search_embedding,))
                else:
                    # Fallback to text search if embedding fails
                    raise Exception("Vector search not available, falling back to text search")
            except Exception as e:
                # Fallback to traditional SQL search
                logger.warning(f"Vector search failed, using text search instead: {str(e)}")
                query = """
                SELECT 
                    cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                    ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                    ced.university_rank
                FROM candidate_profiles cp
                LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
                LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
                WHERE ce.position ILIKE %s OR ce.company_industry ILIKE %s
                LIMIT 50
                """
                self.cursor.execute(query, (f"%{search_term}%", f"%{search_term}%"))
            
            results = self.cursor.fetchall()
            logger.debug(f"Fetched {len(results)} profiles for '{search_term}'")
            return results
        except Exception as e:
            logger.error(f"Failed to fetch profiles: {str(e)}")
            self.conn.rollback()
            raise

    def close(self):
        """Close the cursor and return the connection to the pool."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.rollback()  # Rollback any uncommitted transactions
            self.pool.putconn(self.conn)
            logger.info("PostgreSQL database connection returned to pool")
            self.conn = None
            self.cursor = None