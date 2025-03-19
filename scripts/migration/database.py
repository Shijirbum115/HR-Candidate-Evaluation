import mysql.connector
import psycopg2
from psycopg2.extras import execute_values
import logging
import time
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()

class MySQLConnector:
    """Class for connecting to and querying MySQL database."""
    
    def __init__(self):
        """Initialize configuration."""
        self.config = load_config()
        self.conn = None
    
    def connect(self):
        """Connect to MySQL database."""
        try:
            # Use a fresh connection each time, not pooling which can cause issues
            self.conn = mysql.connector.connect(
                host=self.config["CLOUD_DB_HOST"],
                user=self.config["CLOUD_DB_USER"],
                password=self.config["CLOUD_DB_PASSWORD"],
                database=self.config["CLOUD_DB_NAME"],
                charset='utf8mb4',
                use_pure=True,
                connect_timeout=60
            )
            logger.info("Connected to MySQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("MySQL database connection closed")
    
    def fetch_candidate_batch(self, batch_size, offset):
        """Fetch a batch of candidate data from MySQL."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor(dictionary=True)
            query = """
            SELECT 
                id AS candidate_id,
                birthdate,
                firstname,
                lastname
            FROM career_user_data
            ORDER BY id
            LIMIT %s OFFSET %s
            """
            cursor.execute(query, (batch_size, offset))
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching candidate batch at offset {offset}: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.close()
                self.connect()
                return self.fetch_candidate_batch(batch_size, offset)
            except Exception as e2:
                logger.error(f"Failed to retry fetch_candidate_batch: {str(e2)}")
                raise
    
    def fetch_experiences(self, candidate_ids):
        """Fetch experiences for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            # Make a fresh connection for each query to avoid cursor issues
            self.close()
            self.connect()
            
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.conn.cursor(dictionary=True)
            query = f"""
            SELECT 
                cve.cv_id AS candidate_id,
                cve.company,
                csd.title AS company_industry,
                cve.position,
                cve.start AS expr_start,
                cve.end AS expr_end
            FROM career_cv_exprs cve
            LEFT JOIN career_site_data csd ON cve.branch_id = csd.option_id AND csd.grp_id = 3
            WHERE cve.cv_id IN ({id_list})
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching experiences: {str(e)}")
            raise
    
    def fetch_education(self, candidate_ids):
        """Fetch education for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            # Make a fresh connection for each query to avoid cursor issues
            self.close()
            self.connect()
            
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.conn.cursor(dictionary=True)
            query = f"""
            SELECT 
                cv_id AS candidate_id,
                school,
                pro,
                start AS edu_start,
                end AS edu_end
            FROM career_cv_edus
            WHERE cv_id IN ({id_list})
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching education: {str(e)}")
            raise
    
    def get_total_candidates(self):
        """Get total count of candidate records to migrate."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM career_user_data")
            total = cursor.fetchone()[0]
            cursor.close()
            
            return total
        except Exception as e:
            logger.error(f"Error getting total candidates: {str(e)}")
            raise
    
    def fetch_distinct_universities(self, limit=10000):
        """Fetch distinct university names for clustering."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT DISTINCT school 
                FROM career_cv_edus 
                WHERE school IS NOT NULL AND school != '' 
                LIMIT {limit}
            """)
            universities = [row[0] for row in cursor.fetchall() if row[0]]
            cursor.close()
            
            return universities
        except Exception as e:
            logger.error(f"Error fetching distinct universities: {str(e)}")
            raise
    
    def fetch_distinct_positions(self, limit=10000):
        """Fetch distinct position names for clustering."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT DISTINCT position 
                FROM career_cv_exprs 
                WHERE position IS NOT NULL AND position != '' 
                LIMIT {limit}
            """)
            positions = [row[0] for row in cursor.fetchall() if row[0]]
            cursor.close()
            
            return positions
        except Exception as e:
            logger.error(f"Error fetching distinct positions: {str(e)}")
            raise


class PostgreSQLConnector:
    """Class for connecting to and querying PostgreSQL database."""
    
    def __init__(self):
        """Initialize configuration."""
        self.config = load_config()
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.config["PG_DB_HOST"],
                user=self.config["PG_DB_USER"],
                password=self.config["PG_DB_PASSWORD"],
                database=self.config["PG_DB_NAME"]
            )
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("PostgreSQL database connection closed")
    
    def check_destination_schema(self):
        """Verify destination tables exist with correct structure."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = cursor.fetchall()
            cursor.close()
            
            table_names = [t[0] for t in tables]
            required_tables = ['candidate_profiles', 'candidate_experiences', 'candidate_education']
            
            missing_tables = [t for t in required_tables if t not in table_names]
            if missing_tables:
                logger.error(f"Missing required tables: {missing_tables}")
                return False
                
            logger.info("Destination schema verified successfully")
            return True
        except Exception as e:
            logger.error(f"Error checking destination schema: {str(e)}")
            return False
    
    def check_vector_columns(self):
        """Check if vector columns exist, add them if needed."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            # Check if pgvector extension is installed
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone() is None:
                logger.info("Installing pgvector extension")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check for position_vector column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_experiences' 
                AND column_name = 'position_vector'
            """)
            if cursor.fetchone() is None:
                logger.info("Adding position_vector column to candidate_experiences")
                cursor.execute("ALTER TABLE candidate_experiences ADD COLUMN position_vector vector(1536);")
            else:
                # Use a simpler approach to check/reset the column
                try:
                    # Just drop and recreate to be safe
                    logger.info("Resetting position_vector column to ensure correct dimensions")
                    cursor.execute("ALTER TABLE candidate_experiences DROP COLUMN position_vector;")
                    cursor.execute("ALTER TABLE candidate_experiences ADD COLUMN position_vector vector(1536);")
                except Exception as e:
                    logger.error(f"Error resetting position_vector column: {str(e)}")
                    self.conn.rollback()  # Make sure to rollback on error
            
            # Check for degree_vector column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_education' 
                AND column_name = 'degree_vector'
            """)
            if cursor.fetchone() is None:
                logger.info("Adding degree_vector column to candidate_education")
                cursor.execute("ALTER TABLE candidate_education ADD COLUMN degree_vector vector(1536);")
            else:
                # Use a simpler approach to check/reset the column
                try:
                    # Just drop and recreate to be safe
                    logger.info("Resetting degree_vector column to ensure correct dimensions")
                    cursor.execute("ALTER TABLE candidate_education DROP COLUMN degree_vector;")
                    cursor.execute("ALTER TABLE candidate_education ADD COLUMN degree_vector vector(1536);")
                except Exception as e:
                    logger.error(f"Error resetting degree_vector column: {str(e)}")
                    self.conn.rollback()  # Make sure to rollback on error
            
            self.conn.commit()
            cursor.close()
            
            logger.info("Vector columns verified/added successfully")
            return True
        except Exception as e:
            logger.error(f"Error checking/adding vector columns: {str(e)}")
            if self.conn:
                self.conn.rollback()  # Make sure to rollback on error
            return False
    
    def clear_destination_tables(self):
        """Clear all data from destination tables to start fresh."""
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute("TRUNCATE TABLE candidate_education CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_experiences CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_profiles CASCADE")
            self.conn.commit()
            cursor.close()
            
            logger.info("Destination tables cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing destination tables: {str(e)}")
            raise
    
    def save_candidates(self, candidates):
        """Save candidate data to PostgreSQL."""
        if not candidates:
            return 0
            
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            query = """
            INSERT INTO candidate_profiles 
            (candidate_id, birthdate, firstname, lastname)
            VALUES %s
            ON CONFLICT (candidate_id) DO UPDATE SET
                birthdate = EXCLUDED.birthdate,
                firstname = EXCLUDED.firstname,
                lastname = EXCLUDED.lastname
            """
            execute_values(cursor, query, candidates)
            self.conn.commit()
            
            count = len(candidates)
            logger.info(f"Saved {count} candidates")
            cursor.close()
            
            return count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving candidates: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.close()
                self.connect()
                cursor = self.conn.cursor()
                execute_values(cursor, query, candidates)
                self.conn.commit()
                cursor.close()
                count = len(candidates)
                logger.info(f"Saved {count} candidates (retry)")
                return count
            except Exception as e2:
                logger.error(f"Failed to retry saving candidates: {str(e2)}")
                return 0
    
    def save_experiences(self, experiences):
        """Save experience data to PostgreSQL."""
        if not experiences:
            return 0
            
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            query = """
            INSERT INTO candidate_experiences 
            (candidate_id, company, company_industry, position, start_date, end_date)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, experiences)
            self.conn.commit()
            
            count = len(experiences)
            logger.info(f"Saved {count} experiences")
            cursor.close()
            
            return count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving experiences: {str(e)}")
            return 0
    
    def save_education(self, education):
        """Save education data to PostgreSQL."""
        if not education:
            return 0
            
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            query = """
            INSERT INTO candidate_education 
            (candidate_id, school, university_rank, degree, start_year, end_year)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, education)
            self.conn.commit()
            
            count = len(education)
            logger.info(f"Saved {count} education records")
            cursor.close()
            
            return count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving education: {str(e)}")
            return 0
    
    def update_experience_embeddings(self, data_pairs, batch_size=50):
        """Generate and update embeddings for position data."""
        if not data_pairs:
            return 0
        
        updated_count = 0
        
        # Update database
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            # Process in batches for efficiency
            for i in range(0, len(data_pairs), batch_size):
                batch = data_pairs[i:i+batch_size]
                
                for candidate_id, vector in batch:
                    query = """
                    UPDATE candidate_experiences
                    SET position_vector = %s
                    WHERE candidate_id = %s AND position_vector IS NULL
                    """
                    cursor.execute(query, (vector, candidate_id))
                
                self.conn.commit()
                updated_count += len(batch)
                logger.info(f"Updated {len(batch)} position embeddings")
            
            cursor.close()
            
            return updated_count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating position embeddings: {str(e)}")
            return 0
    
    def update_education_embeddings(self, data_pairs, batch_size=50):
        """Generate and update embeddings for degree data."""
        if not data_pairs:
            return 0
        
        updated_count = 0
        
        # Update database
        try:
            if not self.conn:
                self.connect()
                
            cursor = self.conn.cursor()
            
            # Process in batches for efficiency
            for i in range(0, len(data_pairs), batch_size):
                batch = data_pairs[i:i+batch_size]
                
                for candidate_id, vector in batch:
                    query = """
                    UPDATE candidate_education
                    SET degree_vector = %s
                    WHERE candidate_id = %s AND degree_vector IS NULL
                    """
                    cursor.execute(query, (vector, candidate_id))
                
                self.conn.commit()
                updated_count += len(batch)
                logger.info(f"Updated {len(batch)} degree embeddings")
            
            cursor.close()
            
            return updated_count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating degree embeddings: {str(e)}")
            return 0