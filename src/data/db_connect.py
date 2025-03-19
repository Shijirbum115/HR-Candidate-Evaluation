import mysql.connector
from mysql.connector import pooling
import time
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()

# Global connection pools
_cloud_pool = None
_local_pool = None

def get_cloud_pool():
    """Get or create a MySQL connection pool for cloud database."""
    global _cloud_pool
    if _cloud_pool is None:
        config = load_config()
        _cloud_pool = pooling.MySQLConnectionPool(
            pool_name="cloud_pool",
            pool_size=5,
            host=config["CLOUD_DB_HOST"],
            user=config["CLOUD_DB_USER"],
            password=config["CLOUD_DB_PASSWORD"],
            database=config["CLOUD_DB_NAME"],
            charset='utf8mb4',
            use_pure=True,  # Pure Python implementation for better Unicode support
            connect_timeout=60  # Longer timeout for stable connections
        )
        logger.info("Cloud database connection pool established")
    return _cloud_pool

def get_local_pool():
    """Get or create a MySQL connection pool for local database."""
    global _local_pool
    if _local_pool is None:
        config = load_config()
        _local_pool = pooling.MySQLConnectionPool(
            pool_name="local_pool",
            pool_size=5,
            host=config["LOCAL_DB_HOST"],
            user=config["LOCAL_DB_USER"],
            password=config["LOCAL_DB_PASSWORD"],
            database=config["LOCAL_DB_NAME"],
            charset='utf8mb4',
            use_pure=True
        )
        logger.info("Local database connection pool established")
    return _local_pool

class CloudDatabase:
    """Class for interacting with the cloud MySQL database with connection pooling."""
    
    def __init__(self):
        """Initialize connection from pool with retry logic."""
        self.pool = get_cloud_pool()
        self.conn = None
        self.cursor = None
        self._get_connection()
        logger.info("Cloud database connection established")

    def _get_connection(self):
        """Get a connection from the pool with retry logic."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.conn = self.pool.get_connection()
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
            self.conn.ping(reconnect=False, attempts=1, delay=0)
        except:
            logger.warning("Connection lost, reconnecting...")
            self.close()
            self._get_connection()

    def fetch_candidate_data(self, batch_size=1000, offset=0):
        """Fetch a batch of candidate data with connection check."""
        try:
            self._check_connection()
            
            query = """
            SELECT 
                cud.id AS candidate_id,
                cud.birthdate,
                cve.company,
                csd.title AS company_industry,
                cve.position,
                cve.start AS expr_start,
                cve.end AS expr_end,
                ce.school,
                ce.pro,
                ce.start AS edu_start,
                ce.end AS edu_end,
                cud.firstname,
                cud.lastname
            FROM career_user_data cud
            LEFT JOIN career_cv_exprs cve ON cud.id = cve.cv_id
            LEFT JOIN career_cv_edus ce ON cud.id = ce.cv_id
            LEFT JOIN career_site_data csd ON cve.branch_id = csd.option_id AND csd.grp_id = 3
            ORDER BY cud.id
            LIMIT %s OFFSET %s
            """
            self.cursor.execute(query, (batch_size, offset))
            return self.cursor.fetchall()
        except mysql.connector.Error as e:
            logger.error(f"MySQL error fetching candidate data: {str(e)}")
            # Attempt to reconnect and retry once
            self._get_connection()
            self.cursor.execute(query, (batch_size, offset))
            return self.cursor.fetchall()

    def get_total_rows(self):
        """Get total count of records to migrate with connection check."""
        try:
            self._check_connection()
            
            query = """
            SELECT COUNT(DISTINCT cud.id) 
            FROM career_user_data cud
            """
            self.cursor.execute(query)
            return self.cursor.fetchone()[0]
        except mysql.connector.Error as e:
            logger.error(f"MySQL error getting total rows: {str(e)}")
            # Attempt to reconnect and retry once
            self._get_connection()
            self.cursor.execute(query)
            return self.cursor.fetchone()[0]

    def close(self):
        """Close cursor and return connection to pool."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.cursor = None
        self.conn = None
        logger.info("Cloud database connection closed")

class LocalDatabase:
    """Class for interacting with the local MySQL database with connection pooling."""
    
    def __init__(self):
        """Initialize connection from pool."""
        self.pool = get_local_pool()
        self.conn = None
        self.cursor = None
        self._get_connection()
        logger.info("Local database connection established")

    def _get_connection(self):
        """Get a connection from the pool with retry logic."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.conn = self.pool.get_connection()
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
            self.conn.ping(reconnect=False, attempts=1, delay=0)
        except:
            logger.warning("Connection lost, reconnecting...")
            self.close()
            self._get_connection()

    def save_candidates(self, data):
        """Save candidate profile data with connection check."""
        try:
            self._check_connection()
            
            query = """
            INSERT INTO candidate_profiles 
            (candidate_id, birthdate, firstname, lastname)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                birthdate = VALUES(birthdate),
                firstname = VALUES(firstname),
                lastname = VALUES(lastname)
            """
            self.cursor.executemany(query, data)
            self.conn.commit()
            logger.info(f"Saved {len(data)} candidates")
        except mysql.connector.Error as e:
            self.conn.rollback()
            logger.error(f"MySQL error saving candidates: {str(e)}")
            # Try to reconnect and retry
            self._get_connection()
            self.cursor.executemany(query, data)
            self.conn.commit()

    def save_experiences(self, data):
        """Save work experience data with connection check."""
        try:
            self._check_connection()
            
            query = """
            INSERT IGNORE INTO candidate_experiences 
            (candidate_id, company, company_industry, position, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.executemany(query, data)
            self.conn.commit()
            logger.info(f"Saved {len(data)} experiences")
        except mysql.connector.Error as e:
            self.conn.rollback()
            logger.error(f"MySQL error saving experiences: {str(e)}")
            # Try to reconnect and retry
            self._get_connection()
            self.cursor.executemany(query, data)
            self.conn.commit()

    def save_education(self, data):
        """Save education data with connection check."""
        try:
            self._check_connection()
            
            query = """
            INSERT IGNORE INTO candidate_education 
            (candidate_id, school, university_rank, degree, start_year, end_year)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.executemany(query, data)
            self.conn.commit()
            logger.info(f"Saved {len(data)} education records")
        except mysql.connector.Error as e:
            self.conn.rollback()
            logger.error(f"MySQL error saving education: {str(e)}")
            # Try to reconnect and retry
            self._get_connection()
            self.cursor.executemany(query, data)
            self.conn.commit()

    def fetch_profiles(self, search_term):
        """Fetch candidate profiles matching a search term with connection check."""
        try:
            self._check_connection()
            
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
            query = """
            SELECT 
                cp.candidate_id, cp.birthdate, ce.company, ce.company_industry, ce.position,
                ce.start_date, ce.end_date, ced.school, ced.degree, cp.firstname, cp.lastname,
                ced.university_rank
            FROM candidate_profiles cp
            LEFT JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
            LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
            WHERE cp.candidate_id IN ({})
            """.format(','.join(['%s'] * len(candidate_ids)))
            
            self.cursor.execute(query, tuple(candidate_ids))
            results = self.cursor.fetchall()
            logger.debug(f"Fetched {len(results)} profiles for '{search_term}'")
            return results
        except mysql.connector.Error as e:
            logger.error(f"MySQL error fetching profiles: {str(e)}")
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
            self.conn.close()
        self.cursor = None
        self.conn = None
        logger.info("Local database connection closed")