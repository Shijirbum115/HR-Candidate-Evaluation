import mysql.connector
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()

class CloudDatabase:
    def __init__(self):
        config = load_config()
        self.conn = mysql.connector.connect(
            host=config["CLOUD_DB_HOST"],
            user=config["CLOUD_DB_USER"],
            password=config["CLOUD_DB_PASSWORD"],
            database=config["CLOUD_DB_NAME"]
        )
        self.cursor = self.conn.cursor()
        logger.info("Cloud database connection established")

    def fetch_candidate_data(self, batch_size=1000, offset=0):
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
        LIMIT %s OFFSET %s
        """
        self.cursor.execute(query, (batch_size, offset))
        return self.cursor.fetchall()

    def get_total_rows(self):
        query = """
        SELECT COUNT(*) 
        FROM career_user_data cud
        LEFT JOIN career_cv_exprs cve ON cud.id = cve.cv_id
        LEFT JOIN career_cv_edus ce ON cud.id = ce.cv_id
        LEFT JOIN career_site_data csd ON cve.branch_id = csd.option_id AND csd.grp_id = 3
        """
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    def close(self):
        self.cursor.close()
        self.conn.close()
        logger.info("Cloud database connection closed")

class LocalDatabase:
    def __init__(self):
        config = load_config()
        self.conn = mysql.connector.connect(
            host=config["LOCAL_DB_HOST"],
            user=config["LOCAL_DB_USER"],
            password=config["LOCAL_DB_PASSWORD"],
            database=config["LOCAL_DB_NAME"]
        )
        self.cursor = self.conn.cursor()
        logger.info("Local database connection established")

    def save_candidates(self, data):
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

    def save_experiences(self, data):
        query = """
        INSERT IGNORE INTO candidate_experiences 
        (candidate_id, company, company_industry, position, start_date, end_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cursor.executemany(query, data)
        self.conn.commit()
        logger.info(f"Saved {len(data)} experiences")

    def save_education(self, data):
        query = """
        INSERT IGNORE INTO candidate_education 
        (candidate_id, school, university_rank, degree, start_year, end_year)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cursor.executemany(query, data)
        self.conn.commit()
        logger.info(f"Saved {len(data)} education records")

    def fetch_profiles(self, search_term):
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
        WHERE cp.candidate_id IN (%s)
        """
        # Dynamically create placeholders for candidate_ids
        placeholders = ','.join(['%s'] * len(candidate_ids))
        self.cursor.execute(query % placeholders, tuple(candidate_ids))
        results = self.cursor.fetchall()
        logger.debug(f"Fetched {len(results)} profiles for '{search_term}'")
        return results

    def close(self):
        self.cursor.close()
        self.conn.close()
        logger.info("Local database connection closed")