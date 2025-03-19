import sys
import os
import time
import logging
import argparse
import mysql.connector
import psycopg2
from datetime import datetime, date
from psycopg2.extras import execute_values

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()
logger.setLevel(logging.INFO)

def validate_date(date_value):
    """Validate a date value and return None if invalid."""
    if not date_value:
        return None
        
    # If it's already a date or datetime object, validate it
    if isinstance(date_value, (date, datetime)):
        try:
            # Validate by accessing components - will raise if invalid
            year = date_value.year
            month = date_value.month
            day = date_value.day
            
            # Check for obviously invalid values
            if month < 1 or month > 12 or day < 1 or day > 31:
                logger.warning(f"Invalid date components in {date_value}")
                return None
                
            return date_value
        except Exception as e:
            logger.warning(f"Invalid date object {date_value}: {str(e)}")
            return None
    
    # If it's a string, try to parse it
    if isinstance(date_value, str):
        try:
            # Try multiple formats
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
                try:
                    parsed_date = datetime.strptime(date_value, fmt).date()
                    return parsed_date
                except ValueError:
                    continue
                    
            # If we get here, no format worked
            logger.warning(f"Could not parse date string {date_value}")
            return None
        except Exception as e:
            logger.warning(f"Error handling date string {date_value}: {str(e)}")
            return None
    
    # Unrecognized type
    logger.warning(f"Unrecognized date type {type(date_value)}: {date_value}")
    return None

class OptimizedMigration:
    def __init__(self):
        """Initialize database configurations."""
        self.config = load_config()
        self.mysql_conn = None
        self.pg_conn = None
        
        # Create a checkpoint file path to track progress
        self.checkpoint_file = "migration_checkpoint.txt"
        
    def get_last_processed_offset(self):
        """Read the last processed offset from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return 0
        return 0
    
    def save_checkpoint(self, offset):
        """Save the current offset to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(offset))
    
    def connect_mysql(self):
        """Connect to MySQL database."""
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.config["CLOUD_DB_HOST"],
                user=self.config["CLOUD_DB_USER"],
                password=self.config["CLOUD_DB_PASSWORD"],
                database=self.config["CLOUD_DB_NAME"],
                charset='utf8mb4',
                use_pure=True,
                connect_timeout=60,
                # Add some buffer parameters
                buffered=True,
                pool_size=5
            )
            logger.info("Connected to MySQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False
    
    def connect_postgres(self):
        """Connect to PostgreSQL database."""
        try:
            self.pg_conn = psycopg2.connect(
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
    
    def close_connections(self):
        """Close database connections."""
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
        logger.info("Database connections closed")
    
    def check_destination_schema(self):
        """Verify destination tables exist with correct structure."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
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
    
    def clear_destination_tables(self):
        """Clear all data from destination tables to start fresh."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            cursor.execute("TRUNCATE TABLE candidate_education CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_experiences CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_profiles CASCADE")
            self.pg_conn.commit()
            cursor.close()
            
            logger.info("Destination tables cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing destination tables: {str(e)}")
            raise
    
    def get_total_candidates(self):
        """Get total count of candidate records to migrate."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM career_user_data")
            total = cursor.fetchone()[0]
            cursor.close()
            
            return total
        except Exception as e:
            logger.error(f"Error getting total candidates: {str(e)}")
            raise
    
    def fetch_candidate_batch(self, batch_size, offset):
        """Fetch a batch of candidate data from MySQL."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor(dictionary=True)
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
                self.connect_mysql()
                return self.fetch_candidate_batch(batch_size, offset)
            except:
                raise
    
    def fetch_experiences(self, candidate_ids):
        """Fetch experiences for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.mysql_conn.cursor(dictionary=True)
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
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_experiences(candidate_ids)
            except:
                raise
    
    def fetch_education(self, candidate_ids):
        """Fetch education for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.mysql_conn.cursor(dictionary=True)
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
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_education(candidate_ids)
            except:
                raise
    
    def process_candidates(self, candidates_data):
        """Process candidate data with date validation."""
        processed = []
        
        for row in candidates_data:
            # Validate birthdate
            birthdate = validate_date(row['birthdate'])
            
            processed.append((
                row['candidate_id'],
                birthdate,
                row['firstname'],
                row['lastname']
            ))
        
        return processed
    
    def process_experiences(self, experiences_data):
        """Process experience data with date validation."""
        processed = []
        
        for row in experiences_data:
            if row['company'] and row['position']:
                # Validate dates
                start_date = validate_date(row['expr_start'])
                end_date = validate_date(row['expr_end'])
                
                processed.append((
                    row['candidate_id'],
                    row['company'],
                    row['company_industry'],
                    row['position'],
                    start_date,
                    end_date
                ))
        
        return processed
    
    def process_education(self, education_data):
        """Process education data with date validation."""
        processed = []
        
        for row in education_data:
            if row['school'] and row['pro']:
                # Extract year or set to None for education dates
                start_year = None
                if row['edu_start']:
                    valid_date = validate_date(row['edu_start'])
                    start_year = valid_date.year if valid_date else None
                    
                end_year = None
                if row['edu_end']:
                    valid_date = validate_date(row['edu_end'])
                    end_year = valid_date.year if valid_date else None
                
                processed.append((
                    row['candidate_id'],
                    row['school'],
                    1,  # university_rank placeholder
                    row['pro'],
                    start_year,
                    end_year
                ))
        
        return processed
    
    def save_candidates(self, candidates):
        """Save candidate data to PostgreSQL."""
        if not candidates:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
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
            self.pg_conn.commit()
            
            count = len(candidates)
            logger.info(f"Saved {count} candidates")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving candidates: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, candidates)
                self.pg_conn.commit()
                cursor.close()
                count = len(candidates)
                logger.info(f"Saved {count} candidates (retry)")
                return count
            except:
                logger.error("Failed to save candidates even after reconnection")
                return 0
    
    def save_experiences(self, experiences):
        """Save experience data to PostgreSQL."""
        if not experiences:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            query = """
            INSERT INTO candidate_experiences 
            (candidate_id, company, company_industry, position, start_date, end_date)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, experiences)
            self.pg_conn.commit()
            
            count = len(experiences)
            logger.info(f"Saved {count} experiences")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving experiences: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, experiences)
                self.pg_conn.commit()
                cursor.close()
                count = len(experiences)
                logger.info(f"Saved {count} experiences (retry)")
                return count
            except:
                logger.error("Failed to save experiences even after reconnection")
                return 0
    
    def save_education(self, education):
        """Save education data to PostgreSQL."""
        if not education:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            query = """
            INSERT INTO candidate_education 
            (candidate_id, school, university_rank, degree, start_year, end_year)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, education)
            self.pg_conn.commit()
            
            count = len(education)
            logger.info(f"Saved {count} education records")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving education: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, education)
                self.pg_conn.commit()
                cursor.close()
                count = len(education)
                logger.info(f"Saved {count} education records (retry)")
                return count
            except:
                logger.error("Failed to save education even after reconnection")
                return 0
    
    def migrate(self, batch_size=1000, max_rows=None):
        """Run the migration with separate queries for better performance."""
        start_time = time.time()
        logger.info("Starting optimized data migration")
        
        # Resume from last checkpoint
        offset = self.get_last_processed_offset()
        logger.info(f"Resuming from offset: {offset}")
        
        # Establish connections
        if not self.connect_mysql():
            logger.error("Failed to connect to MySQL. Aborting migration.")
            return False
            
        if not self.connect_postgres():
            logger.error("Failed to connect to PostgreSQL. Aborting migration.")
            return False
        
        total = self.get_total_candidates()
        if max_rows:
            total = min(total, max_rows)
            
        logger.info(f"Will process {total} total candidates")
        
        candidates_migrated = 0
        experiences_migrated = 0
        education_migrated = 0
        
        try:
            # Process in batches
            for current_offset in range(offset, total, batch_size):
                batch_start = time.time()
                
                # Calculate current progress
                current_count = min(current_offset + batch_size, total)
                progress_percent = current_count / total * 100
                logger.info(f"Processing batch: {current_offset+1}-{current_count} of {total} ({progress_percent:.1f}%)")
                
                # Step 1: Fetch candidates
                fetch_start = time.time()
                logger.info(f"Fetching candidates batch at offset {current_offset}")
                candidates_batch = self.fetch_candidate_batch(batch_size, current_offset)
                fetch_time = time.time() - fetch_start
                logger.info(f"Fetched {len(candidates_batch)} candidates in {fetch_time:.2f}s")
                
                if not candidates_batch:
                    logger.warning(f"Empty candidates batch at offset {current_offset}")
                    self.save_checkpoint(current_offset + batch_size)
                    continue
                
                # Get candidate IDs for related data
                candidate_ids = [row['candidate_id'] for row in candidates_batch]
                
                # Step 2: Process and save candidates
                process_start = time.time()
                processed_candidates = self.process_candidates(candidates_batch)
                process_time = time.time() - process_start
                logger.info(f"Processed candidates in {process_time:.2f}s")
                
                save_start = time.time()
                candidates_count = self.save_candidates(processed_candidates)
                candidates_migrated += candidates_count
                save_time = time.time() - save_start
                logger.info(f"Saved candidates in {save_time:.2f}s")
                
                # Step 3: Fetch and process experiences (in smaller chunks if needed)
                if candidate_ids:
                    chunk_size = 500  # Process related data in smaller chunks if needed
                    for i in range(0, len(candidate_ids), chunk_size):
                        chunk_ids = candidate_ids[i:i+chunk_size]
                        
                        # Fetch experiences
                        fetch_start = time.time()
                        logger.info(f"Fetching experiences for {len(chunk_ids)} candidates")
                        experiences_batch = self.fetch_experiences(chunk_ids)
                        fetch_time = time.time() - fetch_start
                        logger.info(f"Fetched {len(experiences_batch)} experiences in {fetch_time:.2f}s")
                        
                        # Process and save experiences
                        if experiences_batch:
                            process_start = time.time()
                            processed_experiences = self.process_experiences(experiences_batch)
                            process_time = time.time() - process_start
                            logger.info(f"Processed experiences in {process_time:.2f}s")
                            
                            save_start = time.time()
                            exp_count = self.save_experiences(processed_experiences)
                            experiences_migrated += exp_count
                            save_time = time.time() - save_start
                            logger.info(f"Saved experiences in {save_time:.2f}s")
                
                # Step 4: Fetch and process education (in smaller chunks if needed)
                if candidate_ids:
                    for i in range(0, len(candidate_ids), chunk_size):
                        chunk_ids = candidate_ids[i:i+chunk_size]
                        
                        # Fetch education
                        fetch_start = time.time()
                        logger.info(f"Fetching education for {len(chunk_ids)} candidates")
                        education_batch = self.fetch_education(chunk_ids)
                        fetch_time = time.time() - fetch_start
                        logger.info(f"Fetched {len(education_batch)} education records in {fetch_time:.2f}s")
                        
                        # Process and save education
                        if education_batch:
                            process_start = time.time()
                            processed_education = self.process_education(education_batch)
                            process_time = time.time() - process_start
                            logger.info(f"Processed education in {process_time:.2f}s")
                            
                            save_start = time.time()
                            edu_count = self.save_education(processed_education)
                            education_migrated += edu_count
                            save_time = time.time() - save_start
                            logger.info(f"Saved education in {save_time:.2f}s")
                
                # Update checkpoint after successful processing of this batch
                self.save_checkpoint(current_offset + batch_size)
                
                # Log batch timing
                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s - Progress: {current_count}/{total} candidates")
                
            # Migration complete
            elapsed = time.time() - start_time
            logger.info(f"Migration completed in {elapsed:.2f} seconds")
            logger.info(f"Summary: {candidates_migrated} candidates, {experiences_migrated} experiences, {education_migrated} education records")
            
            return True
        except Exception as e:
            logger.error(f"Migration error at offset {offset}: {str(e)}")
            logger.info(f"Migration can be resumed from offset {offset} using the checkpoint file")
            raise
        finally:
            self.close_connections()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized MySQL to PostgreSQL Migration')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size for candidates')
    parser.add_argument('--max', type=int, default=None, help='Maximum candidates to process')
    parser.add_argument('--reset', action='store_true', help='Start fresh by clearing destination tables')
    args = parser.parse_args()
    
    migration = OptimizedMigration()
    
    if args.reset:
        # Remove checkpoint file
        if os.path.exists(migration.checkpoint_file):
            os.remove(migration.checkpoint_file)
            logger.info("Removed existing checkpoint file")
        
        # Clear destination tables
        migration.clear_destination_tables()
        logger.info("Reset complete - starting migration from beginning")
    
    # Validate schema
    if not migration.check_destination_schema():
        logger.error("Schema validation failed. Please fix before proceeding.")
        sys.exit(1)
        
    # Run migration
    success = migration.migrate(batch_size=args.batch, max_rows=args.max)
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed. Check the logs for details.")
        sys.exit(1)