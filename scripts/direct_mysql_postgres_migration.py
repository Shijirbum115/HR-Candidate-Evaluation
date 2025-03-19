import sys
import os
import time
import logging
import argparse
import mysql.connector
import psycopg2
import pandas as pd
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

class DatabaseMigration:
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
                connect_timeout=60
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
    
    def get_total_rows(self):
        """Get total count of records to migrate."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT id) FROM career_user_data")
            total = cursor.fetchone()[0]
            cursor.close()
            
            return total
        except Exception as e:
            logger.error(f"Error getting total rows: {str(e)}")
            raise
    
    def fetch_batch(self, batch_size, offset):
        """Fetch a batch of data from MySQL."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor(dictionary=True)
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
            cursor.execute(query, (batch_size, offset))
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching batch at offset {offset}: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_batch(batch_size, offset)
            except:
                raise
    
    def process_batch(self, batch_data):
        """Process a batch of data - extract candidate, experience, and education info."""
        if not batch_data:
            return [], [], []
            
        # Use dictionaries for deduplication
        candidates = {}
        experiences = {}
        education = {}
        
        for row in batch_data:
            # Process candidate data
            candidate_id = row['candidate_id']
            if candidate_id not in candidates:
                # Validate birthdate
                birthdate = validate_date(row['birthdate'])
                
                candidates[candidate_id] = (
                    candidate_id,
                    birthdate,  # Validated date that might be None if invalid
                    row['firstname'],
                    row['lastname']
                )
            
            # Process experience data if present
            if row['company'] and row['position']:
                exp_key = (candidate_id, row['company'], row['position'])
                if exp_key not in experiences:
                    # Validate dates
                    start_date = validate_date(row['expr_start'])
                    end_date = validate_date(row['expr_end'])
                    
                    experiences[exp_key] = (
                        candidate_id,
                        row['company'],
                        row['company_industry'],
                        row['position'],
                        start_date,
                        end_date
                    )
            
            # Process education data if present
            if row['school'] and row['pro']:
                edu_key = (candidate_id, row['school'], row['pro'])
                if edu_key not in education:
                    # Extract year or set to None for education dates
                    start_year = None
                    if row['edu_start']:
                        valid_date = validate_date(row['edu_start'])
                        start_year = valid_date.year if valid_date else None
                        
                    end_year = None
                    if row['edu_end']:
                        valid_date = validate_date(row['edu_end'])
                        end_year = valid_date.year if valid_date else None
                    
                    education[edu_key] = (
                        candidate_id,
                        row['school'],
                        1,  # university_rank placeholder
                        row['pro'],
                        start_year,
                        end_year
                    )
        
        return list(candidates.values()), list(experiences.values()), list(education.values())
    
    def save_batch(self, candidates, experiences, education):
        """Save processed data to PostgreSQL."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            # Save candidates
            if candidates:
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
                logger.info(f"Saved {len(candidates)} candidates")
            
            # Save experiences
            if experiences:
                query = """
                INSERT INTO candidate_experiences 
                (candidate_id, company, company_industry, position, start_date, end_date)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                execute_values(cursor, query, experiences)
                logger.info(f"Saved {len(experiences)} experiences")
            
            # Save education
            if education:
                query = """
                INSERT INTO candidate_education 
                (candidate_id, school, university_rank, degree, start_year, end_year)
                VALUES %s
                ON CONFLICT DO NOTHING
                """
                execute_values(cursor, query, education)
                logger.info(f"Saved {len(education)} education records")
            
            # Commit the transaction
            self.pg_conn.commit()
            
            return True
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving batch: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                return self.save_batch(candidates, experiences, education)
            except:
                raise
    
    def migrate(self, batch_size=1000, max_rows=None):
        """Run the migration with checkpointing."""
        start_time = time.time()
        logger.info("Starting data migration")
        
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
        
        total = self.get_total_rows()
        if max_rows:
            total = min(total, max_rows)
            
        logger.info(f"Will process {total} total rows")
        
        candidates_migrated = 0
        experiences_migrated = 0
        education_migrated = 0
        
        try:
            for current_offset in range(offset, total, batch_size):
                batch_start = time.time()
                
                # Calculate current progress
                current_count = min(current_offset + batch_size, total)
                progress_percent = current_count / total * 100
                logger.info(f"Processing batch: {current_offset+1}-{current_count} of {total} ({progress_percent:.1f}%)")
                
                # Fetch batch
                logger.info(f"Fetching batch at offset {current_offset}")
                fetch_start = time.time()
                batch_data = self.fetch_batch(batch_size, current_offset)
                fetch_end = time.time()
                logger.info(f"Fetch completed in {fetch_end - fetch_start:.2f}s")
                
                if not batch_data:
                    logger.warning(f"Empty batch at offset {current_offset}")
                    self.save_checkpoint(current_offset + batch_size)
                    continue
                
                # Process the batch
                logger.info(f"Processing batch data")
                process_start = time.time()
                candidates, experiences, education = self.process_batch(batch_data)
                process_end = time.time()
                logger.info(f"Processing completed in {process_end - process_start:.2f}s")
                
                # Save to destination
                logger.info(f"Saving to PostgreSQL")
                save_start = time.time()
                self.save_batch(candidates, experiences, education)
                save_end = time.time()
                logger.info(f"Save completed in {save_end - save_start:.2f}s")
                
                # Update counts
                candidates_migrated += len(candidates)
                experiences_migrated += len(experiences)
                education_migrated += len(education)
                
                # Update checkpoint after successful save
                self.save_checkpoint(current_offset + batch_size)
                
                # Log batch timing
                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s - Progress: {current_count}/{total} rows")
                
                # Add a small delay to prevent overwhelming the database
                time.sleep(0.2)
                
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
    parser = argparse.ArgumentParser(description='Migrate from MySQL to PostgreSQL')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size')
    parser.add_argument('--max', type=int, default=None, help='Maximum rows to process')
    parser.add_argument('--reset', action='store_true', help='Start fresh by clearing destination tables')
    args = parser.parse_args()
    
    migration = DatabaseMigration()
    
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