import sys
import os
from datetime import datetime, date
import logging
from src.utils.logger import setup_logging
from scripts.migration.database import MySQLConnector, PostgreSQLConnector

logger = setup_logging()

def convert_unix_to_date(unix_timestamp):
    """Convert Unix timestamp to datetime.date object.
    
    Args:
        unix_timestamp: Unix timestamp (seconds since epoch)
        
    Returns:
        datetime.date: Converted date or None if conversion fails
    """
    if not unix_timestamp:
        return None
        
    try:
        # Convert to integer if it's a string
        if isinstance(unix_timestamp, str):
            unix_timestamp = int(unix_timestamp)
            
        # Convert Unix timestamp to datetime
        dt = datetime.fromtimestamp(unix_timestamp)
        return dt.date()
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to convert unix timestamp {unix_timestamp}: {str(e)}")
        return None

def migrate_last_login_dates():
    """Migrate last_login data from MySQL to PostgreSQL.
    
    This function fetches last_login timestamps from the career_user_data table
    in MySQL, converts them to dates, and updates the candidate_profiles table
    in PostgreSQL with this information.
    """
    mysql_conn = MySQLConnector()
    pg_conn = PostgreSQLConnector()
    
    try:
        # Connect to both databases
        if not mysql_conn.connect() or not pg_conn.connect():
            logger.error("Failed to connect to one or both databases")
            return
        
        # Create cursor for MySQL connection
        mysql_cursor = mysql_conn.conn.cursor(buffered=True)
        
        # Get total number of candidates to process
        mysql_cursor.execute("SELECT COUNT(id) FROM career_user_data")
        total_candidates = mysql_cursor.fetchone()[0]
        logger.info(f"Processing last_login dates for {total_candidates} candidates")
        
        # Process in batches to avoid memory issues
        batch_size = 500
        processed = 0
        
        while processed < total_candidates:
            # Fetch a batch of candidate data with last_login
            mysql_cursor.execute("""
                SELECT id AS candidate_id, last_login
                FROM career_user_data
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (batch_size, processed))
            
            candidates = mysql_cursor.fetchall()
            if not candidates:
                break
                
            # Prepare data for PostgreSQL update
            update_data = []
            for candidate in candidates:
                candidate_id = candidate[0]
                unix_timestamp = candidate[1]
                login_date = convert_unix_to_date(unix_timestamp)
                
                if login_date:
                    update_data.append((login_date, candidate_id))
            
            # Update PostgreSQL in batches
            if update_data:
                pg_cursor = pg_conn.conn.cursor()
                for login_date, candidate_id in update_data:
                    pg_cursor.execute("""
                        UPDATE candidate_profiles
                        SET last_login_date = %s
                        WHERE candidate_id = %s
                    """, (login_date, candidate_id))
                pg_conn.conn.commit()
                pg_cursor.close()
                
                logger.info(f"Updated {len(update_data)} last_login dates")
            
            processed += len(candidates)
            logger.info(f"Progress: {processed}/{total_candidates} candidates processed")
        
        mysql_cursor.close()
        logger.info("Last login date migration completed successfully")
    except Exception as e:
        logger.error(f"Error during last_login migration: {str(e)}")
        if pg_conn.conn:
            pg_conn.conn.rollback()
    finally:
        mysql_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    logger.info("Starting last_login date migration")
    migrate_last_login_dates()
    logger.info("Migration process completed")