import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.db_connect import CloudDatabase
from src.data.pg_connect import LocalPostgresDatabase  # New PostgreSQL connection
from src.utils.logger import setup_logging
import time
import logging

logger = setup_logging()
logger.setLevel(logging.INFO)

def migrate_to_postgres(batch_size=1000, max_rows=None):
    """Migrate data from MySQL to PostgreSQL with pgVector."""
    start_time = time.time()
    logger.info("Starting MySQL to PostgreSQL migration")
    
    try:
        # Connect to databases
        logger.info("Connecting to databases...")
        cloud_db = CloudDatabase()  # Source MySQL
        pg_db = LocalPostgresDatabase()  # Destination PostgreSQL
        logger.info("Database connections established")
        
        # Get total count
        total = cloud_db.get_total_rows()
        if max_rows:
            total = min(total, max_rows)
        logger.info(f"Will process {total} rows (max_rows={max_rows})")
        
        # Migration counters
        candidates_migrated = 0
        experiences_migrated = 0
        education_migrated = 0
        
        # Process in batches
        for offset in range(0, total, batch_size):
            batch_start = time.time()
            current_count = min(offset + batch_size, total)
            logger.info(f"Processing batch: {offset+1}-{current_count} of {total} ({current_count/total*100:.1f}%)")
            
            rows = cloud_db.fetch_candidate_data(batch_size, offset)
            
            # Process the batch...
            # [existing batch processing code]
            
            # Log batch completion with timing
            batch_time = time.time() - batch_start
            logger.info(f"Batch completed in {batch_time:.2f}s - Running total: {candidates_migrated} candidates")
            
            # Optional: Add a small delay to make logs more readable during testing
            # time.sleep(0.5)
            
        # Close connections
        cloud_db.close()
        pg_db.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Migration completed in {elapsed:.2f} seconds")
        logger.info(f"Summary: {candidates_migrated} candidates, {experiences_migrated} experiences, {education_migrated} education records")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Migrate from MySQL to PostgreSQL')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size')
    parser.add_argument('--max', type=int, default=None, help='Maximum rows to process')
    args = parser.parse_args()
    
    migrate_to_postgres(batch_size=args.batch, max_rows=args.max)