# import sys
# import os
# import time
# import logging
# import argparse
# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import SQLAlchemyError
# from tenacity import retry, stop_after_attempt, wait_exponential
# import urllib.parse

# # Add the project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.utils.config import load_config
# from src.utils.logger import setup_logging

# logger = setup_logging()
# logger.setLevel(logging.INFO)

# class DatabaseMigration:
#     def __init__(self):
#         """Initialize database connections and configuration."""
#         config = load_config()
        
#         mysql_user = urllib.parse.quote_plus(config['CLOUD_DB_USER'])
#         mysql_password = urllib.parse.quote_plus(config['CLOUD_DB_PASSWORD'])
#         pg_user = urllib.parse.quote_plus(config['PG_DB_USER'])
#         pg_password = urllib.parse.quote_plus(config['PG_DB_PASSWORD'])
        
#         # Source MySQL connection string
#         self.source_engine = create_engine(
#     f"mysql+pymysql://{mysql_user}:{mysql_password}@"
#     f"{config['CLOUD_DB_HOST']}/{config['CLOUD_DB_NAME']}?charset=utf8mb4"
# )
        
#         # Destination PostgreSQL connection string with encoded credentials
#         self.dest_engine = create_engine(
#             f"postgresql://{pg_user}:{pg_password}@"
#             f"{config['PG_DB_HOST']}/{config['PG_DB_NAME']}"
#         )
        
#         # Create a checkpoint file path to track progress
#         self.checkpoint_file = "migration_checkpoint.txt"
        
#     def get_last_processed_offset(self):
#         """Read the last processed offset from checkpoint file."""
#         if os.path.exists(self.checkpoint_file):
#             with open(self.checkpoint_file, 'r') as f:
#                 try:
#                     return int(f.read().strip())
#                 except ValueError:
#                     return 0
#         return 0
    
#     def save_checkpoint(self, offset):
#         """Save the current offset to checkpoint file."""
#         with open(self.checkpoint_file, 'w') as f:
#             f.write(str(offset))
    
#     def check_destination_schema(self):
#         """Verify destination tables exist with correct structure."""
#         try:
#             with self.dest_engine.connect() as conn:
#                 # Use text() for raw SQL
#                 result = conn.execute(text("""
#                     SELECT table_name FROM information_schema.tables 
#                     WHERE table_schema = 'public'
#                 """))
                
#                 tables = result.fetchall()
#                 table_names = [t[0] for t in tables]
#                 required_tables = ['candidate_profiles', 'candidate_experiences', 'candidate_education']
                
#                 missing_tables = [t for t in required_tables if t not in table_names]
#                 if missing_tables:
#                     logger.error(f"Missing required tables: {missing_tables}")
#                     return False
                    
#             logger.info("Destination schema verified successfully")
#             return True
#         except Exception as e:
#             logger.error(f"Error checking destination schema: {str(e)}")
#             return False
    
#     def clear_destination_tables(self):
#         """Clear all data from destination tables to start fresh."""
#         try:
#             with self.dest_engine.connect() as conn:
#                 # Use text() for raw SQL statements
#                 conn.execute(text("TRUNCATE TABLE candidate_education CASCADE"))
#                 conn.execute(text("TRUNCATE TABLE candidate_experiences CASCADE"))
#                 conn.execute(text("TRUNCATE TABLE candidate_profiles CASCADE"))
#                 # Commit the changes
#                 conn.commit()
#             logger.info("Destination tables cleared successfully")
#         except Exception as e:
#             logger.error(f"Error clearing destination tables: {str(e)}")
#             raise
    
#     def get_total_rows(self):
#         """Get total count of records to migrate."""
#         query = """
#         SELECT COUNT(DISTINCT cud.id) 
#         FROM career_user_data cud
#         """
#         with self.source_engine.connect() as conn:
#             result = conn.execute(text(query))
#             return result.scalar_one()
    
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#     def fetch_batch(self, batch_size, offset):
#         """Fetch a batch of data with retry logic."""
#         query = """
#         SELECT 
#             cud.id AS candidate_id,
#             cud.birthdate,
#             cve.company,
#             csd.title AS company_industry,
#             cve.position,
#             cve.start AS expr_start,
#             cve.end AS expr_end,
#             ce.school,
#             ce.pro,
#             ce.start AS edu_start,
#             ce.end AS edu_end,
#             cud.firstname,
#             cud.lastname
#         FROM career_user_data cud
#         LEFT JOIN career_cv_exprs cve ON cud.id = cve.cv_id
#         LEFT JOIN career_cv_edus ce ON cud.id = ce.cv_id
#         LEFT JOIN career_site_data csd ON cve.branch_id = csd.option_id AND csd.grp_id = 3
#         ORDER BY cud.id
#         LIMIT :limit OFFSET :offset
#         """
#         try:
#             with self.source_engine.connect() as conn:
#                 df = pd.read_sql(
#                     text(query), 
#                     conn, 
#                     params={"limit": batch_size, "offset": offset}
#                 )
#             return df
#         except Exception as e:
#             logger.error(f"Error fetching batch at offset {offset}: {str(e)}")
#             raise
    
#     def process_batch(self, df):
#         """Process a batch of data - extract candidate, experience, and education info."""
#         if df.empty:
#             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
#         # Process candidates - only unique records
#         candidates = df[['candidate_id', 'birthdate', 'firstname', 'lastname']].drop_duplicates('candidate_id')
        
#         # Process experiences - only rows with company data
#         experiences = df[df['company'].notna()][
#             ['candidate_id', 'company', 'company_industry', 'position', 'expr_start', 'expr_end']
#         ].rename(columns={'expr_start': 'start_date', 'expr_end': 'end_date'}).drop_duplicates()
        
#         # Process education - only rows with school data
#         education = df[df['school'].notna()][
#             ['candidate_id', 'school', 'pro', 'edu_start', 'edu_end']
#         ].rename(columns={
#             'pro': 'degree',
#             'edu_start': 'start_year',
#             'edu_end': 'end_year'
#         }).drop_duplicates()
        
#         # Add university_rank placeholder - to be filled by preprocessing later
#         education['university_rank'] = 1
        
#         # Convert date columns if needed
#         if not education.empty:
#             # Extract year from dates if they are date objects
#             for col in ['start_year', 'end_year']:
#                 if education[col].dtype == 'datetime64[ns]':
#                     education[col] = education[col].dt.year
        
#         return candidates, experiences, education
    
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#     def save_batch(self, candidates, experiences, education):
#         """Save processed data to destination with retry logic."""
#         try:
#             # Begin a transaction
#             with self.dest_engine.begin() as conn:
#                 # Save candidates
#                 if not candidates.empty:
#                     candidates.to_sql('candidate_profiles', conn, if_exists='append', 
#                                       index=False, method='multi', chunksize=500)
#                     logger.info(f"Saved {len(candidates)} candidates")
                
#                 # Save experiences
#                 if not experiences.empty:
#                     experiences.to_sql('candidate_experiences', conn, if_exists='append', 
#                                       index=False, method='multi', chunksize=500)
#                     logger.info(f"Saved {len(experiences)} experiences")
                
#                 # Save education
#                 if not education.empty:
#                     education.to_sql('candidate_education', conn, if_exists='append', 
#                                     index=False, method='multi', chunksize=500)
#                     logger.info(f"Saved {len(education)} education records")
                
#                 # Transaction will be committed automatically if no errors occur
            
#             return True
#         except Exception as e:
#             logger.error(f"Error saving batch: {str(e)}")
#             # Transaction will be automatically rolled back
#             raise
    
#     def migrate(self, batch_size=1000, max_rows=None):
#         """Run the migration with checkpointing."""
#         start_time = time.time()
#         logger.info("Starting data migration")
        
#         # Resume from last checkpoint
#         offset = self.get_last_processed_offset()
#         logger.info(f"Resuming from offset: {offset}")
        
#         total = self.get_total_rows()
#         if max_rows:
#             total = min(total, max_rows)
            
#         logger.info(f"Will process {total} total rows")
        
#         candidates_migrated = 0
#         experiences_migrated = 0
#         education_migrated = 0
        
#         try:
#             for current_offset in range(offset, total, batch_size):
#                 batch_start = time.time()
                
#                 # Calculate current progress
#                 current_count = min(current_offset + batch_size, total)
#                 progress_percent = current_count / total * 100
#                 logger.info(f"Processing batch: {current_offset+1}-{current_count} of {total} ({progress_percent:.1f}%)")
                
#                 # Fetch batch with retry logic
#                 df = self.fetch_batch(batch_size, current_offset)
#                 if df.empty:
#                     logger.warning(f"Empty batch at offset {current_offset}")
#                     self.save_checkpoint(current_offset + batch_size)
#                     continue
                
#                 # Process the batch
#                 candidates, experiences, education = self.process_batch(df)
                
#                 # Save to destination
#                 self.save_batch(candidates, experiences, education)
                
#                 # Update counts
#                 candidates_migrated += len(candidates)
#                 experiences_migrated += len(experiences)
#                 education_migrated += len(education)
                
#                 # Update checkpoint after successful save
#                 self.save_checkpoint(current_offset + batch_size)
                
#                 # Log batch timing
#                 batch_time = time.time() - batch_start
#                 logger.info(f"Batch completed in {batch_time:.2f}s - Progress: {current_count}/{total} rows")
                
#                 # Add a small delay to prevent overwhelming the database
#                 time.sleep(0.2)
                
#             # Migration complete
#             elapsed = time.time() - start_time
#             logger.info(f"Migration completed in {elapsed:.2f} seconds")
#             logger.info(f"Summary: {candidates_migrated} candidates, {experiences_migrated} experiences, {education_migrated} education records")
            
#         except Exception as e:
#             logger.error(f"Migration error at offset {offset}: {str(e)}")
#             logger.info(f"Migration can be resumed from offset {offset} using the checkpoint file")
#             raise

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Migrate from MySQL to PostgreSQL')
#     parser.add_argument('--batch', type=int, default=1000, help='Batch size')
#     parser.add_argument('--max', type=int, default=None, help='Maximum rows to process')
#     parser.add_argument('--reset', action='store_true', help='Start fresh by clearing destination tables')
#     args = parser.parse_args()
    
#     migration = DatabaseMigration()
    
#     if args.reset:
#         # Remove checkpoint file
#         if os.path.exists(migration.checkpoint_file):
#             os.remove(migration.checkpoint_file)
#             logger.info("Removed existing checkpoint file")
        
#         # Clear destination tables
#         migration.clear_destination_tables()
#         logger.info("Reset complete - starting migration from beginning")
    
#     # Validate schema
#     if not migration.check_destination_schema():
#         logger.error("Schema validation failed. Please fix before proceeding.")
#         sys.exit(1)
        
#     # Run migration
#     migration.migrate(batch_size=args.batch, max_rows=args.max)