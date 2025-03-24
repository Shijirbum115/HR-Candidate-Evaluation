# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.data.db_connect import CloudDatabase, LocalDatabase
# from src.utils.logger import setup_logging
# from src.preprocess.preprocess import preprocess_school
# import time
# import logging

# logger = setup_logging()
# logger.setLevel(logging.DEBUG)

# def process_cloud_data(batch_size=1000):
#     """Fetch data from cloud DB, preprocess schools, and save to local DB."""
#     start_time = time.time()
    
#     logger.debug("Starting process_cloud_data")
#     cloud_db = CloudDatabase()
#     local_db = LocalDatabase()
    
#     total = cloud_db.get_total_rows()
#     logger.info(f"Total rows to process: {total}")
    
#     for offset in range(0, total, batch_size):
#         logger.debug(f"Fetching batch at offset {offset}")
#         rows = cloud_db.fetch_candidate_data(batch_size, offset)
#         logger.info(f"Processing batch: {offset} to {offset + len(rows)}")
#         logger.debug(f"Batch sample (first row): {rows[0] if rows else 'Empty'}")
        
#         processed = []
#         for row in rows:
#             (candidate_id, birthdate, company, company_industry, position, 
#              start, end, school, pro, firstname, lastname) = row
#             cleaned_school, _ = preprocess_school(school)
#             processed.append((
#                 candidate_id, birthdate, company, company_industry, position,
#                 start, end, cleaned_school, pro, firstname, lastname
#             ))
        
#         if processed:
#             logger.debug(f"Saving {len(processed)} rows to local DB")
#             local_db.save_processed(processed)
    
#     cloud_db.close()
#     local_db.close()
#     elapsed = time.time() - start_time
#     logger.info(f"Processing completed in {elapsed:.2f} seconds")

# if __name__ == "__main__":
#     process_cloud_data(batch_size=1000)