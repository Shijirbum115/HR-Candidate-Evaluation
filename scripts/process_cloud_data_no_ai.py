# import sys
# import os
# sys.stdout.reconfigure(encoding='utf-8')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.data.db_connect import CloudDatabase, LocalDatabase
# from src.utils.logger import setup_logging
# import time
# import logging

# logger = setup_logging()
# logger.setLevel(logging.DEBUG)

# def preprocess_school(school):
#     if not school or not isinstance(school, str) or not school.strip():
#         return "unknown", 1  # Unknown or invalid

#     school = school.strip().lower()

#     top_mongolian_unis = {
#         "шутис": "шинжлэх ухаан технологийн их сургууль",
#         "муис": "монгол улсын их сургууль",
#         "монгол улсын их сургууль": "монгол улсын их сургууль",
#         "хааис": "хөдөө аж ахуйн их сургууль",
#         "хааис эзбс": "хөдөө аж ахуйн их сургууль",
#         "сэзис": "санхүү эдийн засгийн их сургууль",
#         "шмтдс": "шинэ монгол технологийн дээд сургууль",
#         "суис": "соёл урлагийн их сургууль",
#         "ашиүис": "анагаахын шинжлэх ухааны үндэсний их сургууль",
#         "изис": "их засаг их сургууль",
#         "мубис": "монгол улсын боловсролын их сургууль",
#         "хүмүүнлэгийн ухааны их сургууль": "хүмүүнлэгийн ухааны их сургууль",
#         "мандах-бүртгэл дээд сургууль": "мандах-бүртгэл дээд сургууль",
#         "монголын үндэсний их сургууль": "монголын үндэсний их сургууль",
#         "олон улсын улаанбаатар их сургууль": "олон улсын улаанбаатар их сургууль",
#         "батлан хамгаалахын их сургууль": "батлан хамгаалахын их сургууль",
#         "отгонтэнгэр их сургууль": "отгонтэнгэр их сургууль"
#     }

#     for key, value in top_mongolian_unis.items():
#         if key in school:
#             return value, 4  # Top Mongolian
#     if any(term in school for term in ["их сургууль", "дээд сургууль"]):
#         return school, 3  # Other Mongolian university
#     if "коллеж" in school or "сургууль" in school:
#         return school, 2  # College or school
#     return school, 1  # Unknown or foreign

# def process_cloud_data_no_ai(batch_size=1000, max_rows=5000):
#     start_time = time.time()
#     logger.debug("Starting process_cloud_data_no_ai")
#     try:
#         cloud_db = CloudDatabase()
#         local_db = LocalDatabase()
        
#         total = max_rows
        
#         candidate_data = {}
#         experience_data = {}
#         education_data = {}
        
#         for offset in range(0, min(total, max_rows), batch_size):
#             logger.debug(f"Fetching batch at offset {offset}")
#             rows = cloud_db.fetch_candidate_data(batch_size, offset)
#             logger.info(f"Processing batch: {offset} to {offset + len(rows)}")
#             logger.debug(f"Batch sample (first row): {rows[0] if rows else 'Empty'}")
            
#             for row in rows:
#                 (candidate_id, birthdate, company, company_industry, position, 
#                  expr_start, expr_end, school, pro, edu_start, edu_end, 
#                  firstname, lastname) = row
                
#                 if candidate_id not in candidate_data:
#                     candidate_data[candidate_id] = (candidate_id, birthdate, firstname, lastname)
                
#                 if company and position:
#                     exp_key = (candidate_id, company, position, expr_start, expr_end)
#                     if exp_key not in experience_data:
#                         experience_data[exp_key] = (candidate_id, company, company_industry, position, expr_start, expr_end)
                
#                 if school and pro:
#                     edu_key = (candidate_id, school, pro, edu_start, edu_end)
#                     if edu_key not in education_data:
#                         cleaned_school, university_rank = preprocess_school(school)
#                         education_data[edu_key] = (candidate_id, cleaned_school, university_rank, pro, edu_start, edu_end)
        
#         if candidate_data:
#             logger.debug(f"Saving {len(candidate_data)} candidates to local DB")
#             local_db.save_candidates(list(candidate_data.values()))
#         if experience_data:
#             logger.debug(f"Saving {len(experience_data)} experiences to local DB")
#             local_db.save_experiences(list(experience_data.values()))
#         if education_data:
#             logger.debug(f"Saving {len(education_data)} education records to local DB")
#             local_db.save_education(list(education_data.values()))
        
#         cloud_db.close()
#         local_db.close()
#         elapsed = time.time() - start_time
#         logger.info(f"Processing completed in {elapsed:.2f} seconds")
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     process_cloud_data_no_ai(batch_size=1000, max_rows=5000)