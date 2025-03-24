# from datetime import date
# import logging

# logger = logging.getLogger(__name__)

# def calculate_vector_score(profile, search_embedding):
#     """Calculate candidate score using vector similarity and other factors."""
#     try:
#         # School ranking (x): 1-4, from stored university_rank
#         x = profile["university_rank"] if profile["university_rank"] is not None else 1
        
#         # Vector similarity score (vs): already calculated in DB query (0-1 range)
#         vs = profile["position_similarity_score"]
        
#         # Experience duration (ed): Years of experience, capped at 5 years
#         start = profile["start_date"]
#         end = profile["end_date"] if profile["end_date"] else date.today()
#         ed = 0
#         if start and isinstance(start, date):
#             ed = min((end - start).days / 365.25, 5) / 5  # Normalize to 0-1
        
#         # Combine factors with appropriate weights
#         # School ranking: 20%, Vector similarity: 50%, Experience duration: 30%
#         total = (0.2 * x/4) + (0.5 * vs) + (0.3 * ed)
        
#         # Scale to 0-10
#         scaled_score = total * 10
        
#         logger.debug(f"Vector score: {scaled_score:.2f} (x={x}, vs={vs:.2f}, ed={ed:.2f})")
#         return scaled_score
#     except Exception as e:
#         logger.error(f"Vector scoring failed: {str(e)}")
#         return 0