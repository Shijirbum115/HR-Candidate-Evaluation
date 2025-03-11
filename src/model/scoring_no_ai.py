from datetime import datetime, date
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)

def calculate_score(profile, search_term):
    """Calculate candidate score based on school, experience, and industry match without AI."""
    try:
        # Use the full search term as job title, no parsing
        job_title = search_term.strip().lower()
        
        # Hardcoded industry keywords (extend as needed)
        industries = ["санхүү", "харилцаа холбоо", "уул уурхай", "боловсрол", "үйлчилгээ", "технологи"]
        industry = None
        for ind in industries:
            if ind in job_title:
                industry = ind
                job_title = job_title.replace(ind, "").strip()  # Remove industry from job title
                break

        # School (x): 1-4 from stored university_rank, default to 1 if None
        x = profile["university_rank"] if profile["university_rank"] is not None else 1
        
        # Experience (y): Duration * Relevance, max 6
        start = profile["start_date"]
        end = profile["end_date"] if profile["end_date"] else date.today()
        y = 0
        if start and isinstance(start, date):
            years = min((end - start).days / 365.25, 5)  # Cap at 5 years
            relevance = fuzz.token_sort_ratio(profile["position"].lower(), job_title) / 100
            y = years * relevance if relevance > 0.7 else 0  # Lower threshold to 70%
            y = min(y, 6)
        
        # Industry (z): 0-2, string matching
        z = 0
        if industry and profile["company_industry"]:
            if industry.lower() in profile["company_industry"].lower():
                z = 2  # Exact match
            elif fuzz.token_sort_ratio(industry.lower(), profile["company_industry"].lower()) > 70:
                z = 1  # Similar match
        
        # Total: Normalize to 0-10 (max raw score = 12: 4 + 6 + 2)
        total = (x + y + z) / 12 * 10
        logger.debug(f"Score: {total:.2f} (x={x}, y={y}, z={z})")
        return total
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return 0