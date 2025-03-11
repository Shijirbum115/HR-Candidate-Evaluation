from datetime import datetime, date
from rapidfuzz import fuzz
from openai import OpenAI
from src.utils.config import load_config
import logging

logger = logging.getLogger(__name__)
client = OpenAI(api_key=load_config()["OPENAI_API_KEY"])

def calculate_score(profile, search_term):
    """Calculate candidate score based on school, experience, and industry match."""
    try:
        # Parse search term with AI, stricter prompt
        prompt = f"From '{search_term}', extract only the job title and industry as two separate items. Ignore adjectives or extra words like 'experienced'. Return 'job_title: [title]' and 'industry: [industry]' or 'None' if no industry. Example: 'Туршлагатай санхүүгийн менежер' -> 'job_title: Санхүүгийн менежер', 'industry: None'"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        parsed = response.choices[0].message.content.split("\n")
        job_title = search_term  # Fallback to full term
        industry = None
        for line in parsed:
            if "job_title:" in line:
                job_title = line.split("job_title:")[1].strip()
            if "industry:" in line and "None" not in line:
                industry = line.split("industry:")[1].strip()

        # Log parsing result for debugging
        logger.debug(f"Parsed job_title: {job_title}, industry: {industry}")

        # School (x): 1-4 from stored university_rank, default to 1 if None
        x = profile["university_rank"] if profile["university_rank"] is not None else 1
        
        # Experience (y): Duration * Relevance, max 6
        start = profile["start_date"]
        end = profile["end_date"] if profile["end_date"] else date.today()
        y = 0
        if start and isinstance(start, date):
            years = min((end - start).days / 365.25, 5)
            relevance = fuzz.token_sort_ratio(profile["position"].lower(), job_title.lower()) / 100
            y = years * relevance if relevance > 0.7 else 0  # Lower threshold to 70%
            y = min(y, 6)
        
        # Industry (z): 0-2, AI-driven
        z = 0
        if industry and profile["company_industry"]:
            if industry.lower() in profile["company_industry"].lower():
                z = 2
            elif fuzz.token_sort_ratio(industry.lower(), profile["company_industry"].lower()) > 80:
                z = 1
        
        # Total: Normalize to 0-10 (max raw score = 12: 4 + 6 + 2)
        total = (x + y + z) / 12 * 10
        logger.debug(f"Score: {total:.2f} (x={x}, y={y}, z={z})")
        return total
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return 0