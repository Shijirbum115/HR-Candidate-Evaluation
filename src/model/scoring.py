from datetime import datetime, date
import logging
import re
import traceback
from typing import Tuple, Optional, Dict, Any, Union
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Constants for scoring weights
EDUCATION_WEIGHT = 0.20
POSITION_WEIGHT = 0.40
EXPERIENCE_WEIGHT = 0.25
INDUSTRY_WEIGHT = 0.15

# Constants for thresholds
RELEVANCE_THRESHOLD = 0.6
MAX_EXPERIENCE_YEARS = 6

# Global variables for model instances
sentence_model = None

try:
    # Use a multilingual model that supports Mongolian
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {str(e)}")
    sentence_model = None


def extract_search_criteria(search_term: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    Extract job title, industry and experience years from search term using regex patterns.
    
    Args:
        search_term: The search query
        
    Returns:
        Tuple containing: (job_title, industry, experience_years)
    """
    if not search_term or not isinstance(search_term, str):
        return search_term, None, None
        
    # Default values
    job_title = search_term
    industry = None
    experience_years = None
    
    # Experience years extraction patterns
    years_patterns = [
        r'(\d+)(?:\+)?\s*(?:жил|year|жилийн|түүнээс дээш жилийн|жилээс дээш)',
        r'(\d+)\s*(?:yr|yrs|жил)',
        r'experience\s*(?:of)?\s*(\d+)'
    ]
    
    for pattern in years_patterns:
        years_match = re.search(pattern, search_term, re.IGNORECASE)
        if years_match:
            try:
                experience_years = int(years_match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Industry extraction based on common terms
    industry_patterns = {
        'Санхүү': r'санхүү|finance|financial',
        'IT': r'it|айти|информацийн технологи|software|программ',
        'Маркетинг': r'маркетинг|marketing|зар сурталчилгаа',
        'Хууль': r'хууль|legal|law|attorney',
        'Боловсрол': r'боловсрол|education|training|сургалт',
        'Эрүүл мэнд': r'эрүүл мэнд|health|medical|эмнэлэг',
        'Худалдаа': r'худалдаа|sales|selling|retail',
        'Үйлчилгээ': r'үйлчилгээ|service|customer',
        'Барилга': r'барилга|construction|building',
        'Тээвэр': r'тээвэр|transportation|logistics'
    }
    
    for ind_name, pattern in industry_patterns.items():
        if re.search(pattern, search_term, re.IGNORECASE):
            industry = ind_name
            break
    
    # Job title extraction based on common patterns
    job_patterns = {
        'Санхүүгийн менежер': r'санхүүгийн\s+менежер|finance\s+manager',
        'Нягтлан бодогч': r'нягтлан\s+бодогч|accountant',
        'Дата аналист': r'дата\s+аналист|өгөгдлийн\s+аналист|data\s+analyst',
        'Программ хөгжүүлэгч': r'программ\s+хөгжүүлэгч|програм\s+хөгжүүлэгч|developer|программист',
        'Борлуулагч': r'борлуулагч|худалдагч|sales|seller',
        'Маркетингийн менежер': r'маркетингийн\s+менежер|marketing\s+manager'
    }
    
    for title, pattern in job_patterns.items():
        if re.search(pattern, search_term, re.IGNORECASE):
            job_title = title
            break
    
    logger.debug(f"Extracted criteria: position='{job_title}', industry='{industry}', experience={experience_years} years")
    return job_title, industry, experience_years


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2 or not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    
    # Clean and normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if not text1 or not text2:
        return 0.0
    
    # Exact match gets highest score
    if text1 == text2:
        return 1.0
    
    # Check if one text is contained in the other
    if text1 in text2 or text2 in text1:
        return 0.9
    
    # Try semantic similarity with sentence transformer
    if sentence_model:
        try:
            embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
            embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            return float(max(0.0, min(similarity, 1.0)))
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
    
    # Fallback to Jaccard similarity (word overlap)
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def parse_date(date_value: Any) -> Optional[date]:
    """
    Parse various date formats into a date object.
    
    Args:
        date_value: Date in string, date, or datetime format
        
    Returns:
        Parsed date or None if parsing fails
    """
    if date_value is None:
        return None
        
    if isinstance(date_value, date):
        return date_value
        
    if isinstance(date_value, str):
        if date_value.lower() == "present":
            return date.today()
            
        try:
            return datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError:
            pass
            
        try:
            return datetime.strptime(date_value, "%d-%m-%Y").date()
        except ValueError:
            pass
            
    return None


def calculate_education_score(university_rank: Optional[int]) -> float:
    """
    Calculate education component score based on university ranking.
    
    Args:
        university_rank: Rank value (1-4)
        
    Returns:
        Normalized education score between 0 and 1
    """
    # Default to 1 if no rank is provided
    if university_rank is None:
        university_rank = 1
        
    # Ensure value is within range
    rank = max(1, min(4, university_rank))
    
    # Normalize to 0-1 scale
    return rank / 4.0


def calculate_experience_score(
    start_date: Any, 
    end_date: Any, 
    position_relevance: float,
    required_years: Optional[int]
) -> float:
    """
    Calculate experience component score based on duration and relevance.
    
    Args:
        start_date: Experience start date
        end_date: Experience end date or "Present"
        position_relevance: Relevance score (0-1) of position to search term
        required_years: Required years of experience (optional)
        
    Returns:
        Experience score between 0 and 1
    """
    # Parse dates
    parsed_start = parse_date(start_date)
    parsed_end = parse_date(end_date) or date.today()
    
    if not parsed_start:
        return 0.0
    
    # Calculate years of experience, capped at MAX_EXPERIENCE_YEARS
    years_of_experience = (parsed_end - parsed_start).days / 365.25
    capped_years = min(years_of_experience, MAX_EXPERIENCE_YEARS)
    
    # Only count experience if position is relevant enough
    if position_relevance < RELEVANCE_THRESHOLD:
        return 0.0
    
    # Normalize to 0-1 scale
    experience_score = capped_years / MAX_EXPERIENCE_YEARS
    
    # Add bonus if meeting specific experience requirements
    if required_years and years_of_experience >= required_years:
        experience_score = min(experience_score + 0.2, 1.0)
    
    return experience_score


def calculate_score(profile: Dict[str, Any], search_term: str) -> float:
    """
    Calculate candidate score based on education, experience, and industry relevance.
    
    Args:
        profile: Candidate profile with education and experience data
        search_term: Search query
        
    Returns:
        Score from 0 to 10
    """
    try:
        # Extract search criteria
        position_title, industry, required_years = extract_search_criteria(search_term)
        
        # Calculate education score (0-1)
        education_score = calculate_education_score(profile.get("university_rank"))
        
        # Calculate position similarity (0-1)
        position_similarity = calculate_text_similarity(
            profile.get("position", ""), 
            position_title
        )
        
        # Calculate experience score (0-1)
        experience_score = calculate_experience_score(
            profile.get("start_date"),
            profile.get("end_date"),
            position_similarity,
            required_years
        )
        
        # Calculate industry match score (0-1)
        industry_score = 0.0
        if industry and profile.get("company_industry"):
            candidate_industry = profile.get("company_industry", "").lower()
            search_industry = industry.lower()
            
            # Calculate industry similarity
            industry_score = calculate_text_similarity(candidate_industry, search_industry)
        
        # Calculate weighted total score
        weighted_score = (
            EDUCATION_WEIGHT * education_score +
            POSITION_WEIGHT * position_similarity +
            EXPERIENCE_WEIGHT * experience_score +
            INDUSTRY_WEIGHT * industry_score
        )
        
        # Scale to 0-10 range
        final_score = weighted_score * 10.0
        
        logger.debug(
            f"Score components: education={education_score:.2f}, "
            f"position={position_similarity:.2f}, "
            f"experience={experience_score:.2f}, "
            f"industry={industry_score:.2f}, "
            f"final={final_score:.2f}"
        )
        
        return float(final_score)
        
    except Exception as e:
        logger.error(f"Scoring calculation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0.0