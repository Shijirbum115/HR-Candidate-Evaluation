from datetime import datetime, date
import logging
import re
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Load models once at module level for efficiency
try:
    # Initialize NER pipeline for Mongolian language
    # Replace with appropriate model for your needs
    ner_pipeline = pipeline("ner", model="monsoon-nlp/mongolian-bert-ner")
    
    # Sentence transformer for generating vectors
    # Use a multilingual model that supports Mongolian
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    logger.info("NLP models loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP models: {str(e)}")
    # Fall back to simple regex-based extraction if models fail to load
    ner_pipeline = None
    sentence_model = None

def extract_from_search_term(search_term):
    """
    Extract job title, industry and experience years from search term using
    Hugging Face NER model or fallback to regex patterns.
    
    Args:
        search_term (str): The search query
        
    Returns:
        tuple: (job_title, industry, experience_years)
    """
    # Default values
    job_title = search_term
    industry = None
    experience_years = None
    
    # Try using the NER pipeline if available
    if ner_pipeline:
        try:
            # Get NER results
            ner_results = ner_pipeline(search_term)
            
            # Process NER results to extract entities
            entities = {}
            current_entity = None
            current_text = ""
            
            for item in ner_results:
                if item['entity'].startswith('B-'):  # Beginning of entity
                    if current_entity:  # Save previous entity if exists
                        entities[current_entity] = current_text.strip()
                    current_entity = item['entity'][2:]  # Remove B- prefix
                    current_text = item['word']
                elif item['entity'].startswith('I-'):  # Inside of entity
                    if current_entity == item['entity'][2:]:  # Same entity type
                        current_text += " " + item['word']
            
            # Save the last entity
            if current_entity:
                entities[current_entity] = current_text.strip()
            
            # Map entities to our expected outputs
            # Adjust these mappings based on your specific NER model's output labels
            position_labels = ["POSITION", "JOB", "TITLE", "ROLE"]
            industry_labels = ["INDUSTRY", "SECTOR", "FIELD"]
            experience_labels = ["EXPERIENCE", "YEARS", "DURATION"]
            
            # Extract job title/position
            for label in position_labels:
                if label in entities:
                    job_title = entities[label]
                    break
            
            # Extract industry
            for label in industry_labels:
                if label in entities:
                    industry = entities[label]
                    break
            
            # Extract experience years
            for label in experience_labels:
                if label in entities:
                    # Try to extract number from the text
                    years_match = re.search(r'(\d+)', entities[label])
                    if years_match:
                        experience_years = int(years_match.group(1))
                    break
                    
            logger.debug(f"NER extracted: job_title='{job_title}', industry='{industry}', experience_years={experience_years}")
            
        except Exception as e:
            logger.warning(f"NER extraction failed: {str(e)}. Falling back to regex.")
    
    # Fallback to regex patterns if NER failed or is unavailable
    if not industry or not experience_years:
        # Common patterns for extracting experience years
        years_pattern = r'(\d+)(?:\+)?\s*(?:жил|year|жилийн|түүнээс дээш жилийн|жилээс дээш)'
        years_match = re.search(years_pattern, search_term, re.IGNORECASE)
        if years_match:
            experience_years = int(years_match.group(1))
        
        # Extract industry based on common Mongolian industry terms
        industry_patterns = {
            'Санхүү': r'санхүү|finance|financial',
            'IT': r'it|айти|информацийн технологи|software|программ',
            'Маркетинг': r'маркетинг|marketing|зар сурталчилгаа',
            'Хууль': r'хууль|legal|law|attorney',
            'Боловсрол': r'боловсрол|education|training|сургалт',
            'Эрүүл мэнд': r'эрүүл мэнд|health|medical|эмнэлэг'
        }
        
        for ind_name, pattern in industry_patterns.items():
            if re.search(pattern, search_term, re.IGNORECASE):
                industry = ind_name
                break
    
    # Extract job title if not found by NER
    # Common job titles in Mongolian
    job_patterns = {
        'Санхүүгийн менежер': r'санхүүгийн\s+менежер|finance\s+manager',
        'Нягтлан бодогч': r'нягтлан\s+бодогч|accountant',
        'Дата аналист': r'дата\s+аналист|өгөгдлийн\s+аналист|data\s+analyst',
        'Программ хөгжүүлэгч': r'программ\s+хөгжүүлэгч|програм\s+хөгжүүлэгч|developer|программист'
    }
    
    for title, pattern in job_patterns.items():
        if re.search(pattern, search_term, re.IGNORECASE):
            job_title = title
            break
    
    return job_title, industry, experience_years

def generate_search_embedding(text):
    """
    Generate vector embedding for search text.
    
    Args:
        text (str): Text to encode
        
    Returns:
        numpy.ndarray: Vector embedding or None if failed
    """
    if not text or not sentence_model:
        return None
    
    try:
        embedding = sentence_model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return None

def calculate_vector_score(profile, search_term):
    """
    Calculate candidate score using vector similarity and other factors.
    
    Args:
        profile (dict): Candidate profile including position_similarity_score
        search_term (str): Search query
        
    Returns:
        float: Score from 0 to 10
    """
    try:
        # Extract components from search term
        job_title, industry, experience_years = extract_from_search_term(search_term)
        logger.debug(f"Extracted: job={job_title}, industry={industry}, years={experience_years}")
        
        # School ranking (x): 1-4, from stored university_rank
        x = profile.get("university_rank", 1) if profile.get("university_rank") is not None else 1
        
        # Vector similarity score (vs): use stored position_similarity_score or calculate
        # Assume the position_similarity_score is between 0-1
        vs = profile.get("position_similarity_score", 0)
        
        if vs == 0 and sentence_model and "position" in profile:
            # If similarity not provided but we have the model, calculate it
            search_embedding = generate_search_embedding(job_title)
            if search_embedding is not None and "position_vector" in profile:
                position_vector = profile.get("position_vector")
                # Calculate cosine similarity if we have both vectors
                if position_vector is not None:
                    vs = np.dot(search_embedding, position_vector) / (
                        np.linalg.norm(search_embedding) * np.linalg.norm(position_vector)
                    )
                    vs = max(0, min(vs, 1))  # Ensure value is between 0 and 1
        
        # Experience duration (ed)
        start_date = profile.get("start_date")
        end_date = profile.get("end_date")
        
        # Handle end_date properly
        if end_date is None or end_date == "Present":
            end_date = date.today()
        elif isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                end_date = date.today()
        
        # Parse start_date if it's a string
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start_date = None
        
        ed = 0
        if start_date and isinstance(start_date, date):
            # Calculate years of experience, cap at 5 years
            years = min((end_date - start_date).days / 365.25, 5)
            
            # Normalize to 0-1 scale
            ed = years / 5
            
            # Apply bonus if this meets the required experience
            if experience_years and years >= experience_years:
                ed += 0.2  # Bonus for meeting specific experience requirement
                ed = min(ed, 1.0)  # Cap at 1
        
        # Industry match (im)
        im = 0
        if industry and profile.get("company_industry"):
            candidate_industry = profile.get("company_industry", "").lower()
            search_industry = industry.lower()
            
            # Exact industry match
            if search_industry in candidate_industry or candidate_industry in search_industry:
                im = 1.0
            # Partial industry match
            elif any(word in candidate_industry for word in search_industry.split()):
                im = 0.5
        
        # Combine factors with appropriate weights
        # School ranking: 20%, Vector similarity: 40%, Experience duration: 25%, Industry match: 15%
        total = (0.2 * (x/4)) + (0.4 * vs) + (0.25 * ed) + (0.15 * im)
        
        # Scale to 0-10
        scaled_score = total * 10
        
        logger.debug(f"Vector score: {scaled_score:.2f} (x={x}, vs={vs:.2f}, ed={ed:.2f}, im={im:.2f})")
        return scaled_score
    
    except Exception as e:
        logger.error(f"Vector scoring failed: {str(e)}")
        return 0