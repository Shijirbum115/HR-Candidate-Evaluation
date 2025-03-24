from datetime import datetime, date
import logging
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Load models once at module level for efficiency
try:
    # Initialize NER pipeline for Mongolian language
    # Replace "monsoon-nlp/mongolian-bert-ner" with the appropriate model for your needs
    tokenizer = AutoTokenizer.from_pretrained("monsoon-nlp/mongolian-bert-ner")
    model = AutoModelForTokenClassification.from_pretrained("monsoon-nlp/mongolian-bert-ner")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    
    # Sentence transformer for similarity if needed (for fallback or validation)
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


def calculate_position_similarity(candidate_position, search_position):
    """
    Calculate similarity between candidate position and search position.
    Uses cosine similarity of text embeddings if models are available,
    otherwise falls back to string matching.
    
    Args:
        candidate_position (str): Candidate's position
        search_position (str): Position from search query
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not candidate_position or not search_position:
        return 0
    
    # Convert to lowercase for better matching
    candidate_position = candidate_position.lower()
    search_position = search_position.lower()
    
    # Exact match gets highest score
    if candidate_position == search_position:
        return 1.0
    
    # Check if search position is contained in candidate position
    if search_position in candidate_position or candidate_position in search_position:
        return 0.9
    
    # Use sentence transformers for semantic similarity if available
    if sentence_model:
        try:
            emb1 = sentence_model.encode(candidate_position, convert_to_tensor=True)
            emb2 = sentence_model.encode(search_position, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            return max(0, min(similarity, 1.0))  # Ensure value is between 0 and 1
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
    
    # Fallback to basic substring matching
    words1 = set(candidate_position.split())
    words2 = set(search_position.split())
    
    # Calculate Jaccard similarity between word sets
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0


def calculate_score(profile, search_term):
    """
    Calculate candidate score based on school, experience, and industry match.
    
    Args:
        profile (dict): Candidate profile with education and experience data
        search_term (str): Search query
        
    Returns:
        float: Score from 0 to 10
    """
    try:
        # Extract components from search term
        job_title, industry, experience_years = extract_from_search_term(search_term)
        logger.debug(f"Extracted: job={job_title}, industry={industry}, years={experience_years}")
        
        # School (x): 1-4 from stored university_rank, default to 1 if None
        # Higher rank means better university
        x = profile.get("university_rank", 1) if profile.get("university_rank") is not None else 1
        
        # Experience (y): Duration * Relevance, max 6
        start_date = profile.get("start_date")
        end_date = profile.get("end_date")
        
        if end_date is None or end_date == "Present":
            end_date = date.today()
        elif isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                end_date = date.today()
                
        if start_date and isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start_date = None
        
        y = 0
        if start_date and isinstance(start_date, date):
            # Calculate experience duration in years, cap at 6 years
            years = min((end_date - start_date).days / 365.25, 6)
            
            # Position relevance score (0-1)
            relevance = calculate_position_similarity(profile.get("position", ""), job_title)
            
            # Experience score = years * relevance (weighted)
            # We only count experience if there's some relevance
            if relevance > 0.6:  # Relevance threshold
                y = years * relevance
                
                # If the candidate exceeds requested experience (if specified)
                if experience_years and years >= experience_years:
                    y += 0.5  # Bonus for meeting specific experience requirement
            
            y = min(y, 6)  # Cap at maximum 6 points
        
        # Industry (z): 0-2, based on industry matching
        z = 0
        if industry and profile.get("company_industry"):
            candidate_industry = profile.get("company_industry", "").lower()
            search_industry = industry.lower()
            
            # Exact industry match
            if search_industry in candidate_industry or candidate_industry in search_industry:
                z = 2
            # Partial industry match
            elif any(word in candidate_industry for word in search_industry.split()):
                z = 1
        
        # Total: Normalize to 0-10 (max raw score = 12: 4 + 6 + 2)
        total = (x + y + z) / 12 * 10
        
        logger.debug(f"Score: {total:.2f} (x={x}, y={y}, z={z})")
        return total
    
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return 0