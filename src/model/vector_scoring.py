from datetime import datetime, date
import logging
import re
import numpy as np
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config
from functools import lru_cache

logger = logging.getLogger(__name__)

# Load models once at module level for efficiency
client = None
sentence_model = None

# Extraction cache to avoid repeated API calls
extraction_cache = {}

try:
    # Initialize OpenAI client
    config = load_config()
    client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
    
    # Sentence transformer for generating vectors
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    logger.info("NLP models loaded successfully")
    logger.info(f"OpenAI client initialized: {client is not None}")
except Exception as e:
    logger.error(f"Error loading NLP models: {str(e)}")
    client = None
    sentence_model = None

def extract_from_search_term(search_term):
    """
    Extract job title, industry and experience years from search term using
    OpenAI API or fallback to regex patterns. Results are cached.
    
    Args:
        search_term (str): The search query
        
    Returns:
        tuple: (job_title, industry, experience_years)
    """
    # Check cache first
    if search_term in extraction_cache:
        logger.info(f"Using cached extraction for: '{search_term}'")
        return extraction_cache[search_term]
    
    # Default values
    job_title = search_term
    industry = None
    experience_years = None
    
    logger.info(f"Processing search term: '{search_term}'")
    logger.info(f"OpenAI client available: {client is not None}")
    
    # Try using OpenAI if available
    if client:
        try:
            logger.info(f"Attempting OpenAI extraction for: '{search_term}'")
            
            prompt = f"""
            Extract the following information from this Mongolian job search query:
            1. Position/job title
            2. Industry (if mentioned)
            3. Years of experience (if mentioned)
            
            Format your response as JSON with these keys: position, industry, years
            
            Query: {search_term}
            """
            
            logger.info("Sending query to OpenAI for extraction")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You extract structured information from Mongolian job search queries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            extracted_json = response.choices[0].message.content
            logger.info(f"OpenAI response content: {extracted_json}")
            
            extracted = json.loads(extracted_json)
            
            if "position" in extracted and extracted["position"]:
                job_title = extracted["position"]
                # Normalize case - capitalize first letter
                job_title = job_title.capitalize()
            
            if "industry" in extracted and extracted["industry"]:
                industry = extracted["industry"]
                # Normalize case for industry
                industry = industry.capitalize()
            
            if "years" in extracted and extracted["years"]:
                # Try to extract a number
                if isinstance(extracted["years"], (int, float)):
                    experience_years = int(extracted["years"])
                else:
                    # Extract number from text like "5+" or "5 years"
                    years_match = re.search(r'(\d+)', str(extracted["years"]))
                    if years_match:
                        experience_years = int(years_match.group(1))
            
            logger.info(f"OpenAI extracted: job_title='{job_title}', industry='{industry}', experience_years={experience_years}")
            
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {str(e)}. Falling back to regex.")
    else:
        logger.info("OpenAI client not available, using regex patterns")
    
    # Fallback to regex patterns if OpenAI failed or is unavailable
    if not industry or not experience_years:
        logger.info("Using regex patterns for missing information")
        
        # Common patterns for extracting experience years
        years_pattern = r'(\d+)(?:\+)?\s*(?:жил|year|жилийн|түүнээс дээш жилийн|жилээс дээш|аас дээш)'
        years_match = re.search(years_pattern, search_term, re.IGNORECASE)
        if years_match:
            experience_years = int(years_match.group(1))
            logger.info(f"Regex extracted experience years: {experience_years}")
        
        # Extract industry based on common Mongolian industry terms
        industry_patterns = {
            'Санхүү/Бүртгэл': r'санхүү|санхүүгийн|санхүүгийн салбар|бүртгэл|нягтлан|нягтлан бодох бүртгэл|аудит|татвар|finance|financial|accounting|audit|tax',
            'IT/Мэдээллийн технологи/Програм хангамж': r'it|айти|информацийн технологи|мэдээллийн технологи|программ|програм хангамж|код|программист|developer|software|hardware|network|програмчлал|вэб хөгжүүлэлт|кибер аюулгүй байдал|систем админ',
            'Маркетинг': r'маркетинг|маркетингийн|зар сурталчилгаа|борлуулалт|сурталчилгаа|реклам|брэндинг|брэнд|борлуулалтын стратеги|marketing|advertising|branding|sales',
            'Хууль, эрх зүй/Нийцэл': r'хууль|эрх зүй|хуульч|өмгөөлөгч|хуульчлах|эрх зүйн зөвлөх|нотариат|нийцэл|комплаенс|compliance|legal|law|attorney|lawyer|jurisprudence|notary',
            'Боловсрол': r'боловсрол|сургалт|багш|сурган хүмүүжүүлэгч|их сургууль|сургууль|коллеж|эрдэм шинжилгээ|академик|education|training|teaching|research|university|school',
            'Эрүүл мэнд': r'эрүүл мэнд|эмнэлэг|анагаах|анагаах ухаан|эмч|сувилагч|эмчилгээ|хоол судлал|фитнес|эрүүл амьдрал|health|medical|medicine|doctor|nurse|treatment|nutrition|fitness',
            'Инженер техник/Инженеринг': r'инженер|инженерийн|техник|механик|цахилгаан|автоматжуулалт|барилга|дэд бүтэц|сантехник|engineering|mechanical|electrical|automation|construction|civil engineering',
            'Хөдөлмөрийн аюулгүй байдал, эрүүл ахуй, байгаль орчин': r'хөдөлмөрийн аюулгүй байдал|аюулгүй байдал|эрүүл ахуй|байгаль орчин|экологи|хог хаягдал|цахилгааны аюулгүй байдал|хөдөлмөрийн эрүүл мэнд|safety|health and safety|environment|hse|ecology|waste management',
            'Үйлдвэрлэл': r'үйлдвэрлэл|үйлдвэр|үйлдвэрлэлийн шугам|үйлдвэрлэлийн техник|машин механизм|борлуулалт|production|manufacturing|factory|industrial',
            'Үйлчилгээ': r'үйлчилгээ|үйлчилгээний байгууллага|үйлчилгээний ажилтан|харилцагчийн үйлчилгээ|зочид буудал|хэрэглэгчийн үйлчилгээ|сувилахуйн үйлчилгээ|service|customer service|hospitality|client service',
            'Хүний нөөц/захиргаа': r'хүний нөөц|хүний нөөцийн менежмент|цалин хөлс|цалин|нөөцийн төлөвлөлт|хөдөлмөрийн харилцаа|захиргаа|бичиг баримт|hr|human resources|administration|payroll|workforce planning|employee relations',
            'Бусад': r'бусад|өөр|ерөнхий|ангилагдаагүй|miscellaneous|other',
            'Удирдлага': r'удирдлага|менежмент|дарга|захирал|менежер|тэргүүлэгч|ажил хэргийн удирдлага|бизнесийн стратеги|management|leadership|executive|director|strategy',
            'Худалдан авалт': r'худалдан авалт|худалдан авах|тендер|нийлүүлэгч|үнэлгээ|договораар худалдан авах|procurement|purchasing|tender|supplier|evaluation',
            'Тээвэр ложистик': r'тээвэр|ачаа тээвэр|тээврийн үйлчилгээ|ачааны машин|ложистик|агуулах|бараа хүргэлт|transport|logistics|warehouse|shipping|cargo',
            'Олон нийтийн харилцаа': r'олон нийтийн харилцаа|хэвлэл мэдээлэл|хэвлэлийн төлөөлөгч|пиар|нийтийн харилцаа|нийгмийн харилцаа|public relations|pr|media relations|communication',
            'Контент медиа/Дизайн': r'контент|медиа|дизайн|график дизайн|видео продакшн|анимаци|зураглал|ux|ui|контент бүтээгч|content|media|design|graphic design|animation|illustration|ux/ui',
            'Хөдөө аж ахуй': r'хөдөө аж ахуй|газар тариалан|мал аж ахуй|фермер|хүнс|таримал|хөдөө аж ахуйн технологи|agriculture|farming|livestock|agricultural technology'
        }
        
        for ind_name, pattern in industry_patterns.items():
            if re.search(pattern, search_term, re.IGNORECASE):
                industry = ind_name
                logger.info(f"Regex extracted industry: {industry}")
                break
    
    # Extract job title if not found by OpenAI
    # Common job titles in Mongolian
    job_patterns = {
        'Санхүүгийн менежер': r'санхүүгийн\s+менежер|finance\s+manager',
        'Нягтлан бодогч': r'нягтлан\s+бодогч|нягтлан|accountant',
        'Аудитор': r'аудитор|auditor',
        'Татварын мэргэжилтэн': r'татварын\s+мэргэжилтэн|татварын\s+зөвлөх|tax\s+specialist|tax\s+consultant',
        'Банкны теллер': r'банкны\s+теллер|теллер|bank\s+teller',
        
        'Дата аналист': r'дата\s+аналист|өгөгдлийн\s+аналист|data\s+analyst',
        'Мэдээллийн технологийн инженер': r'мэдээллийн\s+технологийн\s+инженер|айти\s+инженер|it\s+engineer',
        'Программ хөгжүүлэгч': r'программ\s+хөгжүүлэгч|програм\s+хөгжүүлэгч|developer|программист',
        'Системийн админ': r'системийн\s+админ|сүлжээний\s+админ|system\s+administrator|network\s+admin',
        'Кибер аюулгүй байдлын шинжээч': r'кибер\s+аюулгүй\s+байдлын\s+шинжээч|cyber\s+security\s+specialist',
        
        'Борлуулагч': r'борлуулагч|худалдагч|sales|seller',
        'Маркетингийн менежер': r'маркетингийн\s+менежер|маркетинг\s+удирдагч|marketing\s+manager',
        'Брэнд менежер': r'брэнд\s+менежер|brand\s+manager',
        'Зар сурталчилгааны мэргэжилтэн': r'зар\s+сурталчилгаа|реклам|advertising\s+specialist',
        
        'Хуульч': r'хуульч|өмгөөлөгч|lawyer|attorney',
        'Хууль эрх зүйн зөвлөх': r'хууль\s+эрх\s+зүйн\s+зөвлөх|legal\s+consultant',
        'Нотариат': r'нотариат|notary',
        
        'Багш': r'багш|сурган\s+хүмүүжүүлэгч|teacher',
        'Их сургуулийн багш': r'их\s+сургуулийн\s+багш|профессор|university\s+lecturer|professor',
        'Сургалтын менежер': r'сургалтын\s+менежер|training\s+manager',
        
        'Эмч': r'эмч|анагаахын\s+мэргэжилтэн|doctor',
        'Сувилагч': r'сувилагч|nurse',
        'Эм зүйч': r'эм\s+зүйч|фармацевт|pharmacist',
        'Шүдний эмч': r'шүдний\s+эмч|dentist',
        
        'Барилгын инженер': r'барилгын\s+инженер|civil\s+engineer',
        'Цахилгааны инженер': r'цахилгааны\s+инженер|electrical\s+engineer',
        'Механик инженер': r'механик\s+инженер|mechanical\s+engineer',
        'Автоматжуулалтын инженер': r'автоматжуулалтын\s+инженер|automation\s+engineer',
        
        'Хөдөлмөрийн аюулгүй байдлын мэргэжилтэн': r'хөдөлмөрийн\s+аюулгүй\s+байдлын\s+мэргэжилтэн|hse\s+specialist|safety\s+officer',
        
        'Үйлдвэрийн оператор': r'үйлдвэрийн\s+оператор|factory\s+operator',
        'Тоног төхөөрөмжийн инженер': r'тоног\s+төхөөрөмжийн\s+инженер|equipment\s+engineer',
        
        'Зочид буудлын менежер': r'зочид\s+буудлын\s+менежер|hotel\s+manager',
        'Үйлчилгээний ажилтан': r'үйлчилгээний\s+ажилтан|customer\s+service\s+representative',
        
        'Хүний нөөцийн менежер': r'хүний\s+нөөцийн\s+менежер|hr\s+manager',
        'Цалин хөлсний мэргэжилтэн': r'цалин\s+хөлсний\s+мэргэжилтэн|payroll\s+specialist',
        
        'Гүйцэтгэх захирал': r'гүйцэтгэх\s+захирал|ceo|executive\s+director',
        'Төслийн менежер': r'төслийн\s+менежер|project\s+manager',
        
        'Худалдан авалтын менежер': r'худалдан\s+авалтын\s+менежер|procurement\s+manager',
        'Тээврийн менежер': r'тээврийн\s+менежер|logistics\s+manager|transport\s+manager',
        
        'Контент бүтээгч': r'контент\s+бүтээгч|content\s+creator',
        'График дизайнер': r'график\s+дизайнер|graphic\s+designer',
        'UX/UI дизайнер': r'ux/ui\s+дизайнер|ux\s+designer|ui\s+designer',
        
        'Фермер': r'фермер|газар\s+тариалангийн\s+мэргэжилтэн|farmer',
        'Малын эмч': r'малын\s+эмч|veterinarian'
}
    
    for title, pattern in job_patterns.items():
        if re.search(pattern, search_term, re.IGNORECASE):
            job_title = title
            logger.info(f"Regex extracted job title: {job_title}")
            break
    
    logger.info(f"Final extracted values: job_title='{job_title}', industry='{industry}', experience_years={experience_years}")
    
    # Cache the results for future use
    result = (job_title, industry, experience_years)
    extraction_cache[search_term] = result
    
    return result

@lru_cache(maxsize=100)
def generate_search_embedding(text):
    """
    Generate vector embedding for search text.
    Results are cached using lru_cache.
    
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

def calculate_vector_score(profile, job_title, industry, experience_years):
    """
    Calculate candidate score using position matching, experience, and other factors.
    
    Args:
        profile (dict): Candidate profile including position_similarity_score
        job_title (str): Extracted job title
        industry (str): Extracted industry
        experience_years (int): Extracted years of experience
        
    Returns:
        float: Score from 0 to 10, or 0 if required experience not met
    """
    try:
        # First check if the candidate meets minimum experience requirement
        if experience_years:
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
            
            # Calculate total years of experience
            if start_date and isinstance(start_date, date):
                years = (end_date - start_date).days / 365.25
                
                # Check if meets minimum experience requirement
                if years < experience_years:
                    # Candidate doesn't meet minimum experience requirement
                    logger.debug(f"Candidate has {years:.1f} years experience, but {experience_years} required. Returning score 0.")
                    return 0
                else:
                    logger.debug(f"Candidate meets minimum {experience_years} years experience with {years:.1f} years.")
        
        # School ranking (x): 1-4, from stored university_rank
        # Invert the ranking so that rank 1 (foreign universities) gets the highest score
        university_rank = profile.get("university_rank", 4)  # Default to 4 (lowest) if not provided
        if university_rank is None:
            university_rank = 4  # Default to lowest
        
        # Normalize university rank - invert so rank 1 (foreign universities) gets 1.0
        x = (5 - university_rank) / 4  # This transforms: 1->1.0, 2->0.75, 3->0.5, 4->0.25
        
        # Position similarity score - should already be between 0-1 from combined text/vector match
        vs = profile.get("position_similarity_score", 0)
        
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
            
            # FIX: Improved industry matching with better logging
            # Log the industry values we're comparing to help debug
            logger.debug(f"Comparing industries: search='{search_industry}' vs candidate='{candidate_industry}'")
            
            # Check for exact or partial industry match
            if search_industry in candidate_industry or candidate_industry in search_industry:
                im = 1.0
                logger.debug(f"Exact industry match: '{search_industry}' and '{candidate_industry}'")
            elif any(word in candidate_industry for word in search_industry.split() if len(word) > 3):
                # Only match on words longer than 3 chars to avoid matching on common words
                im = 0.5
                logger.debug(f"Partial industry match: '{search_industry}' and '{candidate_industry}'")
            else:
                logger.debug(f"No industry match between '{search_industry}' and '{candidate_industry}'")
        else:
            # Log when we don't have industry information to match
            if not industry:
                logger.debug("No industry specified in search term")
            if not profile.get("company_industry"):
                logger.debug("No company_industry in candidate profile")
        
        # Combine factors with appropriate weights
        # Position: 45%, Experience: 35%, Industry: 15%, University: 5%
        total = (0.05 * x) + (0.45 * vs) + (0.35 * ed) + (0.15 * im)
        
        # Scale to 0-10
        scaled_score = total * 10
        
        logger.debug(f"Vector score: {scaled_score:.2f} (x={x:.2f}, vs={vs:.2f}, ed={ed:.2f}, im={im:.2f})")
        return scaled_score
    
    except Exception as e:
        logger.error(f"Vector scoring failed: {str(e)}")
        return 0