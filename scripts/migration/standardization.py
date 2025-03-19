import re
from rapidfuzz import process, fuzz
import logging
from src.utils.logger import setup_logging

logger = setup_logging()

# University classification data
TOP_MONGOLIAN_UNIS = {
    "шутис": "шинжлэх ухаан технологийн их сургууль",
    "муис": "монгол улсын их сургууль",
    "монгол улсын их сургууль": "монгол улсын их сургууль",
    "хааис": "хөдөө аж ахуйн их сургууль",
    "хааис эзбс": "хөдөө аж ахуйн их сургууль",
    "сэзис": "санхүү эдийн засгийн их сургууль",
    "шмтдс": "шинэ монгол технологийн дээд сургууль",
    "суис": "соёл урлагийн их сургууль",
    "ашиүис": "анагаахын шинжлэх ухааны үндэсний их сургууль",
    "изис": "их засаг их сургууль",
    "мубис": "монгол улсын боловсролын их сургууль",
    "хүмүүнлэгийн ухааны их сургууль": "хүмүүнлэгийн ухааны их сургууль",
    "мандах-бүртгэл дээд сургууль": "мандах-бүртгэл дээд сургууль",
    "монголын үндэсний их сургууль": "монголын үндэсний их сургууль",
    "олон улсын улаанбаатар их сургууль": "олон улсын улаанбаатар их сургууль",
    "батлан хамгаалахын их сургууль": "батлан хамгаалахын их сургууль",
    "отгонтэнгэр их сургууль": "отгонтэнгэр их сургууль"
}

TOP_MONGOLIAN_UNIS_EN = [
    "national university of mongolia",
    "mongolian university of science and technology",
    "mongolian university of life sciences",
    "mongolian university of economics and business", 
    "university of the humanities",
    "university of humanity",
    "national technical university"
]

FOREIGN_UNIVERSITY_INDICATORS = [
    "university", "college", "institute", "school of", "технологийн", "academy", 
    "seoul", "beijing", "harvard", "mit", "stanford", "california", "tokyo", "kyoto", 
    "moscow", "berlin", "london", "oxford", "cambridge", "paris", "new york", "hong kong",
    "yale", "princeton", "columbia", "northwestern", "chicago", "duke", "johns hopkins",
    "toronto", "mcgill", "alberta", "montreal", "ubc", "melbourne", "sydney", "auckland",
    "tsinghua", "peking", "fudan", "zhejiang", "shanghai", "nanjing", "korea", "seoul",
    "yonsei", "kaist", "sungkyunkwan", "hanyang", "waseda", "keio", "hokkaido", "nagoya",
    "tsukuba", "tohoku", "osaka", "kyoto", "copenhagen", "aarhus", "helsinki", "stockholm",
    "uppsala", "lund", "oslo", "bergen"
]

# Common position mappings
POSITION_MAPPINGS = {
    # Data roles
    "data engner": "дата инженер",
    "data enginee": "дата инженер",
    "өгөгдлийн инженер": "дата инженер",
    "дата энженер": "дата инженер",
    "big data engineer": "дата инженер",
    "data analyst": "дата аналист",
    "өгөгдлийн аналист": "дата аналист",
    "дата анализ": "дата аналист",
    "data scientist": "дата сайнтист",
    "өгөгдлийн эрдэмтэн": "дата сайнтист",
    
    # Finance roles
    "finance manager": "санхүүгийн менежер",
    "financial manager": "санхүүгийн менежер",
    "санхүүгийн удирдлага": "санхүүгийн менежер",
    "finance specialist": "санхүүгийн мэргэжилтэн",
    "financial specialist": "санхүүгийн мэргэжилтэн",
    "accountant": "нягтлан бодогч",
    "бүртгэлийн нягтлан": "нягтлан бодогч",
    "нягтлан": "нягтлан бодогч",
    
    # Management roles
    "manager": "менежер",
    "менежир": "менежер",
    "удирдлага": "менежер",
    "executive": "гүйцэтгэх удирдлага",
    "director": "захирал",
    "захирагч": "захирал",
    "lead": "ахлах",
    "ахлагч": "ахлах",
    
    # IT roles
    "developer": "хөгжүүлэгч",
    "программист": "хөгжүүлэгч",
    "software engineer": "програм хангамжийн инженер",
    "программ хангамжийн инженер": "програм хангамжийн инженер",
    "програм ханг инженер": "програм хангамжийн инженер",
    "sysadmin": "системийн администратор",
    "system administrator": "системийн администратор",
    "системийн админ": "системийн администратор",
    
    # HR roles
    "hr manager": "хүний нөөцийн менежер",
    "хүний нөөц менежер": "хүний нөөцийн менежер",
    "human resource manager": "хүний нөөцийн менежер",
    "hr specialist": "хүний нөөцийн мэргэжилтэн",
    "human resource specialist": "хүний нөөцийн мэргэжилтэн",
    "хүний нөөц мэргэжилтэн": "хүний нөөцийн мэргэжилтэн"
}


def standardize_university(school_name, cache, clustering_threshold=85):
    """Standardize university name using tiered approach.
    
    Args:
        school_name: Original university name
        cache: StandardizationCache instance
        clustering_threshold: Threshold for fuzzy matching
        
    Returns:
        tuple: (standardized_name, university_rank)
            1 - Foreign university (highest rank)
            2 - Top Mongolian university
            3 - Other Mongolian university
            4 - High school or invalid (lowest rank)
    """
    if not school_name or not isinstance(school_name, str):
        return "unknown", 4  # Group 4: Invalid/null
    
    # Clean and normalize
    name = school_name.lower().strip()
    
    # Check cache first
    cached = cache.get_university(name)
    if cached:
        return cached[0], cached[1]
    
    # Check for non-meaningful text
    if len(name) < 3 or not re.search(r'[a-zA-Z\u0400-\u04FF\u1800-\u18AF]', name):
        cache.set_university(name, "unknown", 4)
        return "unknown", 4  # Group 4: Too short or no alphabetic characters
    
    # TIER 1: Direct mapping for known Mongolian universities
    if name in TOP_MONGOLIAN_UNIS:
        standardized = TOP_MONGOLIAN_UNIS[name]
        cache.set_university(name, standardized, 2)
        return standardized, 2  # Group 2: Top Mongolian university
    
    # Check for foreign universities (Group 1)
    if any(indicator in name for indicator in FOREIGN_UNIVERSITY_INDICATORS) and \
       re.search(r'[a-zA-Z]', name):
        # Keep original case for foreign universities
        cache.set_university(name, school_name, 1)
        return school_name, 1  # Group 1: Foreign university
        
    # TIER 2: Fuzzy matching for Mongolian universities
    # First try abbreviations
    abbrev_matches = process.extractOne(
        name, 
        list(TOP_MONGOLIAN_UNIS.keys()), 
        scorer=fuzz.token_set_ratio
    )
    
    if abbrev_matches and abbrev_matches[1] >= clustering_threshold:
        matched_abbr = abbrev_matches[0]
        standardized = TOP_MONGOLIAN_UNIS[matched_abbr]
        cache.set_university(name, standardized, 2)
        return standardized, 2
    
    # Try full names
    fullname_matches = process.extractOne(
        name, 
        list(TOP_MONGOLIAN_UNIS.values()) + TOP_MONGOLIAN_UNIS_EN, 
        scorer=fuzz.token_set_ratio
    )
    
    if fullname_matches and fullname_matches[1] >= clustering_threshold:
        matched_name = fullname_matches[0]
        if matched_name in TOP_MONGOLIAN_UNIS_EN:
            # Convert to Mongolian name if possible
            for key, value in TOP_MONGOLIAN_UNIS.items():
                if value.lower() == matched_name:
                    matched_name = value
                    break
        cache.set_university(name, matched_name, 2)
        return matched_name, 2
    
    # TIER 3: Check against clustering results
    if name in cache.university_clusters:
        canonical = cache.university_clusters[name]
        # Determine the rank based on the canonical form
        if canonical in TOP_MONGOLIAN_UNIS.values():
            cache.set_university(name, canonical, 2)
            return canonical, 2
        elif "их сургууль" in canonical or "дээд сургууль" in canonical or "коллеж" in canonical:
            cache.set_university(name, canonical, 3)
            return canonical, 3
    
    # Final categorization based on keywords
    if "их сургууль" in name or "дээд сургууль" in name or "коллеж" in name:
        cache.set_university(name, name, 3)
        return name, 3  # Group 3: Other Mongolian university
    
    # If nothing matched but has Mongolian characters, assume other Mongolian university
    if re.search(r'[\u1800-\u18AF]', name):
        cache.set_university(name, name, 3)
        return name, 3  # Group 3: Other Mongolian university
    
    # Default case - high school or unknown
    cache.set_university(name, name, 4)
    return name, 4  # Group 4: High school or unknown


def standardize_position(position, cache, clustering_threshold=85):
    """Standardize job position using tiered approach.
    
    Args:
        position: Original position name
        cache: StandardizationCache instance
        clustering_threshold: Threshold for fuzzy matching
    """
    if not position or not isinstance(position, str):
        return position
    
    # Clean and normalize
    clean_pos = position.lower().strip()
    
    # Check cache first
    cached = cache.get_position(clean_pos)
    if cached:
        return cached
    
    # TIER 1: Direct mapping
    # Check if position matches any of our mappings
    for key, value in POSITION_MAPPINGS.items():
        if clean_pos == key or clean_pos.startswith(key + " ") or clean_pos.endswith(" " + key):
            cache.set_position(clean_pos, value)
            return value
    
    # TIER 2: Fuzzy matching
    matches = process.extractOne(
        clean_pos,
        list(POSITION_MAPPINGS.values()) + list(POSITION_MAPPINGS.keys()),
        scorer=fuzz.token_set_ratio
    )
    
    if matches and matches[1] >= clustering_threshold:
        matched_pos = matches[0]
        # If matched a key, get its value
        if matched_pos in POSITION_MAPPINGS:
            standardized = POSITION_MAPPINGS[matched_pos]
        else:
            standardized = matched_pos
        cache.set_position(clean_pos, standardized)
        return standardized
    
    # TIER 3: Check against clustering results
    if clean_pos in cache.position_clusters:
        canonical = cache.position_clusters[clean_pos]
        cache.set_position(clean_pos, canonical)
        return canonical
    
    # If no matches, return the original
    cache.set_position(clean_pos, position)
    return position