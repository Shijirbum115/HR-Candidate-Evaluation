from openai import OpenAI
from src.utils.config import load_config
from rapidfuzz import process, fuzz
import re
import logging

logger = logging.getLogger(__name__)
client = OpenAI(api_key=load_config()["OPENAI_API_KEY"])

# University mappings
university_mappings = {
    "шутис": "шинжлэх ухаан технологийн их сургууль",
    "хааис эзбс": "хөдөө аж ахуйн их сургууль",
    "хааис": "хөдөө аж ахуйн их сургууль",
    "муис": "монгол улсын их сургууль",
    "сэзис": "санхүү эдийн засгийн их сургууль",
    "шмтдс": "шинэ монгол технологийн дээд сургууль",
    "суис": "соёл урлагийн их сургууль",
    "ашиүис": "анагаахын шинжлэх ухааны үндэсний их сургууль",
    "изис": "их засаг их сургууль",
    "мубис": "монгол улсын боловсролын их сургууль"
}

# University classification lists
top_mongolia_universities = [
    "монгол улсын их сургууль",
    "шинжлэх ухаан технологийн их сургууль",
    "санхүү эдийн засгийн их сургууль",
    "хөдөө аж ахуйн их сургууль",
    "анагаахын шинжлэх ухааны үндэсний их сургууль",
    "хүмүүнлэгийн ухааны их сургууль",
    "их засаг их сургууль",
    "монгол улсын боловсролын их сургууль",
    "мандах-бүртгэл дээд сургууль",
    "монголын үндэсний их сургууль",
    "олон улсын улаанбаатар их сургууль",
    "батлан хамгаалахын их сургууль",
    "отгонтэнгэр их сургууль",
    "шинэ монгол технологийн дээд сургууль"
]

english_top_mongolia_universities = [
    "national university of mongolia",
    "mongolian university of science and technology",
    "mongolian university of life sciences",
    "mongolian university of economics and business",
    "university of the humanities",
    "university of humanity",
    "national technical university"
]

other_mongolia_universities = ["их сургууль", "дээд сургууль", "коллеж"]

def clean_university_name(name):
    """Clean and standardize university names with hybrid approach."""
    if not isinstance(name, str) or not name.strip():
        return "unknown", 1  # Score 1 for null/meaningless
    
    name = name.lower().strip()
    
    # Manual: Direct mapping
    if name in university_mappings:
        return university_mappings[name], 3  # Top Mongolian
    
    # Manual: Fuzzy match against mappings and top universities
    choices = list(university_mappings.keys()) + top_mongolia_universities
    match, score, _ = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if score > 80:
        return university_mappings.get(match, match), 3  # Top Mongolian
    
    # Manual: Abbreviations with extra text
    for abbr, full_name in university_mappings.items():
        if abbr in name:
            extra = name.replace(abbr, "").strip()
            return f"{full_name} {extra}".strip() if extra else full_name, 3
    
    # AI fallback for foreign or meaningless cases
    prompt = f"Standardize this school name or classify as foreign/meaningless:\n{name}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    cleaned = response.choices[0].message.content.strip().lower()
    
    # Classify AI output
    if "meaningless" in cleaned or not cleaned:
        return cleaned.title() or "unknown", 1  # High school/null
    if any(c.isalpha() and ord(c) < 128 for c in cleaned) or re.search(r'[\u4e00-\u9faf|\uAC00-\uD7AF]', cleaned):
        return cleaned.title(), 4  # Foreign
    if any(term in cleaned for term in other_mongolia_universities):
        return cleaned.title(), 2  # Other Mongolian
    return cleaned, 3  # Default to Top Mongolian if AI is confident

def preprocess_school(name):
    """Preprocess school name and return standardized name and score."""
    cleaned_name, score = clean_university_name(name)
    return cleaned_name, score  # Returns name and x score (1-4)