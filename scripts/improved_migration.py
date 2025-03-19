import sys
import os
import time
import json
import logging
import argparse
import mysql.connector
import psycopg2
import re
import concurrent.futures
import pickle
import numpy as np
from datetime import datetime, date
from psycopg2.extras import execute_values
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()
logger.setLevel(logging.INFO)

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

class StandardizationCache:
    """Cache for standardized names to avoid redundant processing"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.university_cache = {}
        self.position_cache = {}
        self.university_clusters = {}  # For clustering results
        self.position_clusters = {}
        self.embedding_cache = {}
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.load_cache()
        
    def load_cache(self):
        try:
            university_cache_file = os.path.join(self.cache_dir, "university_cache.json")
            if os.path.exists(university_cache_file):
                with open(university_cache_file, 'r', encoding='utf-8') as f:
                    self.university_cache = json.load(f)
                logger.info(f"Loaded {len(self.university_cache)} cached university standardizations")
                
            position_cache_file = os.path.join(self.cache_dir, "position_cache.json")
            if os.path.exists(position_cache_file):
                with open(position_cache_file, 'r', encoding='utf-8') as f:
                    self.position_cache = json.load(f)
                logger.info(f"Loaded {len(self.position_cache)} cached position standardizations")
                
            university_clusters_file = os.path.join(self.cache_dir, "university_clusters.pkl")
            if os.path.exists(university_clusters_file):
                with open(university_clusters_file, 'rb') as f:
                    self.university_clusters = pickle.load(f)
                logger.info(f"Loaded university clustering data")
                
            position_clusters_file = os.path.join(self.cache_dir, "position_clusters.pkl")
            if os.path.exists(position_clusters_file):
                with open(position_clusters_file, 'rb') as f:
                    self.position_clusters = pickle.load(f)
                logger.info(f"Loaded position clustering data")
                
            embeddings_cache_file = os.path.join(self.cache_dir, "embeddings_cache.pkl")
            if os.path.exists(embeddings_cache_file):
                with open(embeddings_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
                
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            # Initialize empty caches if loading fails
            self.university_cache = {}
            self.position_cache = {}
            self.university_clusters = {}
            self.position_clusters = {}
            self.embedding_cache = {}
            
    def save_cache(self):
        try:
            university_cache_file = os.path.join(self.cache_dir, "university_cache.json")
            with open(university_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.university_cache, f, ensure_ascii=False, indent=2)
                
            position_cache_file = os.path.join(self.cache_dir, "position_cache.json")
            with open(position_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.position_cache, f, ensure_ascii=False, indent=2)
                
            university_clusters_file = os.path.join(self.cache_dir, "university_clusters.pkl")
            with open(university_clusters_file, 'wb') as f:
                pickle.dump(self.university_clusters, f)
                
            position_clusters_file = os.path.join(self.cache_dir, "position_clusters.pkl")
            with open(position_clusters_file, 'wb') as f:
                pickle.dump(self.position_clusters, f)
                
            embeddings_cache_file = os.path.join(self.cache_dir, "embeddings_cache.pkl")
            with open(embeddings_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
                
            logger.info(f"Saved standardization and embedding caches")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            
    def get_university(self, name):
        """Get standardized university name and rank from cache"""
        if not name:
            return None
        key = name.lower().strip()
        return self.university_cache.get(key)
        
    def set_university(self, name, standardized, rank):
        """Add university standardization to cache"""
        if not name:
            return
        key = name.lower().strip()
        self.university_cache[key] = [standardized, rank]
        
    def get_position(self, name):
        """Get standardized position from cache"""
        if not name:
            return None
        key = name.lower().strip()
        return self.position_cache.get(key)
        
    def set_position(self, name, standardized):
        """Add position standardization to cache"""
        if not name:
            return
        key = name.lower().strip()
        self.position_cache[key] = standardized
        
    def get_embedding(self, text):
        """Get embedding from cache"""
        if not text:
            return None
        key = text.lower().strip()
        return self.embedding_cache.get(key)
        
    def set_embedding(self, text, embedding):
        """Add embedding to cache"""
        if not text or embedding is None:
            return
        key = text.lower().strip()
        self.embedding_cache[key] = embedding

def validate_date(date_value):
    """Validate a date value and return None if invalid."""
    if not date_value:
        return None
        
    # If it's already a date or datetime object, validate it
    if isinstance(date_value, (date, datetime)):
        try:
            # Validate by accessing components - will raise if invalid
            year = date_value.year
            month = date_value.month
            day = date_value.day
            
            # Check for obviously invalid values
            if month < 1 or month > 12 or day < 1 or day > 31:
                logger.warning(f"Invalid date components in {date_value}")
                return None
                
            return date_value
        except Exception as e:
            logger.warning(f"Invalid date object {date_value}: {str(e)}")
            return None
    
    # If it's a string, try to parse it
    if isinstance(date_value, str):
        try:
            # Try multiple formats
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
                try:
                    parsed_date = datetime.strptime(date_value, fmt).date()
                    return parsed_date
                except ValueError:
                    continue
                    
            # If we get here, no format worked
            logger.warning(f"Could not parse date string {date_value}")
            return None
        except Exception as e:
            logger.warning(f"Error handling date string {date_value}: {str(e)}")
            return None
    
    # Unrecognized type
    logger.warning(f"Unrecognized date type {type(date_value)}: {date_value}")
    return None

def extract_year(date_value):
    """Extract year from a date value, with validation."""
    if not date_value:
        return None
        
    valid_date = validate_date(date_value)
    if valid_date:
        return valid_date.year
    
    # If not a valid date but might contain a year
    if isinstance(date_value, str):
        # Try to extract a 4-digit year
        year_match = re.search(r'(19\d\d|20\d\d)', date_value)
        if year_match:
            return int(year_match.group(1))
    
    return None

def build_university_clusters(universities, eps=0.3, min_samples=2):
    """Build university clusters to group similar names.
    
    Args:
        universities: List of university names
        eps: DBSCAN epsilon parameter for clustering
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Dictionary mapping cluster IDs to canonical forms
    """
    if not universities:
        return {}
    
    # Remove empty and duplicate entries
    unique_unis = list(set([u for u in universities if u and isinstance(u, str)]))
    if not unique_unis:
        return {}
    
    # Create TF-IDF vectors (character n-grams work better for typos)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(unique_unis)
    
    # Cluster similar university names
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(tfidf_matrix)
    
    # Create mapping from cluster ID to canonical university name
    cluster_to_canonical = {}
    name_to_cluster = {}
    
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1:  # Noise points
            continue
            
        name = unique_unis[i]
        name_to_cluster[name] = cluster_id
        
        if cluster_id not in cluster_to_canonical:
            cluster_to_canonical[cluster_id] = name
        else:
            # Choose the longer name as canonical (usually more descriptive)
            current = cluster_to_canonical[cluster_id]
            if len(name) > len(current):
                cluster_to_canonical[cluster_id] = name
    
    # Create reverse mapping: from each name to canonical form
    canonical_mapping = {}
    for name, cluster_id in name_to_cluster.items():
        if cluster_id in cluster_to_canonical:
            canonical_mapping[name] = cluster_to_canonical[cluster_id]
    
    return canonical_mapping

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

def generate_embedding(text, model=None, cache=None):
    """Generate embedding vector using SentenceTransformer.
    
    Args:
        text: Text to generate embedding for
        model: SentenceTransformer model
        cache: StandardizationCache instance for caching
        
    Returns:
        numpy.ndarray: Vector embedding or None if failed
    """
    if not text or not isinstance(text, str) or not model:
        return None
    
    # Check cache first if provided
    if cache:
        cached = cache.get_embedding(text)
        if cached is not None:
            return cached
    
    try:
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Cache if cache is provided
        if cache:
            cache.set_embedding(text, embedding)
            
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for '{text}': {str(e)}")
        return None

class ImprovedMigration:
    def __init__(self):
        """Initialize migration configurations."""
        self.config = load_config()
        self.mysql_conn = None
        self.pg_conn = None
        self.sentence_transformer = None
        self.cache = StandardizationCache()
        
        # Create a checkpoint file path to track progress
        self.checkpoint_file = "migration_checkpoint.txt"
        
    def get_last_processed_offset(self):
        """Read the last processed offset from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return 0
        return 0
    
    def save_checkpoint(self, offset):
        """Save the current offset to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(offset))
    
    def connect_mysql(self):
        """Connect to MySQL database."""
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.config["CLOUD_DB_HOST"],
                user=self.config["CLOUD_DB_USER"],
                password=self.config["CLOUD_DB_PASSWORD"],
                database=self.config["CLOUD_DB_NAME"],
                charset='utf8mb4',
                use_pure=True,
                connect_timeout=60,
                # Add some buffer parameters
                buffered=True,
                pool_size=5
            )
            logger.info("Connected to MySQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False
    
    def connect_postgres(self):
        """Connect to PostgreSQL database."""
        try:
            self.pg_conn = psycopg2.connect(
                host=self.config["PG_DB_HOST"],
                user=self.config["PG_DB_USER"],
                password=self.config["PG_DB_PASSWORD"],
                database=self.config["PG_DB_NAME"]
            )
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    def load_embedding_model(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """Load SentenceTransformer model for embeddings."""
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.sentence_transformer = None
            return False
    
    def close_connections(self):
        """Close database connections."""
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
        logger.info("Database connections closed")
        
        # Save cache before exiting
        self.cache.save_cache()
        logger.info("Standardization cache saved")
    
    def check_destination_schema(self):
        """Verify destination tables exist with correct structure."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = cursor.fetchall()
            cursor.close()
            
            table_names = [t[0] for t in tables]
            required_tables = ['candidate_profiles', 'candidate_experiences', 'candidate_education']
            
            missing_tables = [t for t in required_tables if t not in table_names]
            if missing_tables:
                logger.error(f"Missing required tables: {missing_tables}")
                return False
                
            logger.info("Destination schema verified successfully")
            return True
        except Exception as e:
            logger.error(f"Error checking destination schema: {str(e)}")
            return False
    
    def check_vector_columns(self):
        """Check if vector columns exist, add them if needed."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            # Check if pgvector extension is installed
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone() is None:
                logger.info("Installing pgvector extension")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check for position_vector column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_experiences' 
                AND column_name = 'position_vector'
            """)
            if cursor.fetchone() is None:
                logger.info("Adding position_vector column to candidate_experiences")
                cursor.execute("ALTER TABLE candidate_experiences ADD COLUMN position_vector vector(384);")
            
            # Check for degree_vector column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidate_education' 
                AND column_name = 'degree_vector'
            """)
            if cursor.fetchone() is None:
                logger.info("Adding degree_vector column to candidate_education")
                cursor.execute("ALTER TABLE candidate_education ADD COLUMN degree_vector vector(384);")
            
            self.pg_conn.commit()
            cursor.close()
            
            logger.info("Vector columns verified/added successfully")
            return True
        except Exception as e:
            logger.error(f"Error checking/adding vector columns: {str(e)}")
            return False
    
    def clear_destination_tables(self):
        """Clear all data from destination tables to start fresh."""
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            cursor.execute("TRUNCATE TABLE candidate_education CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_experiences CASCADE")
            cursor.execute("TRUNCATE TABLE candidate_profiles CASCADE")
            self.pg_conn.commit()
            cursor.close()
            
            logger.info("Destination tables cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing destination tables: {str(e)}")
            raise
    
    def get_total_candidates(self):
        """Get total count of candidate records to migrate."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM career_user_data")
            total = cursor.fetchone()[0]
            cursor.close()
            
            return total
        except Exception as e:
            logger.error(f"Error getting total candidates: {str(e)}")
            raise
    
    def build_clustering_data(self, sample_size=10000):
        """Build clustering data from a sample of universities and positions."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            # Get distinct universities
            cursor = self.mysql_conn.cursor()
            cursor.execute(f"""
                SELECT DISTINCT school 
                FROM career_cv_edus 
                WHERE school IS NOT NULL AND school != '' 
                LIMIT {sample_size}
            """)
            universities = [row[0] for row in cursor.fetchall() if row[0]]
            
            # Get distinct positions
            cursor.execute(f"""
                SELECT DISTINCT position 
                FROM career_cv_exprs 
                WHERE position IS NOT NULL AND position != '' 
                LIMIT {sample_size}
            """)
            positions = [row[0] for row in cursor.fetchall() if row[0]]
            
            cursor.close()
            
            # Build clustering for universities
            logger.info(f"Building clustering model for {len(universities)} universities")
            uni_clusters = build_university_clusters(universities)
            self.cache.university_clusters = uni_clusters
            
            # Build clustering for positions
            logger.info(f"Building clustering model for {len(positions)} positions")
            pos_clusters = build_university_clusters(positions, eps=0.25)  # Stricter for positions
            self.cache.position_clusters = pos_clusters
            
            # Save the cache
            self.cache.save_cache()
            
            logger.info(f"Built clustering data: {len(uni_clusters)} university clusters, {len(pos_clusters)} position clusters")
            return True
        except Exception as e:
            logger.error(f"Error building clustering data: {str(e)}")
            return False
    
    def fetch_candidate_batch(self, batch_size, offset):
        """Fetch a batch of candidate data from MySQL."""
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            cursor = self.mysql_conn.cursor(dictionary=True)
            query = """
            SELECT 
                id AS candidate_id,
                birthdate,
                firstname,
                lastname
            FROM career_user_data
            ORDER BY id
            LIMIT %s OFFSET %s
            """
            cursor.execute(query, (batch_size, offset))
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching candidate batch at offset {offset}: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_candidate_batch(batch_size, offset)
            except:
                raise
    
    def fetch_experiences(self, candidate_ids):
        """Fetch experiences for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.mysql_conn.cursor(dictionary=True)
            query = f"""
            SELECT 
                cve.cv_id AS candidate_id,
                cve.company,
                csd.title AS company_industry,
                cve.position,
                cve.start AS expr_start,
                cve.end AS expr_end
            FROM career_cv_exprs cve
            LEFT JOIN career_site_data csd ON cve.branch_id = csd.option_id AND csd.grp_id = 3
            WHERE cve.cv_id IN ({id_list})
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching experiences: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_experiences(candidate_ids)
            except:
                raise
    
    def fetch_education(self, candidate_ids):
        """Fetch education for given candidate IDs."""
        if not candidate_ids:
            return []
            
        try:
            if not self.mysql_conn:
                self.connect_mysql()
                
            # Convert IDs list to a string for IN clause
            id_list = ','.join(str(cid) for cid in candidate_ids)
            
            cursor = self.mysql_conn.cursor(dictionary=True)
            query = f"""
            SELECT 
                cv_id AS candidate_id,
                school,
                pro,
                start AS edu_start,
                end AS edu_end
            FROM career_cv_edus
            WHERE cv_id IN ({id_list})
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Error fetching education: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_mysql()
                return self.fetch_education(candidate_ids)
            except:
                raise
    
    def process_candidates(self, candidates_data):
        """Process candidate data with date validation."""
        processed = []
        
        for row in candidates_data:
            # Validate birthdate
            birthdate = validate_date(row['birthdate'])
            
            processed.append((
                row['candidate_id'],
                birthdate,
                row['firstname'],
                row['lastname']
            ))
        
        return processed
    
    def process_experiences(self, experiences_data):
        """Process experience data with standardization and embeddings."""
        processed = []
        embedding_source_data = []
        
        for row in experiences_data:
            if row['company'] and row['position']:
                # Standardize position
                std_position = standardize_position(row['position'], self.cache)
                
                # Validate dates
                start_date = validate_date(row['expr_start'])
                end_date = validate_date(row['expr_end'])
                
                candidate_id = row['candidate_id']
                processed_row = (
                    candidate_id,
                    row['company'],
                    row['company_industry'] if row['company_industry'] else None,
                    std_position,  # Use standardized position
                    start_date,
                    end_date
                )
                
                processed.append(processed_row)
                
                # Also save this for later vector generation
                if std_position:
                    embedding_source_data.append((candidate_id, std_position))
        
        return processed, embedding_source_data
    
    def process_education(self, education_data):
        """Process education data with university standardization and embeddings."""
        processed = []
        embedding_source_data = []
        
        for row in education_data:
            if row['school']:
                # Standardize school name and get university rank
                std_school, uni_rank = standardize_university(row['school'], self.cache)
                
                # Extract years from dates
                start_year = extract_year(row['edu_start'])
                end_year = extract_year(row['edu_end'])
                
                candidate_id = row['candidate_id']
                degree = row['pro'] if row['pro'] else "Unknown"
                
                processed_row = (
                    candidate_id,
                    std_school,
                    uni_rank,  # Use university rank from standardization
                    degree,
                    start_year,
                    end_year
                )
                
                processed.append(processed_row)
                
                # Also save this for later vector generation
                if degree and degree != "Unknown":
                    embedding_source_data.append((candidate_id, degree))
        
        return processed, embedding_source_data
    
    def save_candidates(self, candidates):
        """Save candidate data to PostgreSQL."""
        if not candidates:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            query = """
            INSERT INTO candidate_profiles 
            (candidate_id, birthdate, firstname, lastname)
            VALUES %s
            ON CONFLICT (candidate_id) DO UPDATE SET
                birthdate = EXCLUDED.birthdate,
                firstname = EXCLUDED.firstname,
                lastname = EXCLUDED.lastname
            """
            execute_values(cursor, query, candidates)
            self.pg_conn.commit()
            
            count = len(candidates)
            logger.info(f"Saved {count} candidates")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving candidates: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, candidates)
                self.pg_conn.commit()
                cursor.close()
                count = len(candidates)
                logger.info(f"Saved {count} candidates (retry)")
                return count
            except:
                logger.error("Failed to save candidates even after reconnection")
                return 0
    
    def save_experiences(self, experiences):
        """Save experience data to PostgreSQL."""
        if not experiences:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            query = """
            INSERT INTO candidate_experiences 
            (candidate_id, company, company_industry, position, start_date, end_date)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, experiences)
            self.pg_conn.commit()
            
            count = len(experiences)
            logger.info(f"Saved {count} experiences")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving experiences: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, experiences)
                self.pg_conn.commit()
                cursor.close()
                count = len(experiences)
                logger.info(f"Saved {count} experiences (retry)")
                return count
            except:
                logger.error("Failed to save experiences even after reconnection")
                return 0
    
    def save_education(self, education):
        """Save education data to PostgreSQL."""
        if not education:
            return 0
            
        try:
            if not self.pg_conn:
                self.connect_postgres()
                
            cursor = self.pg_conn.cursor()
            
            query = """
            INSERT INTO candidate_education 
            (candidate_id, school, university_rank, degree, start_year, end_year)
            VALUES %s
            ON CONFLICT DO NOTHING
            """
            execute_values(cursor, query, education)
            self.pg_conn.commit()
            
            count = len(education)
            logger.info(f"Saved {count} education records")
            cursor.close()
            
            return count
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error saving education: {str(e)}")
            # Try to reconnect if connection lost
            try:
                self.connect_postgres()
                cursor = self.pg_conn.cursor()
                execute_values(cursor, query, education)
                self.pg_conn.commit()
                cursor.close()
                count = len(education)
                logger.info(f"Saved {count} education records (retry)")
                return count
            except:
                logger.error("Failed to save education even after reconnection")
                return 0
    
    def update_experience_embeddings(self, data_pairs, batch_size=50):
        """Generate and update embeddings for position data.
        
        Args:
            data_pairs: List of tuples (candidate_id, text)
            batch_size: Number of items to process at once
        """
        if not data_pairs or not self.sentence_transformer:
            return 0
        
        updated_count = 0
        
        # Process in batches for efficiency
        for i in range(0, len(data_pairs), batch_size):
            batch = data_pairs[i:i+batch_size]
            
            # Generate embeddings
            embeddings_batch = []
            for candidate_id, text in batch:
                vector = generate_embedding(text, self.sentence_transformer, self.cache)
                if vector is not None:
                    embeddings_batch.append((candidate_id, vector.tolist()))
            
            # Update database
            if embeddings_batch:
                try:
                    if not self.pg_conn:
                        self.connect_postgres()
                        
                    cursor = self.pg_conn.cursor()
                    
                    for candidate_id, vector in embeddings_batch:
                        query = """
                        UPDATE candidate_experiences
                        SET position_vector = %s
                        WHERE candidate_id = %s AND position_vector IS NULL
                        """
                        cursor.execute(query, (vector, candidate_id))
                    
                    self.pg_conn.commit()
                    cursor.close()
                    
                    updated_count += len(embeddings_batch)
                    logger.info(f"Updated {len(embeddings_batch)} position embeddings")
                except Exception as e:
                    self.pg_conn.rollback()
                    logger.error(f"Error updating position embeddings: {str(e)}")
        
        return updated_count
    
    def update_education_embeddings(self, data_pairs, batch_size=50):
        """Generate and update embeddings for degree data.
        
        Args:
            data_pairs: List of tuples (candidate_id, text)
            batch_size: Number of items to process at once
        """
        if not data_pairs or not self.sentence_transformer:
            return 0
        
        updated_count = 0
        
        # Process in batches for efficiency
        for i in range(0, len(data_pairs), batch_size):
            batch = data_pairs[i:i+batch_size]
            
            # Generate embeddings
            embeddings_batch = []
            for candidate_id, text in batch:
                vector = generate_embedding(text, self.sentence_transformer, self.cache)
                if vector is not None:
                    embeddings_batch.append((candidate_id, vector.tolist()))
            
            # Update database
            if embeddings_batch:
                try:
                    if not self.pg_conn:
                        self.connect_postgres()
                        
                    cursor = self.pg_conn.cursor()
                    
                    for candidate_id, vector in embeddings_batch:
                        query = """
                        UPDATE candidate_education
                        SET degree_vector = %s
                        WHERE candidate_id = %s AND degree_vector IS NULL
                        """
                        cursor.execute(query, (vector, candidate_id))
                    
                    self.pg_conn.commit()
                    cursor.close()
                    
                    updated_count += len(embeddings_batch)
                    logger.info(f"Updated {len(embeddings_batch)} degree embeddings")
                except Exception as e:
                    self.pg_conn.rollback()
                    logger.error(f"Error updating degree embeddings: {str(e)}")
        
        return updated_count
    
    def migrate(self, batch_size=1000, max_rows=None, skip_embeddings=False, build_clusters=False):
        """Run the migration with improved standardization and embeddings."""
        start_time = time.time()
        logger.info("Starting improved data migration")
        
        # Resume from last checkpoint
        offset = self.get_last_processed_offset()
        logger.info(f"Resuming from offset: {offset}")
        
        # Establish connections
        if not self.connect_mysql():
            logger.error("Failed to connect to MySQL. Aborting migration.")
            return False
            
        if not self.connect_postgres():
            logger.error("Failed to connect to PostgreSQL. Aborting migration.")
            return False
        
        # Set up clustering models if needed
        if build_clusters:
            logger.info("Building clustering models from sample data")
            self.build_clustering_data()
        
        # Check and add vector columns if needed
        if not skip_embeddings:
            if not self.check_vector_columns():
                logger.error("Failed to set up vector columns. Aborting migration.")
                return False
                
            # Load embedding model if needed
            if not self.load_embedding_model():
                logger.warning("Failed to load embedding model. Embeddings will be skipped.")
                skip_embeddings = True
        
        total = self.get_total_candidates()
        if max_rows:
            total = min(total, max_rows)
            
        logger.info(f"Will process {total} total candidates")
        
        candidates_migrated = 0
        experiences_migrated = 0
        education_migrated = 0
        experience_embeddings_updated = 0
        education_embeddings_updated = 0
        
        try:
            # Process in batches
            for current_offset in range(offset, total, batch_size):
                batch_start = time.time()
                
                # Calculate current progress
                current_count = min(current_offset + batch_size, total)
                progress_percent = current_count / total * 100
                logger.info(f"Processing batch: {current_offset+1}-{current_count} of {total} ({progress_percent:.1f}%)")
                
                # Step 1: Fetch candidates
                fetch_start = time.time()
                logger.info(f"Fetching candidates batch at offset {current_offset}")
                candidates_batch = self.fetch_candidate_batch(batch_size, current_offset)
                fetch_time = time.time() - fetch_start
                logger.info(f"Fetched {len(candidates_batch)} candidates in {fetch_time:.2f}s")
                
                if not candidates_batch:
                    logger.warning(f"Empty candidates batch at offset {current_offset}")
                    self.save_checkpoint(current_offset + batch_size)
                    continue
                
                # Get candidate IDs for related data
                candidate_ids = [row['candidate_id'] for row in candidates_batch]
                
                # Step 2: Process and save candidates
                process_start = time.time()
                processed_candidates = self.process_candidates(candidates_batch)
                process_time = time.time() - process_start
                logger.info(f"Processed candidates in {process_time:.2f}s")
                
                save_start = time.time()
                candidates_count = self.save_candidates(processed_candidates)
                candidates_migrated += candidates_count
                save_time = time.time() - save_start
                logger.info(f"Saved candidates in {save_time:.2f}s")
                
                # Create thread pool for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Process experiences and education in parallel
                    experience_future = executor.submit(self.process_experiences_batch, candidate_ids)
                    education_future = executor.submit(self.process_education_batch, candidate_ids)
                    
                    # Wait for both to complete
                    experiences_result = experience_future.result()
                    education_result = education_future.result()
                    
                    # Update counts
                    if experiences_result:
                        experiences_migrated += experiences_result[0]
                        if not skip_embeddings:
                            experience_embeddings_updated += experiences_result[1]
                    
                    if education_result:
                        education_migrated += education_result[0]
                        if not skip_embeddings:
                            education_embeddings_updated += education_result[1]
                
                # Update checkpoint after successful processing of this batch
                self.save_checkpoint(current_offset + batch_size)
                
                # Log batch timing
                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s - Progress: {current_count}/{total} candidates")
                
                # Save cache periodically
                if current_offset % (batch_size * 10) == 0:
                    self.cache.save_cache()
            
            # Save cache one final time
            self.cache.save_cache()
                
            # Migration complete
            elapsed = time.time() - start_time
            logger.info(f"Migration completed in {elapsed:.2f} seconds")
            logger.info(f"Summary: {candidates_migrated} candidates, {experiences_migrated} experiences ({experience_embeddings_updated} with embeddings), " 
                       f"{education_migrated} education records ({education_embeddings_updated} with embeddings)")
            
            return True
        except Exception as e:
            logger.error(f"Migration error at offset {offset}: {str(e)}")
            logger.info(f"Migration can be resumed from offset {offset} using the checkpoint file")
            raise
        finally:
            self.close_connections()
    
    def process_experiences_batch(self, candidate_ids):
        """Process and save experiences for a batch of candidates."""
        try:
            # Fetch experiences
            fetch_start = time.time()
            logger.info(f"Fetching experiences for {len(candidate_ids)} candidates")
            experiences_batch = self.fetch_experiences(candidate_ids)
            fetch_time = time.time() - fetch_start
            logger.info(f"Fetched {len(experiences_batch)} experiences in {fetch_time:.2f}s")
            
            if not experiences_batch:
                return (0, 0)  # No experiences to process
            
            # Process experiences
            process_start = time.time()
            processed_experiences, exp_embedding_pairs = self.process_experiences(experiences_batch)
            process_time = time.time() - process_start
            logger.info(f"Processed experiences in {process_time:.2f}s")
            
            # Save experiences
            save_start = time.time()
            exp_count = self.save_experiences(processed_experiences)
            save_time = time.time() - save_start
            logger.info(f"Saved experiences in {save_time:.2f}s")
            
            # Update position embeddings if needed
            emb_count = 0
            if self.sentence_transformer and exp_embedding_pairs:
                emb_start = time.time()
                emb_count = self.update_experience_embeddings(exp_embedding_pairs)
                emb_time = time.time() - emb_start
                logger.info(f"Updated {emb_count} position embeddings in {emb_time:.2f}s")
            
            return (exp_count, emb_count)
        except Exception as e:
            logger.error(f"Error processing experiences batch: {str(e)}")
            return (0, 0)
    
    def process_education_batch(self, candidate_ids):
        """Process and save education for a batch of candidates."""
        try:
            # Fetch education
            fetch_start = time.time()
            logger.info(f"Fetching education for {len(candidate_ids)} candidates")
            education_batch = self.fetch_education(candidate_ids)
            fetch_time = time.time() - fetch_start
            logger.info(f"Fetched {len(education_batch)} education records in {fetch_time:.2f}s")
            
            if not education_batch:
                return (0, 0)  # No education to process
            
            # Process education
            process_start = time.time()
            processed_education, edu_embedding_pairs = self.process_education(education_batch)
            process_time = time.time() - process_start
            logger.info(f"Processed education in {process_time:.2f}s")
            
            # Save education
            save_start = time.time()
            edu_count = self.save_education(processed_education)
            save_time = time.time() - save_start
            logger.info(f"Saved education in {save_time:.2f}s")
            
            # Update degree embeddings if needed
            emb_count = 0
            if self.sentence_transformer and edu_embedding_pairs:
                emb_start = time.time()
                emb_count = self.update_education_embeddings(edu_embedding_pairs)
                emb_time = time.time() - emb_start
                logger.info(f"Updated {emb_count} degree embeddings in {emb_time:.2f}s")
            
            return (edu_count, emb_count)
        except Exception as e:
            logger.error(f"Error processing education batch: {str(e)}")
            return (0, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved MySQL to PostgreSQL Migration')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size for candidates')
    parser.add_argument('--max', type=int, default=None, help='Maximum candidates to process')
    parser.add_argument('--reset', action='store_true', help='Start fresh by clearing destination tables')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation (faster)')
    parser.add_argument('--build-clusters', action='store_true', help='Build clustering models from sample data')
    args = parser.parse_args()
    
    migration = ImprovedMigration()
    
    if args.reset:
        # Remove checkpoint file
        if os.path.exists(migration.checkpoint_file):
            os.remove(migration.checkpoint_file)
            logger.info("Removed existing checkpoint file")
        
        # Clear destination tables
        migration.clear_destination_tables()
        logger.info("Reset complete - starting migration from beginning")
    
    # Validate schema
    if not migration.check_destination_schema():
        logger.error("Schema validation failed. Please fix before proceeding.")
        sys.exit(1)
        
    # Run migration
    success = migration.migrate(
        batch_size=args.batch, 
        max_rows=args.max, 
        skip_embeddings=args.skip_embeddings,
        build_clusters=args.build_clusters
    )
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed. Check the logs for details.")
        sys.exit(1)