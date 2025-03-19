import re
import os
import json
import pickle
import logging
from datetime import datetime, date
from src.utils.logger import setup_logging

logger = setup_logging()

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
    
    # Handle integer years (like 2018)
    if isinstance(date_value, int):
        # Convert year to a date (January 1st of that year)
        try:
            if 1900 <= date_value <= 2100:  # Reasonable year range
                return date(date_value, 1, 1)
            else:
                logger.warning(f"Integer year out of reasonable range: {date_value}")
                return None
        except Exception as e:
            logger.warning(f"Failed to convert integer to date: {date_value}, {str(e)}")
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
    
    # If date_value is directly an integer year
    if isinstance(date_value, int) and 1900 <= date_value <= 2100:
        return date_value
    
    return None