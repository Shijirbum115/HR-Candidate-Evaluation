import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from src.utils.logger import setup_logging

logger = setup_logging()

def load_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Load SentenceTransformer model for embeddings.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (model, embedding_dimension)
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        # Model outputs 768-dimensional vectors, which we'll pad to 1536
        embedding_dimension = 768  
        logger.info("Embedding model loaded successfully")
        return model, embedding_dimension
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        return None, 0

def generate_embedding(text, model=None, cache=None, target_dim=1536):
    """Generate embedding vector using SentenceTransformer.
    
    Args:
        text: Text to generate embedding for
        model: SentenceTransformer model
        cache: StandardizationCache instance for caching
        target_dim: Target dimension for the embedding vector
        
    Returns:
        numpy.ndarray: Vector embedding padded to target_dim or None if failed
    """
    if not text or not isinstance(text, str) or not model:
        return None
    
    # Check cache first if provided
    if cache:
        cached = cache.get_embedding(text)
        if cached is not None:
            # Ensure cached embedding has right dimension
            if len(cached) == target_dim:
                return cached
    
    try:
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Pad the embedding to target_dim if necessary
        if len(embedding) < target_dim:
            # Determine how many times to repeat the vector
            repeat_count = target_dim // len(embedding)
            remainder = target_dim % len(embedding)
            
            # Create the padded vector
            padded = np.tile(embedding, repeat_count)
            if remainder > 0:
                padded = np.concatenate([padded, embedding[:remainder]])
                
            embedding = padded
        
        # Cache if cache is provided
        if cache:
            cache.set_embedding(text, embedding)
            
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for '{text}': {str(e)}")
        return None