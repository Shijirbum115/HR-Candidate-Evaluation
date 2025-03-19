from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import logging
from src.utils.logger import setup_logging

logger = setup_logging()

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