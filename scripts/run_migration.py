import os
import sys
import time
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger import setup_logging

# Import migration modules
from migration.database import MySQLConnector, PostgreSQLConnector
from migration.standardization import standardize_university, standardize_position
from migration.clustering import build_university_clusters
from migration.vectors import load_embedding_model, generate_embedding
from migration.utils import StandardizationCache, validate_date, extract_year
from src.utils.config import load_config
logger = setup_logging()
logger.setLevel(logging.INFO)

class MigrationManager:
    """Main class to manage the migration process."""
    
    def __init__(self):
        """Initialize the migration manager."""
        self.mysql = MySQLConnector()
        self.postgres = PostgreSQLConnector()
        self.cache = StandardizationCache()
        self.embedding_model = None
        self.embedding_dimension = 0
        
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
    
    def build_clustering_data(self, sample_size=10000):
        """Build clustering data for universities and positions."""
        logger.info("Building clustering models from sample data")
        
        # Fetch data for clustering
        universities = self.mysql.fetch_distinct_universities(sample_size)
        positions = self.mysql.fetch_distinct_positions(sample_size)
        
        # Build clustering models
        logger.info(f"Building clustering model for {len(universities)} universities")
        uni_clusters = build_university_clusters(universities)
        self.cache.university_clusters = uni_clusters
        
        logger.info(f"Building clustering model for {len(positions)} positions")
        pos_clusters = build_university_clusters(positions, eps=0.25)  # Stricter for positions
        self.cache.position_clusters = pos_clusters
        
        # Save cache
        self.cache.save_cache()
        
        logger.info(f"Built clustering data: {len(uni_clusters)} university clusters, {len(pos_clusters)} position clusters")
        return True
    
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
    
    def prepare_embeddings_for_positions(self, embedding_pairs):
        """Prepare position embeddings for database update."""
        if not self.embedding_model or not embedding_pairs:
            return []
            
        embeddings_batch = []
        for candidate_id, text in embedding_pairs:
            vector = generate_embedding(text, self.embedding_model, self.cache)
            if vector is not None:
                embeddings_batch.append((candidate_id, vector.tolist()))
                
        return embeddings_batch
    
    def prepare_embeddings_for_education(self, embedding_pairs):
        """Prepare education embeddings for database update."""
        if not self.embedding_model or not embedding_pairs:
            return []
            
        embeddings_batch = []
        for candidate_id, text in embedding_pairs:
            vector = generate_embedding(text, self.embedding_model, self.cache)
            if vector is not None:
                embeddings_batch.append((candidate_id, vector.tolist()))
                
        return embeddings_batch
    
    def run(self, batch_size=1000, max_rows=None, skip_embeddings=False, build_clusters=False, reset=False):
        """Run the migration process."""
        start_time = time.time()
        logger.info("Starting data migration")
        
        # Reset if requested
        if reset:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("Removed existing checkpoint file")
            
            self.postgres.clear_destination_tables()
            logger.info("Reset complete - starting migration from beginning")
        
        # Resume from last checkpoint
        offset = self.get_last_processed_offset()
        logger.info(f"Resuming from offset: {offset}")
        
        # Validate PostgreSQL schema
        if not self.postgres.check_destination_schema():
            logger.error("Schema validation failed. Please fix before proceeding.")
            return False
        
        # Build clustering models if needed
        if build_clusters:
            self.build_clustering_data()
        
        # Set up embedding model and vector columns if needed
        if not skip_embeddings:
            if not self.postgres.check_vector_columns():
                logger.error("Failed to set up vector columns. Aborting migration.")
                return False
                
            # Load embedding model if needed
            self.embedding_model, self.embedding_dimension = load_embedding_model()
            if not self.embedding_model:
                logger.warning("Failed to load embedding model. Embeddings will be skipped.")
                skip_embeddings = True
        
        # Get total candidates to process
        total = self.mysql.get_total_candidates()
        if max_rows:
            total = min(total, max_rows)
            
        logger.info(f"Will process {total} total candidates")
        
        # Migration statistics
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
                candidates_batch = self.mysql.fetch_candidate_batch(batch_size, current_offset)
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
                candidates_count = self.postgres.save_candidates(processed_candidates)
                candidates_migrated += candidates_count
                save_time = time.time() - save_start
                logger.info(f"Saved candidates in {save_time:.2f}s")
                
                # Step 3: Process experiences (sequentially, no threading)
                fetch_start = time.time()
                logger.info(f"Fetching experiences for {len(candidate_ids)} candidates")
                experiences_batch = self.mysql.fetch_experiences(candidate_ids)
                fetch_time = time.time() - fetch_start
                logger.info(f"Fetched {len(experiences_batch)} experiences in {fetch_time:.2f}s")
                
                if experiences_batch:
                    process_start = time.time()
                    processed_experiences, exp_embedding_pairs = self.process_experiences(experiences_batch)
                    process_time = time.time() - process_start
                    logger.info(f"Processed experiences in {process_time:.2f}s")
                    
                    save_start = time.time()
                    exp_count = self.postgres.save_experiences(processed_experiences)
                    experiences_migrated += exp_count
                    save_time = time.time() - save_start
                    logger.info(f"Saved experiences in {save_time:.2f}s")
                    
                    # Update position embeddings if needed
                    if not skip_embeddings and exp_embedding_pairs:
                        emb_start = time.time()
                        embeddings_batch = self.prepare_embeddings_for_positions(exp_embedding_pairs)
                        if embeddings_batch:
                            emb_count = self.postgres.update_experience_embeddings(embeddings_batch)
                            experience_embeddings_updated += emb_count
                            emb_time = time.time() - emb_start
                            logger.info(f"Updated {emb_count} position embeddings in {emb_time:.2f}s")
                
                # Step 4: Process education (sequentially, no threading)
                fetch_start = time.time()
                logger.info(f"Fetching education for {len(candidate_ids)} candidates")
                education_batch = self.mysql.fetch_education(candidate_ids)
                fetch_time = time.time() - fetch_start
                logger.info(f"Fetched {len(education_batch)} education records in {fetch_time:.2f}s")
                
                if education_batch:
                    process_start = time.time()
                    processed_education, edu_embedding_pairs = self.process_education(education_batch)
                    process_time = time.time() - process_start
                    logger.info(f"Processed education in {process_time:.2f}s")
                    
                    save_start = time.time()
                    edu_count = self.postgres.save_education(processed_education)
                    education_migrated += edu_count
                    save_time = time.time() - save_start
                    logger.info(f"Saved education in {save_time:.2f}s")
                    
                    # Update degree embeddings if needed
                    if not skip_embeddings and edu_embedding_pairs:
                        emb_start = time.time()
                        embeddings_batch = self.prepare_embeddings_for_education(edu_embedding_pairs)
                        if embeddings_batch:
                            emb_count = self.postgres.update_education_embeddings(embeddings_batch)
                            education_embeddings_updated += emb_count
                            emb_time = time.time() - emb_start
                            logger.info(f"Updated {emb_count} degree embeddings in {emb_time:.2f}s")
                
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
            # Close connections
            self.mysql.close()
            self.postgres.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MySQL to PostgreSQL Migration')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size for candidates')
    parser.add_argument('--max', type=int, default=None, help='Maximum candidates to process')
    parser.add_argument('--reset', action='store_true', help='Start fresh by clearing destination tables')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation (faster)')
    parser.add_argument('--build-clusters', action='store_true', help='Build clustering models from sample data')
    args = parser.parse_args()
    
    # Run migration
    migration = MigrationManager()
    success = migration.run(
        batch_size=args.batch,
        max_rows=args.max,
        skip_embeddings=args.skip_embeddings,
        build_clusters=args.build_clusters,
        reset=args.reset
    )
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed. Check the logs for details.")
        sys.exit(1)