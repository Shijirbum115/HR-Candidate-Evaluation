import psycopg2
from psycopg2.extras import execute_values
import pgvector.psycopg2
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = setup_logging()

class LocalPostgresDatabase:
    def __init__(self):
        config = load_config()
        self.conn = psycopg2.connect(
            host=config["PG_DB_HOST"],
            user=config["PG_DB_USER"],
            password=config["PG_DB_PASSWORD"],
            database=config["PG_DB_NAME"]
        )
        self.cursor = self.conn.cursor()
        logger.info("PostgreSQL database connection established")

    def save_candidates(self, data):
        query = """
        INSERT INTO candidate_profiles 
        (candidate_id, birthdate, firstname, lastname)
        VALUES %s
        ON CONFLICT (candidate_id) DO UPDATE SET
            birthdate = EXCLUDED.birthdate,
            firstname = EXCLUDED.firstname,
            lastname = EXCLUDED.lastname
        """
        execute_values(self.cursor, query, data)
        self.conn.commit()
        logger.info(f"Saved {len(data)} candidates")

    def save_experiences(self, data):
        query = """
        INSERT INTO candidate_experiences 
        (candidate_id, company, company_industry, position, start_date, end_date, position_vector)
        VALUES %s
        ON CONFLICT DO NOTHING
        """
        # Preprocess data to add embedding vectors for positions
        processed_data = []
        for row in data:
            # Create a copy of the row as a list to allow modification
            row_list = list(row)
            # Generate embedding for position (assuming position is at index 3)
            position_embedding = self.generate_embedding(row[3]) if row[3] else None
            # Append the embedding to the row
            row_list.append(position_embedding)
            processed_data.append(tuple(row_list))
            
        execute_values(self.cursor, query, processed_data)
        self.conn.commit()
        logger.info(f"Saved {len(data)} experiences with embeddings")

    def save_education(self, data):
        query = """
        INSERT INTO candidate_education 
        (candidate_id, school, university_rank, degree, start_year, end_year, degree_vector)
        VALUES %s
        ON CONFLICT DO NOTHING
        """
        # Preprocess data to add embedding vectors for degrees
        processed_data = []
        for row in data:
            # Create a copy of the row as a list to allow modification
            row_list = list(row)
            # Generate embedding for degree (assuming degree is at index 3)
            degree_embedding = self.generate_embedding(row[3]) if row[3] else None
            # Append the embedding to the row
            row_list.append(degree_embedding)
            processed_data.append(tuple(row_list))
            
        execute_values(self.cursor, query, processed_data)
        self.conn.commit()
        logger.info(f"Saved {len(data)} education records with embeddings")

    def generate_embedding(self, text):
        """Generate embedding vector for text using an embedding model."""
        # This is a placeholder - in a real implementation, you would call an 
        # embedding service like OpenAI or a local model
        from openai import OpenAI
        client = OpenAI(api_key=load_config()["OPENAI_API_KEY"])
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def fetch_profiles(self, search_term):
        # First, get embedding for search term
        search_embedding = self.generate_embedding(search_term)
        
        # Vector similarity search using cosine distance
        query = """
        WITH ranked_experiences AS (
            SELECT 
                cp.candidate_id, 
                cp.firstname, 
                cp.lastname,
                ce.company,
                ce.company_industry,
                ce.position,
                ce.start_date,
                ce.end_date,
                ced.school,
                ced.degree,
                ced.university_rank,
                1 - (ce.position_vector <=> %s) AS position_similarity_score
            FROM candidate_profiles cp
            JOIN candidate_experiences ce ON cp.candidate_id = ce.candidate_id
            LEFT JOIN candidate_education ced ON cp.candidate_id = ced.candidate_id
            WHERE ce.position_vector IS NOT NULL
            ORDER BY position_similarity_score DESC
            LIMIT 50
        )
        SELECT * FROM ranked_experiences
        """
        self.cursor.execute(query, (search_embedding,))
        results = self.cursor.fetchall()
        logger.debug(f"Fetched {len(results)} profiles for '{search_term}' using vector search")
        return results

    def close(self):
        self.cursor.close()
        self.conn.close()
        logger.info("PostgreSQL database connection closed")