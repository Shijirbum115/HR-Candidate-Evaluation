# HR Candidate Evaluation System

A system to preprocess candidate data from a MySQL database, migrate it to PostgreSQL with vector capabilities, and score candidates based on HR search terms using both semantic and text-based matching.

## Features

- **Vector-based candidate search** using semantic similarity
- **Multi-language support** (English and Mongolian)
- **Data standardization** for universities, job positions, and industries
- **Candidate scoring** based on position match, experience, education, and industry
- **Interactive web interface** for easy searching
- **Recency filtering** to focus on recently active candidates

## Prerequisites

- Python 3.8+
- MySQL (source database)
- PostgreSQL 12+ with pgvector extension
- OpenAI API key (optional, for improved search term extraction)

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/HR-Candidate-Evaluation.git
   cd HR-Candidate-Evaluation
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and OpenAI API key
   ```

4. **Set up PostgreSQL with pgvector**:
   ```bash
   # Run the schema setup script
   python scripts/fix_postgres_schema.py
   ```

5. **Migrate data** (if you have existing MySQL data):
   ```bash
   python scripts/run_migration.py
   
   # Migrate last login dates
   python last_login_migrate.py
   ```

6. **Run the API server**:
   ```bash
   ./scripts/run_api.sh
   ```

7. **Access the web interface**:
   Open your browser and navigate to `http://localhost:8002/`

## Usage

### Web Interface

The web interface allows you to search for candidates with a simple search box. You can:
- Enter job titles, industries, or experience requirements
- Filter by candidate recency (1 month, 3 months, 6 months, or 1 year)
- View detailed candidate profiles with experience and education information

### API Endpoints

- **GET `/`**: Serves the search UI
- **GET `/candidates/{search_term}`**: Retrieve top 10 candidates matching the search term
  - Query parameters:
    - `recency`: Filter for candidates active in the last N months (default: 3)
  - Example: `GET /candidates/Data%20Engineer?recency=6`

## Search Examples

The system accepts a wide range of search terms in both English and Mongolian:

- `Data Engineer`
- `Finance Manager with 5+ years experience`
- `Marketing in Technology sector`
- `Борлуулагч санхүүгийн салбарт` (Sales in financial sector)
- `Дата аналист 3 жилийн туршлагатай` (Data analyst with 3 years experience)

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
isort .
flake8
```

### Building Documentation
```bash
mkdocs build
# Serve locally
mkdocs serve
```

## Project Structure

- `src/api/` - API endpoints and FastAPI application
- `src/data/` - Database connection and schema
- `src/model/` - Scoring algorithm and vector operations
- `src/templates/` - HTML templates for the web interface
- `src/utils/` - Configuration and logging utilities
- `scripts/migration/` - Data migration utilities
- `docs/` - Project documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.