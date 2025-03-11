# HR Candidate Evaluation

A system to preprocess candidate data from a MySQL database using GPT-4o-mini and score candidates based on HR search terms.

## Setup
1. Clone: `git clone https://github.com/yourusername/HR-Candidate-Evaluation.git`
2. Install: `pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and fill in values.
4. Initialize DB: `mysql -u root -p hr_db < src/data/schema.sql`
5. Run API: `./scripts/run_api.sh`

## Usage
- GET `/candidates/{search_term}` (e.g., `/candidates/Data Engineer`).