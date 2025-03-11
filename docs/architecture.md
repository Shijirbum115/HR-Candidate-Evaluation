# HR Candidate Evaluation System Architecture

## Overview
The HR Candidate Evaluation system is designed to streamline the process of evaluating job candidates by leveraging data preprocessing, scoring algorithms, and API integration. The architecture is modular, allowing for easy maintenance and scalability.

## Components

### 1. Data Layer
- **Data Access**: The `src/data` module handles all interactions with the MySQL database. It includes:
  - `db_connect.py`: Functions for establishing connections and executing queries.
  - `schema.sql`: Defines the database schema, including tables for candidates, evaluations, and scores.

### 2. Preprocessing Layer
- **Data Preprocessing**: The `src/preprocess` module is responsible for cleaning and transforming raw candidate data. It includes:
  - `preprocess.py`: Main logic for data preprocessing.
  - `school.py`: Functions to extract educational background information.
  - `experience.py`: Functions to extract work experience details.
  - `field.py`: Functions to identify the candidate's field of work.

### 3. Scoring Layer
- **Candidate Scoring**: The `src/model` module implements the scoring logic based on predefined rules. It includes:
  - `scoring.py`: Contains the algorithms used to evaluate candidates based on their qualifications and experiences.

### 4. API Layer
- **API Integration**: The `src/api` module provides a FastAPI application for integration with external systems like PowerApps. It includes:
  - `app.py`: Sets up the FastAPI application and defines the endpoints for data retrieval and submission.

### 5. Utility Layer
- **Utilities**: The `src/utils` module contains helper functions for logging and configuration management. It includes:
  - `logger.py`: Configures logging for the application.
  - `config.py`: Manages application configuration settings.

## Testing
The `tests` directory contains unit and integration tests to ensure the reliability of the system:
- `test_data.py`: Tests for database interactions.
- `test_preprocess.py`: Tests for data preprocessing logic.
- `test_scoring.py`: Tests for the scoring model.

## Documentation
The `docs` directory includes documentation for the system architecture and API endpoints:
- `architecture.md`: Overview of the system architecture.
- `api.md`: Detailed API documentation.

## Scripts
The `scripts` directory contains utility scripts for database population and running the API:
- `populate_db.py`: Script to populate the database with sample data.
- `run_api.sh`: Shell script to run the FastAPI application.

## CI/CD
The `.github` directory contains configuration for continuous integration and deployment, including:
- `workflows/ci.yml`: Defines the CI/CD pipeline for linting and testing.
- `PULL_REQUEST_TEMPLATE.md`: Template for pull requests.

## Conclusion
This architecture provides a comprehensive framework for evaluating HR candidates efficiently, ensuring that all components work together seamlessly to deliver accurate evaluations and insights.