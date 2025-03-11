# API Documentation

## GET /candidates/{search_term}
- **Description**: Retrieve top 10 candidates matching the search term.
- **Example**: `GET /candidates/Data Engineer`
- **Response**:
  ```json
  {
    "candidates": [
      {"id": 1, "school": "MIT", "experience": "5 years", "field": "Data Engineer", "score": 28.0}
    ]
  }