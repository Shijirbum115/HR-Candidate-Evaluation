# Updated src/api/app.py

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.data.db_connect import LocalDatabase
from src.model.vector_scoring import extract_from_search_term, calculate_vector_score
import logging
from typing import Optional

logger = logging.getLogger(__name__)
app = FastAPI(title="HR Candidate Evaluation API")
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the search UI."""
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/candidates/{search_term}")
async def get_candidates(
    search_term: str,
    recency: Optional[int] = Query(3, description="Filter for candidates active in the last N months")
):
    """Return top 10 candidates matching the search term, ranked by score.
    
    Args:
        search_term: Term to search for
        recency: Only include candidates active in the last N months (default: 3)
    """
    try:
        # Validate recency parameter
        if recency <= 0:
            recency = 3  # Default to 3 months
        
        # Extract search parameters once at the beginning
        job_title, industry, experience_years = extract_from_search_term(search_term)
        logger.info(f"Search parameters: job_title='{job_title}', industry='{industry}', experience_years={experience_years}")
        
        db = LocalDatabase()
        profiles = db.fetch_profiles(search_term, recency_months=recency)
        db.close()

        candidates = {}
        for p in profiles:
            candidate_id = p[0]
            if candidate_id not in candidates:
                candidates[candidate_id] = {
                    "candidate_id": p[0],
                    "firstname": p[9],
                    "lastname": p[10],
                    "last_login_date": p[11],
                    "experiences": {},  # Use dict to deduplicate
                    "education": {
                        "university": p[7] if p[7] else "Unknown",
                        "university_rank": p[12] if p[12] is not None else 4,
                        "degree": p[8] if p[8] else "Unknown"
                    }
                }
            if p[2]:  # If company exists
                # Get position vector and similarity scores
                position_vector = p[13] if len(p) > 13 else None
                vector_score = p[14] if len(p) > 14 else 0
                text_score = p[15] if len(p) > 15 else 0
                years_experience = p[16] if len(p) > 16 else 0
                position_similarity_score = p[17] if len(p) > 17 else max(vector_score, text_score)
                
                # Unique key for deduplication
                exp_key = (p[4], p[2], str(p[5]), str(p[6] or "Present"))
                if exp_key not in candidates[candidate_id]["experiences"]:
                    exp = {
                        "company": p[2],
                        "company_industry": p[3],
                        "position": p[4],
                        "start_date": str(p[5]) if p[5] else "Unknown",
                        "end_date": str(p[6]) if p[6] else "Present",
                        "position_similarity_score": position_similarity_score,
                        "years_experience": years_experience
                    }
                    # Use the pre-extracted parameters for scoring
                    score = calculate_vector_score(
                        {**candidates[candidate_id]["education"], **exp}, 
                        job_title, 
                        industry, 
                        experience_years
                    )
                    exp["score"] = round(score, 2)
                    candidates[candidate_id]["experiences"][exp_key] = exp

        # Convert experiences dict to list and rank candidates
        ranked_candidates = []
        for candidate in candidates.values():
            # Only include candidates who have at least one experience with a non-zero score
            # This ensures candidates who don't meet the experience requirement are filtered out
            experiences = list(candidate["experiences"].values())
            max_score = max((e["score"] for e in experiences), default=0)
            if max_score > 0:
                candidate["experiences"] = experiences
                candidate["max_score"] = max_score
                ranked_candidates.append(candidate)
        
        # Sort by maximum score
        ranked_candidates = sorted(
            ranked_candidates,
            key=lambda x: x["max_score"],
            reverse=True
        )[:10]  # Get top 10
        
        # Format the result
        result = []
        for rank, candidate in enumerate(ranked_candidates, 1):
            result.append({
                "rank": rank,
                "name": f"{candidate['firstname']} {candidate['lastname']}",
                "candidate_id": candidate["candidate_id"],
                "last_login_date": str(candidate["last_login_date"]) if candidate["last_login_date"] else "Unknown",
                "experiences": candidate["experiences"],
                "university": candidate["education"]["university"],
                "university_rank": candidate["education"]["university_rank"],
                "degree": candidate["education"]["degree"],
                "top_score": candidate["max_score"]
            })

        logger.info(f"Returned {len(result)} candidates for '{search_term}' with {recency} month recency filter")
        return {"candidates": result}
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise