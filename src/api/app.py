# Updated src/api/app.py

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.data.db_connect import LocalDatabase
from src.model.scoring import calculate_score
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
                    "last_login_date": p[11],  # New field for last login date
                    "experiences": {},  # Use dict to deduplicate
                    "education": {
                        "university": p[7] if p[7] else "Unknown",
                        "university_rank": p[12] if p[12] is not None else 1,
                        "degree": p[8] if p[8] else "Unknown"
                    }
                }
            if p[2]:  # If company exists
                # Unique key for deduplication
                exp_key = (p[4], p[2], str(p[5]), str(p[6] or "Present"))
                if exp_key not in candidates[candidate_id]["experiences"]:
                    exp = {
                        "company": p[2],
                        "company_industry": p[3],
                        "position": p[4],
                        "start_date": str(p[5]) if p[5] else "Unknown",
                        "end_date": str(p[6]) if p[6] else "Present",
                    }
                    score = calculate_score({**candidates[candidate_id]["education"], **exp}, search_term)
                    exp["score"] = round(score, 2)
                    candidates[candidate_id]["experiences"][exp_key] = exp

        # Convert experiences dict to list and rank candidates
        ranked_candidates = []
        for candidate in candidates.values():
            candidate["experiences"] = list(candidate["experiences"].values())
            ranked_candidates.append(candidate)
        
        ranked_candidates = sorted(
            ranked_candidates,
            key=lambda x: max((e["score"] for e in x["experiences"]), default=0),
            reverse=True
        )[:10]
        
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
                "top_score": max((e["score"] for e in candidate["experiences"]), default=0)
            })

        logger.info(f"Returned {len(result)} candidates for '{search_term}' with {recency} month recency filter")
        return {"candidates": result}
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise