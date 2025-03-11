from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.data.db_connect import LocalDatabase
from src.model.scoring import calculate_score
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="HR Candidate Evaluation API")
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the search UI."""
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/candidates/{search_term}")
async def get_candidates(search_term: str):
    """Return top 10 candidates matching the search term, ranked by score."""
    try:
        db = LocalDatabase()
        profiles = db.fetch_profiles(search_term)
        db.close()

        candidates = {}
        for p in profiles:
            candidate_id = p[0]
            if candidate_id not in candidates:
                candidates[candidate_id] = {
                    "candidate_id": p[0],
                    "firstname": p[9],
                    "lastname": p[10],
                    "experiences": {},  # Use dict to deduplicate
                    "education": {
                        "university": p[7] if p[7] else "Unknown",
                        "university_rank": p[11] if p[11] is not None else 1,
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
                "experiences": candidate["experiences"],
                "university": candidate["education"]["university"],
                "university_rank": candidate["education"]["university_rank"],
                "degree": candidate["education"]["degree"],
                "top_score": max((e["score"] for e in candidate["experiences"]), default=0)
            })

        logger.info(f"Returned {len(result)} candidates for '{search_term}'")
        return {"candidates": result}
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise