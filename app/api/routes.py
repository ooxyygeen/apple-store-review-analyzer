import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
import litellm

from app.services.collector import collect_reviews, save_reviews
from app.services.analyzer import analyze
from app.services.visualization import generate_visualization
from app.schemas import CollectResponse, AnalysisResponse

router = APIRouter(prefix="/reviews", tags=["reviews"])

DATA_DIR = os.getenv("DATA_DIR", ".data/raw_reviews")


@router.post("/collect", response_model=CollectResponse)
def collect(app_id: int):
    """
    Collect 100 random reviews for the specified app from the US App Store
    and save them locally for subsequent analysis.
    """
    try:
        metadata, reviews = collect_reviews(app_id)
        collected_at, _ = save_reviews(app_id, metadata, reviews)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return CollectResponse(
        app_id=app_id,
        collected_at=collected_at,
        total_reviews_in_store=metadata["total_reviews"],
        reviews_collected=len(reviews),
    )


@router.get("/insights", response_model=AnalysisResponse)
def insights(app_id: int):
    """
    Run the full analysis pipeline on previously collected reviews.

    Returns rating metrics, distribution, year-over-year trends,
    and LLM-generated actionable insights from negative reviews.
    """
    try:
        result = analyze(app_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except litellm.exceptions.ServiceUnavailableError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "LLM_PROVIDER_UNAVAILABLE",
                "message": "The AI analysis service is currently experiencing high demand. Please try again later."
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return result


@router.get("/download")
def download(app_id: int):
    """
    Download the raw collected reviews as a JSON file.
    """
    path = Path(DATA_DIR) / f"reviews_{app_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No reviews found for app {app_id}. Collect reviews first."
        )
    return FileResponse(
        path=path,
        media_type="application/json",
        filename=f"reviews_{app_id}.json",
    )


@router.get("/visualization")
def visualization(app_id: int):
    """
    Generate and return a PNG visualization of review metrics
    including rating distribution, mean rating by year, and sentiment distribution.
    """
    try:
        image_bytes = generate_visualization(app_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(content=image_bytes, media_type="image/png")