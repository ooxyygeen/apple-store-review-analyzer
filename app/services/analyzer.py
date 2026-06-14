import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.services.preprocessor import preprocess_for_sentiment, preprocess_for_keywords
from app.services.sentiments import run_sentiment_analysis
from app.services.metrics import calculate_metrics
from app.services.keywords import extract_keywords
from app.services.insights import generate_insights

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", ".data/raw_reviews")


def analyze(app_id: int) -> dict:
    """
    Run the full analysis pipeline for a previously collected app.

    Loads raw reviews from disk, runs sentiment analysis, calculates
    rating metrics, extracts keywords from negative reviews, and
    generates actionable insights via LLM.

    Raises FileNotFoundError if reviews have not been collected yet.
    Raises ValueError if the review file is empty.
    """
    path = Path(DATA_DIR) / f"reviews_{app_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"No reviews found for app {app_id}. Collect reviews first.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    reviews = data["reviews"]
    if not reviews:
        raise ValueError(f"Review file for app {app_id} is empty.")

    sentiment_texts = [preprocess_for_sentiment(r) for r in reviews]
    reviews_with_sentiment = run_sentiment_analysis(reviews, sentiment_texts)
    negatives = [r for r in reviews_with_sentiment if r["sentiment"] == "negative"]

    metrics = calculate_metrics(reviews)
    keywords = extract_keywords(negatives, preprocess_for_keywords)
    insights = generate_insights(keywords) if keywords else {}

    return {
        "app_id": app_id,
        "collected_at": data.get("collected_at"),
        "total_reviews_in_store": data.get("metadata", {}).get("total_reviews"),
        "metrics": metrics,
        "actionable_insights": insights,
    }