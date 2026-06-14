import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", ".data/raw_reviews")

METADATA_URL = "https://itunes.apple.com/us/customer-reviews/id{app_id}?displayable-kind=11"
REVIEWS_URL = "https://itunes.apple.com/WebObjects/MZStore.woa/wa/userReviewsRow"

HEADERS = {
    "User-Agent": "iTunes/12.0 (Macintosh; OS X 10.15.7) AppleWebKit/605.1.15",
    "X-Apple-Store-Front": "143441-1,32",
    "Accept": "application/json",
}

REQUEST_DELAY = 0.3
BATCH_SIZE = 5
REQUEST_TIMEOUT = 10


def parse_review(raw: dict) -> dict:
    """Extract and normalize relevant fields from a raw App Store review."""
    return {
        "id": raw["userReviewId"],
        "title": (raw.get("title") or "").strip(),
        "body": (raw.get("body") or "").strip(),
        "rating": int(raw["rating"]),
        "author": (raw.get("name") or "").strip(),
        "date": raw.get("date", ""),
        "vote_count": raw.get("voteCount", 0),
        "vote_sum": raw.get("voteSum", 0),
    }


def get_app_metadata(app_id: int) -> dict:
    """
    Fetch app metadata from the US App Store.

    Returns total review count, average rating, and rating distribution.
    Raises ValueError if the app is not found or has no reviews.
    Raises RuntimeError on network or API errors.
    """
    try:
        resp = requests.get(
            METADATA_URL.format(app_id=app_id),
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not reach the App Store. Check your internet connection.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out while fetching app metadata.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"App Store returned an error: {e}")
    except ValueError:
        raise RuntimeError("Received an unexpected response from the App Store.")

    total = data.get("totalNumberOfReviews")
    if total is None:
        raise ValueError(f"App ID {app_id} was not found.")
    if total == 0:
        raise ValueError(f"App ID {app_id} has no reviews.")

    return {
        "app_id": app_id,
        "total_reviews": total,
        "rating_average": data.get("ratingAverage"),
        "rating_count": data.get("ratingCount"),
        "rating_distribution": data.get("ratingCountList", []),
    }


def fetch_review_batch(app_id: int, start_index: int) -> list[dict]:
    """
    Fetch a batch of reviews starting at the given index.

    Returns an empty list on timeout or connection errors so the
    caller can continue with the next batch without interruption.
    Retries once on HTTP 429 rate limiting.
    """
    try:
        resp = requests.get(
            REVIEWS_URL,
            headers=HEADERS,
            params={
                "id": app_id,
                "displayable-kind": 11,
                "startIndex": start_index,
                "endIndex": start_index + BATCH_SIZE - 1,
                "sort": 2,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("userReviewList", [])
    except requests.exceptions.Timeout:
        return []
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            time.sleep(5)
            return fetch_review_batch(app_id, start_index)
        raise RuntimeError(f"Failed to fetch reviews: {e}")
    except (requests.exceptions.ConnectionError, ValueError):
        return []


def collect_reviews(app_id: int, count: int = 100) -> tuple[dict, list[dict]]:
    """
    Collect a random sample of reviews for the given app.

    Fetches reviews in batches from random offsets across the full
    review pool until the requested count of unique reviews is reached.
    Returns a tuple of (metadata, reviews).
    """
    metadata = get_app_metadata(app_id)
    total = metadata["total_reviews"]
    actual_count = min(count, total)
    max_offset = max(total - BATCH_SIZE, 0)

    seen_ids: set[str] = set()
    used_offsets: set[int] = set()
    results: list[dict] = []

    while len(results) < actual_count:
        if len(used_offsets) > max_offset:
            break

        idx = random.randint(0, max_offset)
        if idx in used_offsets:
            continue
        used_offsets.add(idx)

        for raw in fetch_review_batch(app_id, idx):
            review_id = raw.get("userReviewId")
            if not review_id or review_id in seen_ids:
                continue
            if len(results) >= actual_count:
                break
            seen_ids.add(review_id)
            results.append(parse_review(raw))

        time.sleep(REQUEST_DELAY)

    return metadata, results


def save_reviews(app_id: int, metadata: dict, reviews: list[dict]) -> Path:
    """Save collected reviews and metadata to a JSON file in DATA_DIR."""
    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"reviews_{app_id}.json"
    path.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "reviews": reviews,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path