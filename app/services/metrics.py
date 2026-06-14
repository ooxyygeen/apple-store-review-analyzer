from collections import Counter
from statistics import mean, median, mode, stdev


def calculate_metrics(reviews: list[dict]) -> dict:
    """
    Calculate rating statistics and distributions from a list of reviews.

    Returns:
        - stats: mean, median, mode, standard deviation, and cumulative
          positive percentage (3 stars and above)
        - distribution: per-star count and percentage for ratings 1-5
        - by_year: review count and mean rating grouped by year, derived
          from the review date field
    """
    if not reviews:
        raise ValueError("No reviews to calculate metrics from.")

    ratings = [r["rating"] for r in reviews]
    counts = Counter(ratings)

    distribution = {
        str(star): {
            "count": counts.get(star, 0),
            "percentage": round(counts.get(star, 0) / len(ratings) * 100, 1),
        }
        for star in range(1, 6)
    }

    positive_count = sum(c for star, c in counts.items() if star >= 3)

    by_year: dict[str, list[int]] = {}
    for review in reviews:
        year = review.get("date", "")[:4]
        if not year.isdigit():
            continue
        by_year.setdefault(year, []).append(review["rating"])

    by_year_stats = {
        year: {
            "count": len(year_ratings),
            "mean_rating": round(mean(year_ratings), 2),
        }
        for year, year_ratings in sorted(by_year.items())
    }

    return {
        "stats": {
            "count": len(ratings),
            "mean": round(mean(ratings), 2),
            "median": median(ratings),
            "mode": mode(ratings),
            "std_dev": round(stdev(ratings), 2) if len(ratings) > 1 else 0.0,
            "cumulative_positive_pct": round(positive_count / len(ratings) * 100, 1),
        },
        "distribution": distribution,
        "by_year": by_year_stats,
    }