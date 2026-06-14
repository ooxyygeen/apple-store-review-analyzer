import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

from app.services.preprocessor import preprocess_for_sentiment
from app.services.sentiments import run_sentiment_analysis
from app.services.metrics import calculate_metrics

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", ".data/raw_reviews")


def generate_visualization(app_id: int) -> bytes:
    """
    Generate a PNG report with metric visualizations for the specified app.

    Loads raw reviews from disk, runs sentiment analysis and metrics
    calculation, then renders three charts in a grid:
    - Rating distribution bar chart
    - Mean rating by year line chart
    - Sentiment distribution pie chart

    Returns raw PNG bytes ready to be served as an HTTP response.
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

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in reviews_with_sentiment:
        sentiment_counts[r["sentiment"]] += 1

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"App Store Review Analysis — App ID {app_id}", fontsize=14, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Rating distribution ---
    ax1 = fig.add_subplot(gs[0])
    stars = [str(i) for i in range(1, 6)]
    counts = [metrics["distribution"][s]["count"] for s in stars]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax1.bar(stars, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Rating Distribution", fontweight="bold")
    ax1.set_xlabel("Stars")
    ax1.set_ylabel("Number of Reviews")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha="center", va="bottom", fontsize=9)

    # --- Mean rating by year ---
    ax2 = fig.add_subplot(gs[1])
    years = list(metrics["by_year"].keys())
    means = [metrics["by_year"][y]["mean_rating"] for y in years]
    ax2.plot(years, means, marker="o", color="#3498db", linewidth=2, markersize=6)
    ax2.set_title("Mean Rating by Year", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean Rating")
    ax2.set_ylim(0, 5.5)
    ax2.tick_params(axis="x", rotation=45)
    for x, y in zip(years, means):
        ax2.text(x, y + 0.15, str(y), ha="center", fontsize=9)

    # --- Sentiment distribution ---
    ax3 = fig.add_subplot(gs[2])
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    sentiment_colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
    wedges, texts, autotexts = ax3.pie(
        sizes,
        labels=labels,
        colors=sentiment_colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax3.set_title("Sentiment Distribution", fontweight="bold")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()